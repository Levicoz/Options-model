# -------------------------------------------------------------------------
# GPU-Accelerated Options Pricing Model
# -------------------------------------------------------------------------

import datetime
import logging
import time
import traceback
import argparse
import math
from typing import Optional, Dict, List, Any
import copy

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda
import yfinance as yf
import plotly.graph_objects as go
from tqdm import tqdm
import plotly.io as pio
from numpy.random import default_rng
from torch.utils.data import TensorDataset, DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
pio.renderers.default = "browser"

# -------------------------------------------------------------------------
# GPU Utilities and Device Management
# -------------------------------------------------------------------------

def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("WARNING: CUDA not available, falling back to CPU")
    return device

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Also trigger Python GC to free host side refs
    import gc as _gc
    _gc.collect()

def check_gpu_memory():
    """Check GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")

# -------------------------------------------------------------------------
# Streaming Monte Carlo with Vectorized Welford
# -------------------------------------------------------------------------

def welford_batch_update(mean, m2, n, batch):
    batch = np.asarray(batch, dtype=np.float64)
    b_n = batch.size
    if b_n == 0:
        return mean, m2, n
    
    batch_mean = batch.mean()
    batch_m2 = ((batch - batch_mean)**2).sum()
    delta = batch_mean - mean
    new_n = n + b_n
    
    if new_n == 0:
        return mean, m2, n
        
    mean_new = mean + delta * (b_n / new_n)
    m2_new = m2 + batch_m2 + delta**2 * n * b_n / new_n
    return mean_new, m2_new, new_n

def monte_carlo_price_streaming(simulator_func, total_paths, chunk_size, *sim_args, **sim_kwargs):
    n_done = 0
    mean = 0.0
    m2 = 0.0
    
    while n_done < total_paths:
        batch = min(chunk_size, total_paths - n_done)
        payoffs = simulator_func(batch, *sim_args, **sim_kwargs)
        mean, m2, n_done = welford_batch_update(mean, m2, n_done, payoffs)
    
    variance = m2 / (n_done - 1) if n_done > 1 else 0.0
    stderr = np.sqrt(variance / n_done) if n_done > 0 else 0.0
    return mean, stderr, n_done

# -------------------------------------------------------------------------
# RNG Management with Enhanced Reproducibility
# -------------------------------------------------------------------------

class RNGManager:
    def __init__(self, master_seed: int = 42):
        self.master_rng = default_rng(master_seed)
        self.master_seed = master_seed
    
    def get_child_rng(self) -> np.random.Generator:
        child_seed = self.master_rng.integers(0, 2**31-1)
        return default_rng(child_seed)
    
    def get_child_seed(self) -> int:
        return self.master_rng.integers(0, 2**31-1)

# -------------------------------------------------------------------------
# GPU-Accelerated Path Simulation Functions
# -------------------------------------------------------------------------

def simulate_bs_paths_torch(
    S0: float, r: float, T: float, sigma: float, 
    num_simulations: int, num_time_steps: int, device: torch.device
) -> torch.Tensor:
    """GPU-accelerated Black-Scholes path simulation"""
    dt = T / num_time_steps
    M = num_simulations // 2 * 2  # Ensure even number for antithetic
    
    # Initialize paths on GPU
    S = torch.full((num_time_steps + 1, M), S0, dtype=torch.float32, device=device)
    
    # Pre-compute constants
    drift = (r - 0.5 * sigma ** 2) * dt
    diffusion = sigma * torch.sqrt(torch.tensor(dt, device=device))
    
    # Generate antithetic random numbers on GPU
    Z_half = torch.randn(num_time_steps, M // 2, device=device)
    Z = torch.cat([Z_half, -Z_half], dim=1)
    
    # Vectorized path generation
    for t in range(1, num_time_steps + 1):
        S[t] = S[t - 1] * torch.exp(drift + diffusion * Z[t - 1])
    
    # Handle odd number of simulations
    if num_simulations % 2 != 0:
        S_odd = torch.full((num_time_steps + 1, 1), S0, dtype=torch.float32, device=device)
        Z_odd = torch.randn(num_time_steps, 1, device=device)
        for t in range(1, num_time_steps + 1):
            S_odd[t] = S_odd[t - 1] * torch.exp(drift + diffusion * Z_odd[t - 1])
        S = torch.cat([S, S_odd], dim=1)
    
    return S

def simulate_bs_paths_torch_bandwidth_optimized(
    S0: float, r: float, T: float, sigma: float,
    num_simulations: int, num_time_steps: int, device: torch.device
) -> torch.Tensor:
    """Bandwidth-optimized Black-Scholes simulation using larger contiguous blocks and log-space sums."""
    dt = T / max(1, num_time_steps)
    M = num_simulations
    dtype = torch.float32

    # Target ~1.5GB usage if possible to saturate memory bandwidth
    bytes_per_float = 4
    elements_per_path = (num_time_steps + 1)
    target_mem_gb = 1.5
    max_batch = max(10000, int((target_mem_gb * 1e9) / (bytes_per_float * elements_per_path)))
    effective_batch = min(M, max_batch)

    drift = (r - 0.5 * sigma * sigma) * dt
    diffusion = sigma * math.sqrt(dt)

    batches = []
    for start in range(0, M, effective_batch):
        bs = min(effective_batch, M - start)
        # Pre-generate all random numbers for this batch
        Z = torch.randn(num_time_steps, bs, device=device, dtype=dtype)
        log_S = torch.empty(num_time_steps + 1, bs, device=device, dtype=dtype)
        log_S[0].fill_(math.log(max(S0, 1e-12)))
        increments = drift + diffusion * Z
        for t in range(1, num_time_steps + 1):
            log_S[t] = log_S[t - 1] + increments[t - 1]
        S = torch.exp(log_S)
        batches.append(S)
        # cleanup per-batch
        del Z, log_S, increments, S
        torch.cuda.empty_cache()

    return torch.cat(batches, dim=1)

def simulate_heston_paths_torch(
    S0: float, r: float, T: float, v0: float, kappa: float, theta: float, 
    xi: float, rho: float, num_simulations: int, num_time_steps: int, device: torch.device
) -> torch.Tensor:
    """GPU-accelerated Heston path simulation"""
    dt = T / num_time_steps
    M = num_simulations // 2 * 2
    
    # Initialize paths on GPU
    S = torch.full((num_time_steps + 1, M), S0, dtype=torch.float32, device=device)
    v = torch.full((num_time_steps + 1, M), v0, dtype=torch.float32, device=device)
    
    # Pre-compute constants
    dt_tensor = torch.tensor(dt, device=device)
    kappa_tensor = torch.tensor(kappa, device=device)
    theta_tensor = torch.tensor(theta, device=device)
    xi_tensor = torch.tensor(xi, device=device)
    rho_tensor = torch.tensor(rho, device=device)
    sqrt_1_rho2 = torch.sqrt(1 - rho_tensor ** 2)
    
    for t in range(1, num_time_steps + 1):
        # Generate correlated random numbers with antithetic variance reduction
        z1_half = torch.randn(M // 2, device=device)
        z2_half = torch.randn(M // 2, device=device)
        z1 = torch.cat([z1_half, -z1_half])
        z2 = torch.cat([z2_half, -z2_half])
        
        w1 = z1
        w2 = rho_tensor * z1 + sqrt_1_rho2 * z2
        
        # Update variance with Feller condition
        v_prev = torch.clamp(v[t - 1], min=0)
        sqrt_v_dt = torch.sqrt(v_prev * dt_tensor)
        
        v[t] = v_prev + kappa_tensor * (theta_tensor - v_prev) * dt_tensor + xi_tensor * sqrt_v_dt * w2
        v[t] = torch.clamp(v[t], min=0)
        
        # Update stock price
        S[t] = S[t - 1] * torch.exp((r - 0.5 * v_prev) * dt_tensor + sqrt_v_dt * w1)
    
    # Handle odd number of simulations
    if num_simulations % 2 != 0:
        S_odd = torch.full((num_time_steps + 1, 1), S0, dtype=torch.float32, device=device)
        v_odd = torch.full((num_time_steps + 1, 1), v0, dtype=torch.float32, device=device)
        
        for t in range(1, num_time_steps + 1):
            z1 = torch.randn(1, device=device)
            z2 = torch.randn(1, device=device)
            w1 = z1
            w2 = rho_tensor * z1 + sqrt_1_rho2 * z2
            
            v_prev = torch.clamp(v_odd[t - 1], min=0)
            sqrt_v_dt = torch.sqrt(v_prev * dt_tensor)
            
            v_odd[t] = v_prev + kappa_tensor * (theta_tensor - v_prev) * dt_tensor + xi_tensor * sqrt_v_dt * w2
            v_odd[t] = torch.clamp(v_odd[t], min=0)
            
            S_odd[t] = S_odd[t - 1] * torch.exp((r - 0.5 * v_prev) * dt_tensor + sqrt_v_dt * w1)
        
        S = torch.cat([S, S_odd], dim=1)
    
    return S

def simulate_local_vol_paths_torch(
    S0: float, r: float, T: float, num_simulations: int, num_time_steps: int,
    iv_model, K: float, device: torch.device
) -> torch.Tensor:
    """GPU-accelerated local volatility path simulation"""
    dt = T / num_time_steps
    M = num_simulations // 2 * 2
    
    # Initialize paths on GPU
    S = torch.full((num_time_steps + 1, M), S0, dtype=torch.float32, device=device)
    
    # Pre-generate all random numbers
    Z_half = torch.randn(num_time_steps, M // 2, device=device)
    Z = torch.cat([Z_half, -Z_half], dim=1)
    
    dt_tensor = torch.tensor(dt, device=device)
    sqrt_dt = torch.sqrt(dt_tensor)
    r_tensor = torch.tensor(r, device=device)
    
    for t in range(1, num_time_steps + 1):
        S_prev = S[t - 1]
        tau_t = max(T - (t - 1) * dt, 1e-6)
        
        # Get volatilities using GPU-optimized method
        sigmas = iv_model.get_volatility_batch_torch(K, S_prev, tau_t)
        
        # Compute drift and diffusion
        drift = (r_tensor - 0.5 * sigmas**2) * dt_tensor
        diffusion = sigmas * sqrt_dt
        
        # Update paths
        S[t] = S_prev * torch.exp(drift + diffusion * Z[t - 1])
    
    # Handle odd number of simulations
    if num_simulations % 2 != 0:
        S_odd = torch.full((num_time_steps + 1, 1), S0, dtype=torch.float32, device=device)
        Z_odd = torch.randn(num_time_steps, 1, device=device)
        
        for t in range(1, num_time_steps + 1):
            S_prev = S_odd[t - 1]
            tau_t = max(T - (t - 1) * dt, 1e-6)
            sigmas = iv_model.get_volatility_batch_torch(K, S_prev, tau_t)
            drift = (r_tensor - 0.5 * sigmas**2) * dt_tensor
            diffusion = sigmas * sqrt_dt
            S_odd[t] = S_prev * torch.exp(drift + diffusion * Z_odd[t - 1])
        
        S = torch.cat([S, S_odd], dim=1)
    
    return S

# -------------------------------------------------------------------------
# Neural Networks for LSM
# -------------------------------------------------------------------------

class SingleLSMNet(nn.Module):
    """Single NN for all LSM timesteps with proper weight management"""
    def __init__(self, input_dim=7, hidden_dim=128, num_layers=3, dropout=0.1):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def create_regression_features(S, K, r, T, t_current):
    """Create features for LSM regression (NumPy)"""
    x = S / K
    tau = T - t_current
    tau_sqrt = float(np.sqrt(max(tau, 1e-6)))
    tau_col = np.full_like(x, tau_sqrt, dtype=np.float64)

    features = np.column_stack([
        np.ones_like(x, dtype=np.float64),
        x.astype(np.float64),
        (x**2).astype(np.float64),
        (x**3).astype(np.float64),
        np.maximum(x - 1, 0).astype(np.float64),
        tau_col,
        (x * tau_col).astype(np.float64)
    ])
    return features

def create_regression_features_torch(S_t: torch.Tensor, K: float, r: float, T: float, t_current: float) -> torch.Tensor:
    """
    Create LSM regression features in pure PyTorch on the same device as S_t.
    Features: [1, x, x^2, x^3, max(x-1,0), sqrt(tau), x*sqrt(tau)], where x = S/K and tau = T - t_current.
    """
    device = S_t.device
    dtype = S_t.dtype
    x = S_t / K
    tau = max(T - t_current, 1e-6)
    tau_sqrt = torch.tensor(tau, device=device, dtype=dtype).sqrt()
    one = torch.ones_like(x)
    x2 = x * x
    x3 = x2 * x
    relu = torch.clamp(x - 1.0, min=0.0)
    tau_col = torch.full_like(x, tau_sqrt.item())
    x_tau = x * tau_col
    # Stack as (N, 7)
    return torch.stack([one, x, x2, x3, relu, tau_col, x_tau], dim=1).to(dtype)

def create_regression_features_torch_vectorized(S_batch: torch.Tensor, K: float, r: float, T: float, t_current: float) -> torch.Tensor:
    """Vectorized feature creation to maximize memory bandwidth."""
    device = S_batch.device
    dtype = S_batch.dtype
    x = S_batch / K
    tau = max(T - t_current, 1e-6)
    tau_sqrt = math.sqrt(tau)
    n = S_batch.shape[0]
    feats = torch.zeros(n, 7, device=device, dtype=dtype)
    feats[:, 0] = 1.0
    feats[:, 1] = x
    feats[:, 2] = x * x
    feats[:, 3] = feats[:, 2] * x
    feats[:, 4] = torch.clamp(x - 1.0, min=0.0)
    feats[:, 5] = tau_sqrt
    feats[:, 6] = x * tau_sqrt
    return feats

# -------------------------------------------------------------------------
# Utility Classes
# -------------------------------------------------------------------------

class BlackScholesGreeks:
    @staticmethod
    def greeks(S, K, T, r, sigma, option_type='call'):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            delta = stats.norm.cdf(d1)
            theta = (-S * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                     - r * K * np.exp(-r * T) * stats.norm.cdf(d2))
            rho = K * T * np.exp(-r * T) * stats.norm.cdf(d2)
        else:
            delta = -stats.norm.cdf(-d1)
            theta = (-S * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                     + r * K * np.exp(-r * T) * stats.norm.cdf(-d2))
            rho = -K * T * np.exp(-r * T) * stats.norm.cdf(-d2)
        gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * stats.norm.pdf(d1) * np.sqrt(T)
        theta = theta / 365
        vega = vega / 100
        rho = rho / 100
        return {'Delta': delta, 'Gamma': gamma, 'Vega': vega, 'Theta': theta, 'Rho': rho}
    
    @staticmethod
    def black_scholes_price(S, K, T, r, sigma, option_type='call'):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
        
        return price

def get_live_iv(ticker: str, expiry: str, strike: float, option_type: str = "call") -> float:
    """Fetch live implied volatility from market data"""
    ticker_obj = yf.Ticker(ticker)
    try:
        available_expiries = ticker_obj.options
        if expiry not in available_expiries:
            print(f"Requested expiry {expiry} not in available expiries: {available_expiries}")
            return np.nan
        opt_chain = ticker_obj.option_chain(expiry)
        df = opt_chain.calls if option_type == "call" else opt_chain.puts
        available_strikes = df['strike'].values
        closest_idx = np.abs(available_strikes - strike).argmin()
        closest_strike = available_strikes[closest_idx]
        if abs(closest_strike - strike) > 1e-3:
            print(f"Warning: Closest strike is {closest_strike}, requested {strike}")
        iv = df.iloc[closest_idx]['impliedVolatility']
        
        if np.isnan(iv) or iv < 0.01 or iv > 2.0:
            if np.isnan(iv):
                print("IV is NaN for selected strike/expiry, falling back to historical volatility.")
            else:
                print(f"IV={iv:.4f} is out of reasonable range, falling back to historical volatility.")
            return np.nan
        else:
            print(f"Using IV={iv:.2%} for strike={closest_strike}, expiry={expiry}")
        
        return float(iv)
    except Exception as e:
        print(f"Could not fetch IV: {e}")
        return np.nan

class MarketDataFetcher:
    @staticmethod
    def get_live_quote(ticker: str, vol_window: str = "1y") -> tuple[float, float]:
        """Fetch live stock price and historical volatility"""
        data = yf.Ticker(ticker)
        hist = data.history(period="1d")
        if hist.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        S0 = hist['Close'].iloc[-1]
        hist1 = data.history(period=vol_window)
        closes = hist1['Close'].dropna()
        if len(closes) < 2:
            raise ValueError(f"Not enough history to estimate volatility for {ticker}")
        logrets = np.log(closes / closes.shift(1)).dropna()
        sigma = logrets.std() * np.sqrt(252)
        return S0, sigma

# -------------------------------------------------------------------------
# Import from NN training module with enhanced error handling
# -------------------------------------------------------------------------

try:
    from NN_training_stock_iv import run_iv_nn_training, get_sigma_iv
    NN_MODULE_AVAILABLE = True
    print("Neural Network IV module loaded successfully")
except ImportError as e:
    print(f"Warning: NN_training_stock_iv module not found: {e}")
    print("Neural network IV features will be disabled.")
    run_iv_nn_training = None
    get_sigma_iv = None
    NN_MODULE_AVAILABLE = False

# -------------------------------------------------------------------------
# Enhanced IVModel with GPU Support
# -------------------------------------------------------------------------

class IVModel:
    def __init__(self, nn_model: torch.nn.Module):
        # Ensure model is on GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn_model.to(self.device).eval()
        
        # Fixed scaler access
        if hasattr(nn_model, 'scaler') and nn_model.scaler is not None:
            self.m_scale = nn_model.scaler.m_scale
            self.tau_scale = nn_model.scaler.tau_scale
        else:
            raise ValueError("Model does not have a fitted scaler")

    def get_volatility_batch_torch(self, K: float, S_batch: torch.Tensor, tau: float) -> torch.Tensor:
        """GPU-optimized volatility batch computation"""
        tau = max(float(tau), 1e-6)
        
        # Validate inputs
        if K <= 0:
            raise ValueError(f"K must be positive, got {K}")
        if torch.any(S_batch <= 0):
            raise ValueError("All S_batch values must be positive")
        
        # Ensure S_batch is on correct device
        S_batch = S_batch.to(self.device)
        
        # Compute features on GPU
        m = torch.log(torch.clamp(torch.tensor(K, device=self.device) / torch.clamp(S_batch, min=1e-8), min=1e-8))
        m_norm = m / self.m_scale
        tau_norm = torch.full_like(m_norm, tau / self.tau_scale)
        X = torch.stack([m_norm, tau_norm], dim=1)
        
        with torch.no_grad():
            sig = self.model(X).squeeze(1).clamp_min(1e-6)
        
        return sig

    def get_volatility_batch(self, K: float, S_batch: np.ndarray, tau: float) -> np.ndarray:
        """Legacy NumPy interface for compatibility"""
        tau = max(float(tau), 1e-6)
        S_batch = np.asarray(S_batch, dtype=np.float64)
        
        # Validate inputs
        if K <= 0:
            raise ValueError(f"K must be positive, got {K}")
        if np.any(S_batch <= 0):
            raise ValueError("All S_batch values must be positive")
        
        # Convert to tensor and use GPU method
        S_tensor = torch.from_numpy(S_batch).float()
        sig_tensor = self.get_volatility_batch_torch(K, S_tensor, tau)
        result = sig_tensor.cpu().numpy().astype(np.float64)
        
        # Clear GPU memory
        del S_tensor, sig_tensor
        clear_gpu_memory()
        
        return result

# -------------------------------------------------------------------------
# GPU-Enhanced Advanced Option Pricer
# -------------------------------------------------------------------------

class AdvancedOptionPricer:
    def __init__(
        self,
        K: float,
        r: float,
        sigma: Optional[float],
        option_type: str = 'call',
        rng_manager: Optional[RNGManager] = None,
        use_heston: bool = False,
        heston_params: Optional[Dict[str, Any]] = None,
        nn_hidden: int = 128,
        nn_epochs: int = 25,
        nn_lr: float = 1e-3,
        nn_layers: int = 3,
        nn_dropout: float = 0.10,
        verbose: bool = False,
        iv_model: Optional[IVModel] = None,
        use_streaming: bool = True,
        chunk_size: int = 500,
        european_approximation: bool = False,
        use_control_variate: bool = True
    ):
        # Enable GPU perf optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.K = K
        self.r = r
        self.sigma = sigma
        self.option_type = option_type
        self.rng_manager = rng_manager or RNGManager()
        self.use_heston = use_heston
        self.heston_params = heston_params
        self.nn_hidden = nn_hidden
        self.nn_epochs = nn_epochs
        self.nn_lr = nn_lr
        self.nn_layers = nn_layers
        self.nn_dropout = nn_dropout
        self.verbose = verbose
        self.iv_model = iv_model
        self.use_streaming = use_streaming
        self.chunk_size = chunk_size
        self.european_approximation = european_approximation
        self.use_control_variate = use_control_variate
    self.device = get_device()
    # Cache a single LSM network across pricing calls
    self._lsm_net: Optional[SingleLSMNet] = None

    def _payoff_torch(self, S: torch.Tensor) -> torch.Tensor:
        """Compute payoff on GPU"""
        if self.option_type == "call":
            return torch.clamp(S - self.K, min=0)
        else:
            return torch.clamp(self.K - S, min=0)

    def price_european_gpu(
        self,
        S0: float,
        T: float,
        num_simulations: int = 10000,
        num_time_steps: int = 50
    ) -> float:
        """GPU-accelerated European option pricing"""
        # Batch size management for GPU memory
        max_paths_per_batch = min(num_simulations, 100000)
        
        total_payoff = 0.0
        total_paths = 0
        discount = math.exp(-self.r * T)
        
        # Process in batches to manage GPU memory
        for batch_start in range(0, num_simulations, max_paths_per_batch):
            batch_size = min(max_paths_per_batch, num_simulations - batch_start)
            
            # Generate paths on GPU
            if self.iv_model is not None:
                S = simulate_local_vol_paths_torch(
                    S0, self.r, T, batch_size, num_time_steps, self.iv_model, self.K, self.device
                )
            elif self.use_heston and self.heston_params is not None:
                S = simulate_heston_paths_torch(
                    S0, self.r, T,
                    self.heston_params["v0"], self.heston_params["kappa"],
                    self.heston_params["theta"], self.heston_params["xi"], self.heston_params["rho"],
                    batch_size, num_time_steps, self.device
                )
            else:
                if self.sigma is None:
                    raise ValueError("sigma is None: provide sigma, iv_model, or heston configuration")
                S = simulate_bs_paths_torch(S0, self.r, T, self.sigma, batch_size, num_time_steps, self.device)
            
            # Compute payoffs on GPU
            payoffs = self._payoff_torch(S[-1]) * discount
            
            # Accumulate results
            batch_total = torch.sum(payoffs).item()
            total_payoff += batch_total
            total_paths += batch_size
            
            # Clear GPU memory
            del S, payoffs
            clear_gpu_memory()
        
        return total_payoff / total_paths

    def price_american_enhanced_lsm_gpu(
        self,
        S0: float,
        T: float,
        num_simulations: int = 10000,
        num_time_steps: int = 50
    ) -> float:
        """GPU-accelerated American option pricing with LSM (fully on GPU)"""
        if S0 <= 0 or self.K <= 0 or T <= 0:
            raise ValueError("S0, K, T must be positive.")
        
        # Adaptive steps for short maturities
        days = max(1, int(round(T * 365)))
        if days < 10:
            num_time_steps = max(10, min(25, days * 2))

        dt = T / num_time_steps
        disc_step = math.exp(-self.r * dt)
        
        # Memory management
        max_paths = min(num_simulations, 50000)
        
        # Generate all paths on GPU
        if self.iv_model is not None:
            S = simulate_local_vol_paths_torch(
                S0, self.r, T, max_paths, num_time_steps, self.iv_model, self.K, self.device
            )
        elif self.use_heston and self.heston_params is not None:
            S = simulate_heston_paths_torch(
                S0, self.r, T,
                self.heston_params["v0"], self.heston_params["kappa"],
                self.heston_params["theta"], self.heston_params["xi"], self.heston_params["rho"],
                max_paths, num_time_steps, self.device
            )
        else:
            if self.sigma is None:
                raise ValueError("sigma is None: provide sigma, iv_model, or heston configuration")
            # Prefer bandwidth-optimized simulator
            S = simulate_bs_paths_torch_bandwidth_optimized(S0, self.r, T, self.sigma, max_paths, num_time_steps, self.device)
        
        # Initialize cashflows and exercise indicators on GPU
        cashflows = self._payoff_torch(S[-1])
        exercised = torch.zeros(max_paths, dtype=torch.bool, device=self.device)
        discount = disc_step
        
        # Collect training data for neural network (stay on GPU)
        features_all: List[torch.Tensor] = []
        targets_all: List[torch.Tensor] = []
        
        # Backward induction (first pass to build regression dataset)
        for t in range(num_time_steps - 1, 0, -1):
            cashflows *= discount
            immediate_payoff = self._payoff_torch(S[t])
            itm = (immediate_payoff > 0) & (~exercised)
            if not torch.any(itm):
                continue
            
            # Extract in-the-money paths
            S_itm = S[t, itm]                        # (N_itm,)
            cashflows_itm = cashflows[itm].unsqueeze(1)  # (N_itm, 1)
            
            # Create features on GPU (vectorized)
            feats = create_regression_features_torch_vectorized(S_itm, self.K, self.r, T, t * dt)  # (N_itm, 7)
            
            if feats.numel() > 0 and feats.shape[0] == cashflows_itm.shape[0]:
                features_all.append(feats)
                targets_all.append(cashflows_itm)
        
        if not features_all:
            return cashflows.mean().item()
        
        # Stack training data on GPU
        X_all = torch.cat(features_all, dim=0)  # (N, 7)
        Y_all = torch.cat(targets_all, dim=0)   # (N, 1)
        
        # Scale targets and normalize features on GPU
        Y_mean = Y_all.mean()
        Y_std = Y_all.std().clamp_min(1e-8)
        Y_all_scaled = (Y_all - Y_mean) / Y_std
        
        feat_mean = X_all.mean(dim=0)
        feat_std = X_all.std(dim=0)
        feat_std = torch.where(feat_std == 0, torch.ones_like(feat_std), feat_std)
        X_all_norm = (X_all - feat_mean) / feat_std
        
        # Create or reuse neural network on GPU
        if self._lsm_net is None:
            self._lsm_net = SingleLSMNet(
                input_dim=X_all.shape[1],
                hidden_dim=self.nn_hidden,
                num_layers=self.nn_layers,
                dropout=self.nn_dropout
            ).to(self.device)
        net = self._lsm_net

        dataset = TensorDataset(X_all_norm, Y_all_scaled)
        # Larger batch sizes to utilize bandwidth (~256MB target)
        elems_per_sample = X_all_norm.shape[1] + 1
        target_bytes = 256 * 1024 * 1024
        bs = min(len(dataset), max(512, min(8192, int(target_bytes / (elems_per_sample * 4)))))
        loader = DataLoader(dataset, batch_size=bs, shuffle=True, pin_memory=torch.cuda.is_available())
        
        opt = optim.AdamW(net.parameters(), lr=self.nn_lr * (2.0 if days < 10 else 1.0), weight_decay=1e-4)
        
        # Training loop
        best_loss = float('inf')
        best_state_dict = None
        patience_counter = 0
        loss_fn = nn.MSELoss()
        
        # Adaptive epochs for short maturities
        max_epochs = self.nn_epochs if days >= 10 else max(5, min(self.nn_epochs // 2, 15))
        if self.verbose:
            print(f"LSM training: epochs={max_epochs}, batch_size={bs}, dataset={len(dataset)}")
            if torch.cuda.is_available():
                print(f"GPU allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB, cached: {torch.cuda.memory_reserved()/1e9:.2f}GB")

        for epoch in range(max_epochs):
            total_loss = 0.0
            for batch_X, batch_y in loader:
                # Tensors already on GPU
                pred = net(batch_X)
                loss = loss_fn(pred, batch_y)
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / max(1, len(loader))
            
            if avg_loss < best_loss - 1e-6:
                best_loss = avg_loss
                best_state_dict = copy.deepcopy(net.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 3:
                    break
        
        # Load best weights
        if best_state_dict is not None:
            net.load_state_dict(best_state_dict)
        
        # Second pass: exercise decisions on GPU
        cashflows = self._payoff_torch(S[-1])
        exercised = torch.zeros(max_paths, dtype=torch.bool, device=self.device)
        
        for t in range(num_time_steps - 1, 0, -1):
            cashflows *= discount
            immediate_payoff = self._payoff_torch(S[t])
            itm = (immediate_payoff > 0) & (~exercised)
            if not torch.any(itm):
                continue
            
            S_itm = S[t, itm]
            
            # Create features and predict continuation values on GPU
            feats = create_regression_features_torch(S_itm, self.K, self.r, T, t * dt)
            feats_norm = (feats - feat_mean) / feat_std
            
            with torch.no_grad():
                continuation_scaled = net(feats_norm).squeeze(1)        # (N_itm,)
                continuation = continuation_scaled * Y_std + Y_mean     # (N_itm,)
            
            # Exercise decision (GPU)
            immediate_itm = immediate_payoff[itm]                       # (N_itm,)
            to_exercise = immediate_itm > continuation                  # bool mask
            if not torch.any(to_exercise):
                continue
            
            # Update cashflows and exercise status
            idx_itm = torch.where(itm)[0]
            ex_idx = idx_itm[to_exercise]
            cashflows[ex_idx] = immediate_payoff[ex_idx]
            exercised[ex_idx] = True
        
        result = cashflows.mean().item()
        
        # Clean up GPU memory
        del S, cashflows, exercised, X_all, Y_all, X_all_norm, Y_all_scaled
        clear_gpu_memory()
        
        return result

    def price_american_with_control_variate(
        self,
        S0: float,
        T: float,
        num_simulations: int = 10000,
        num_time_steps: int = 50
    ) -> float:
        """American pricing with control variate"""
        american_price = self.price_american_enhanced_lsm_gpu(S0, T, num_simulations, num_time_steps)
        
        if not self.use_control_variate or self.sigma is None:
            return american_price
        
        european_mc = self.price_european_gpu(S0, T, num_simulations, num_time_steps)
        european_analytical = BlackScholesGreeks.black_scholes_price(
            S0, self.K, T, self.r, self.sigma, self.option_type
        )
        
        beta = 1.0
        american_cv = american_price + beta * (european_analytical - european_mc)
        
        if self.verbose:
            print(f"American: {american_price:.4f}, European MC: {european_mc:.4f}, "
                  f"European Analytical: {european_analytical:.4f}, CV Adjusted: {american_cv:.4f}")
        
        return american_cv

    def price_american_option(
        self,
        S0: float,
        T: float,
        num_simulations: int = 10000,
        num_time_steps: int = 50,
        plot_paths: bool = False
    ) -> float:
        """Main American option pricing method"""
        if self.european_approximation:
            if self.verbose:
                print("WARNING: Using European approximation for American option")
            return self.price_european_gpu(S0, T, num_simulations, num_time_steps)

        if self.use_control_variate and self.sigma is not None:
            return self.price_american_with_control_variate(S0, T, num_simulations, num_time_steps)
        else:
            return self.price_american_enhanced_lsm_gpu(S0, T, num_simulations, num_time_steps)

    def compute_curve_for_S0(
        self,
        S0: float,
        intervals_per_day: int,
        total_points: int,
        num_simulations: int,
        plot_paths: bool
    ) -> List[Dict[str, Any]]:
        """Compute option value curve for a given S0"""
        records = []
        for i in range(total_points, 0, -1):
            d = i / intervals_per_day
            T = d / 365
            steps = max(10, min(130, int(np.ceil(d))))
            
            est_price = self.price_american_option(S0, T, num_simulations, steps, plot_paths)
            records.append({'S0': S0, 'Days to Expiry': d, 'Option Value': est_price})
        return records

# -------------------------------------------------------------------------
# Worker Function (kept simple; used sequentially)
# -------------------------------------------------------------------------

def compute_curve_worker_gpu(
    S0, K, r, sigma, option_type, worker_seed,
    intervals_per_day, total_points, num_simulations, plot_paths, use_heston, heston_params,
    nn_hidden=128, nn_epochs=25, nn_lr=1e-3, nn_layers=3, nn_dropout=0.10,
    verbose=False, european_approximation=False, use_control_variate=True, iv_model=None
):
    """GPU-optimized worker function"""
    try:
        # Set seeds for reproducibility
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        
        rng_manager = RNGManager(worker_seed)
        pricer = AdvancedOptionPricer(
            K, r, sigma, option_type, rng_manager, use_heston, heston_params,
            nn_hidden=nn_hidden, nn_epochs=nn_epochs, nn_lr=nn_lr, nn_layers=nn_layers,
            nn_dropout=nn_dropout, verbose=verbose, iv_model=iv_model,
            european_approximation=european_approximation, use_control_variate=use_control_variate
        )
        return pricer.compute_curve_for_S0(S0, intervals_per_day, total_points, num_simulations, plot_paths)
    except Exception as e:
        logging.error(f"Error in GPU worker for S0={S0}: {e}")
        return []

def compute_multiple_S0_gpu_batch(
    s0_list: List[float], K: float, r: float, sigma: Optional[float], option_type: str,
    intervals_per_day: int, total_points: int, num_simulations: int,
    use_heston: bool = False, heston_params: Optional[Dict[str, Any]] = None,
    nn_hidden: int = 128, nn_epochs: int = 25, nn_lr: float = 1e-3, nn_layers: int = 3, nn_dropout: float = 0.10,
    verbose: bool = False, european_approximation: bool = False, use_control_variate: bool = True,
    iv_model: Optional[IVModel] = None, seed: int = 42
):
    """Reuse a single pricer across many S0 values and periodically free memory."""
    rng_mgr = RNGManager(seed)
    pricer = AdvancedOptionPricer(
        K, r, sigma, option_type, rng_mgr, use_heston, heston_params,
        nn_hidden=nn_hidden, nn_epochs=nn_epochs, nn_lr=nn_lr, nn_layers=nn_layers,
        nn_dropout=nn_dropout, verbose=verbose, iv_model=iv_model,
        european_approximation=european_approximation, use_control_variate=use_control_variate
    )
    all_records: List[Dict[str, Any]] = []
    for i, S0 in enumerate(s0_list, 1):
        recs = pricer.compute_curve_for_S0(S0, intervals_per_day, total_points, num_simulations, False)
        all_records.extend(recs)
        if i % 2 == 0:
            clear_gpu_memory()
    return all_records

# -------------------------------------------------------------------------
# Plotting Functions
# -------------------------------------------------------------------------

def plot_option_curves(df, s0_list, S0_live, K, sigma, r, option_type, ticker, model_name):
    """Plot option value curves"""
    fig = go.Figure()
    for S0 in s0_list:
        curve = df[df['S0'] == S0]
        is_live = math.isclose(float(S0), float(S0_live), rel_tol=1e-3)
        fig.add_trace(go.Scatter(
            x=curve['Days to Expiry'],
            y=curve['Option Value'],
            mode='lines',
            name=f"S0 = ${S0}" + (" (Live)" if is_live else ""),
            line=dict(width=4 if is_live else 2, dash='solid' if is_live else 'dot'),
            hovertemplate=(
                'S0: $%{text}<br>'
                'Days to Expiry: %{x:.2f}<br>'
                'Option Value: %{y:.4f}<extra></extra>'
            ),
            text=[S0]*len(curve)
        ))

    fig.update_layout(
        title=dict(
            text=f"{model_name} American {option_type.capitalize()} Option Value vs. Days to Expiry"
                 f"<br><sup>{ticker} | K=${K} | σ={sigma:.2f} | r={r:.2%}</sup>",
            x=0.5,
            xanchor='center'
        ),
        legend=dict(
            title="Spot Price (S0)",
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.5)"
        ),
        template="plotly_white",
        dragmode='pan'
    )
    fig.update_xaxes(
        title="Days to Expiry",
        autorange='reversed',
        showgrid=True,
        ticks="outside",
        tick0=0,
        dtick=1,
        showline=True,
        linewidth=2,
        linecolor='black'
    )
    fig.update_yaxes(
        title="Option Value",
        showgrid=True,
        ticks="outside",
        showline=True,
        linewidth=2,
        linecolor='black'
    )
    fig.show()

def plot_option_curves_nn(df, s0_list, S0_live, K, ivs_for_plot, r, option_type, ticker, model_name):
    """Plot option curves with neural network IV surface"""
    fig = go.Figure()
    for idx, S0 in enumerate(s0_list):
        curve = df[df['S0'] == S0]
        iv_label = f"IV={ivs_for_plot[idx]:.2%}"
        is_live = math.isclose(float(S0), float(S0_live), rel_tol=1e-3)
        fig.add_trace(go.Scatter(
            x=curve['Days to Expiry'],
            y=curve['Option Value'],
            mode='lines',
            name=f"S0 = ${S0} ({iv_label})" + (" (Live)" if is_live else ""),
            line=dict(width=4 if is_live else 2, dash='solid' if is_live else 'dot'),
            hovertemplate=(
                'S0: $%{text}<br>'
                'Days to Expiry: %{x:.2f}<br>'
                'Option Value: %{y:.4f}<br>'
                f'{iv_label}<extra></extra>'
            ),
            text=[S0]*len(curve)
        ))

    fig.update_layout(
        title=dict(
            text=f"{model_name} American {option_type.capitalize()} Option Value vs. Days to Expiry"
                 f"<br><sup>{ticker} | K=${K} | r={r:.2%} | Neural Network IV Surface</sup>",
            x=0.5,
            xanchor='center'
        ),
        legend=dict(
            title="Spot Price (S0)",
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.5)"
        ),
        template="plotly_white",
        dragmode='pan'
    )
    fig.update_xaxes(
        title="Days to Expiry",
        autorange='reversed',
        showgrid=True,
        ticks="outside",
        tick0=0,
        dtick=1,
        linewidth=2,
        linecolor='black'
    )
    fig.update_yaxes(
        title="Option Value",
        showgrid=True,
        ticks="outside",
        showline=True,
        linewidth=2,
        linecolor='black'
    )
    fig.show()

# -------------------------------------------------------------------------
# Input Functions
# -------------------------------------------------------------------------

def get_user_inputs():
    """Get user inputs for option pricing"""
    parser = argparse.ArgumentParser(description='GPU-Accelerated Options Pricing')
    parser.add_argument('--ticker', type=str, help='Stock ticker')
    parser.add_argument('--expiry', type=str, help='Expiry date (YYYY-MM-DD)')
    parser.add_argument('--strike', type=float, help='Strike price')
    parser.add_argument('--rate', type=float, help='Risk-free rate')
    parser.add_argument('--option_type', type=str, choices=['call', 'put'], help='Option type')
    parser.add_argument('--num_sims', type=int, help='Number of simulations')
    parser.add_argument('--time_steps', type=int, help='Number of time steps')
    parser.add_argument('--s0_min', type=float, help='Minimum S0 for grid')
    parser.add_argument('--s0_max', type=float, help='Maximum S0 for grid')
    parser.add_argument('--s0_step', type=float, help='S0 grid step size')
    parser.add_argument('--model', type=str, choices=['bs', 'heston', 'both'], help='Model type')
    parser.add_argument('--iv_source', type=str, choices=['hist', 'live', 'nn'], help='IV source')
    parser.add_argument('--plot', type=str, choices=['y', 'n'], help='Show plots')
    # NN hyperparameters keep defaults so they don’t force interactive mode off
    parser.add_argument('--nn_hidden', type=int, default=128, help='NN hidden layer size')
    parser.add_argument('--nn_layers', type=int, default=3, help='NN number of layers')
    parser.add_argument('--nn_dropout', type=float, default=0.10, help='NN dropout')
    parser.add_argument('--nn_epochs', type=int, default=25, help='NN training epochs')
    parser.add_argument('--nn_lr', type=float, default=1e-3, help='NN learning rate')

    args = parser.parse_args()
    args_dict = vars(args).copy()

    # Core args that decide whether we need interactive prompts
    required_keys = [
        'ticker', 'expiry', 'strike', 'rate', 'option_type',
        'num_sims', 'time_steps', 's0_min', 's0_max', 's0_step',
        'model', 'iv_source', 'plot'
    ]

    def prompt_missing(key: str):
        """Prompt only for missing required values with proper types/choices."""
        prompts = {
            'ticker': ("Stock ticker: ", str, None),
            'expiry': ("Expiry date (YYYY-MM-DD): ", str, None),
            'strike': ("Strike price: ", float, None),
            'rate': ("Risk-free rate (decimal): ", float, None),
            'option_type': ("Option type (call/put): ", str, ['call', 'put']),
            'num_sims': ("Number of simulations: ", int, None),
            'time_steps': ("Number of time steps: ", int, None),
            's0_min': ("S0 grid minimum: ", float, None),
            's0_max': ("S0 grid maximum: ", float, None),
            's0_step': ("S0 grid step: ", float, None),
            'model': ("Model (bs/heston/both): ", str, ['bs', 'heston', 'both']),
            'iv_source': ("IV source (hist/live/nn): ", str, ['hist', 'live', 'nn']),
            'plot': ("Show plots (y/n): ", str, ['y', 'n']),
        }
        msg, cast, choices = prompts[key]
        while True:
            raw = input(msg).strip()
            try:
                val = cast(raw)
                if isinstance(val, str):
                    val = val.lower()
                if choices and val not in choices:
                    print(f"Please enter one of: {choices}")
                    continue
                return val
            except Exception:
                print(f"Invalid value for {key}. Please try again.")

    # If none of the core args were provided, collect them all interactively
    interactive_mode = not any(args_dict.get(k) is not None for k in required_keys)
    if interactive_mode:
        args_dict['ticker'] = input("Stock ticker: ").strip().upper()
        args_dict['expiry'] = input("Expiry date (YYYY-MM-DD): ").strip()
        args_dict['strike'] = float(input("Strike price: "))
        args_dict['rate'] = float(input("Risk-free rate (decimal): "))
        args_dict['option_type'] = input("Option type (call/put): ").strip().lower()
        args_dict['num_sims'] = int(input("Number of simulations: "))
        args_dict['time_steps'] = int(input("Number of time steps: "))
        args_dict['s0_min'] = float(input("S0 grid minimum: "))
        args_dict['s0_max'] = float(input("S0 grid maximum: "))
        args_dict['s0_step'] = float(input("S0 grid step: "))
        args_dict['model'] = input("Model (bs/heston/both): ").strip().lower()
        args_dict['iv_source'] = input("IV source (hist/live/nn): ").strip().lower()
        args_dict['plot'] = input("Show plots (y/n): ").strip().lower()
        # NN params with defaults and optional prompts
        nn_hidden = input("NN hidden layer size [128]: ").strip() or "128"
        nn_layers = input("NN number of layers [3]: ").strip() or "3"
        nn_dropout = input("NN dropout [0.10]: ").strip() or "0.10"
        nn_epochs = input("NN training epochs [25]: ").strip() or "25"
        nn_lr = input("NN learning rate [0.001]: ").strip() or "0.001"
        args_dict['nn_hidden'] = int(nn_hidden)
        args_dict['nn_layers'] = int(nn_layers)
        args_dict['nn_dropout'] = float(nn_dropout)
        args_dict['nn_epochs'] = int(nn_epochs)
        args_dict['nn_lr'] = float(nn_lr)
        return args_dict

    # Otherwise, prompt only for any missing required values
    for k in required_keys:
        if args_dict.get(k) is None:
            args_dict[k] = prompt_missing(k)

    # Normalize string fields
    args_dict['ticker'] = args_dict['ticker'].upper()
    args_dict['option_type'] = args_dict['option_type'].lower()
    args_dict['model'] = args_dict['model'].lower()
    args_dict['iv_source'] = args_dict['iv_source'].lower()
    args_dict['plot'] = args_dict['plot'].lower()

    return args_dict

# -------------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------------

def main() -> None:
    """Main execution function"""
    print("=== GPU-Accelerated American Option Pricer ===")
    print("This version requires CUDA-compatible GPU for optimal performance")
    
    # Check GPU availability
    device = get_device()
    check_gpu_memory()
    
    try:
        # Optional: better CUDA seed reproducibility
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        # Get user inputs
        args = get_user_inputs()
        
        # Extract parameters
        ticker = args['ticker']
        expiry = args['expiry']
        K = args['strike']
        r = args['rate']
        option_type = args['option_type']
        num_simulations = args['num_sims']
        num_time_steps = args['time_steps']
        s0_min = args['s0_min']
        s0_max = args['s0_max'] 
        s0_step = args['s0_step']
        model = args['model']
        iv_source = args['iv_source']
        show_plots = args['plot'] == 'y'
        # NN params
        nn_hidden = args.get('nn_hidden', 128)
        nn_layers = args.get('nn_layers', 3)
        nn_dropout = args.get('nn_dropout', 0.10)
        nn_epochs = args.get('nn_epochs', 25)
        nn_lr = args.get('nn_lr', 1e-3)
        
        # Parse models to run
        run_bs = model in ['bs', 'both']
        run_heston = model in ['heston', 'both']
        
        # Parse IV source
        use_nn_iv = iv_source == 'nn'
        use_live_iv = iv_source == 'live'
        
        # Get market data
        print(f"\nFetching market data for {ticker}...")
        S0_live, sigma_live = MarketDataFetcher.get_live_quote(ticker)
        print(f"Current price: ${S0_live:.2f}, Historical volatility: {sigma_live:.2%}")
        
        # Calculate days to expiry
        expiry_date = datetime.datetime.strptime(expiry, "%Y-%m-%d").date()
        today = datetime.date.today()
        days_to_expiry = (expiry_date - today).days
        print(f"Days to expiry: {days_to_expiry}")
        
        if days_to_expiry <= 0:
            print("Error: Option has already expired!")
            return
        
        # Get volatility
        sigma = sigma_live
        iv_model = None
        nn_iv_surface = None
        
        if use_live_iv:
            live_iv = get_live_iv(ticker, expiry, K, option_type)
            if not np.isnan(live_iv):
                sigma = live_iv
                print(f"Using live IV: {sigma:.2%}")
            else:
                print("Falling back to historical volatility")
                
        elif use_nn_iv:
            if not NN_MODULE_AVAILABLE:
                print("Error: Neural network IV features are not available.")
                print("Falling back to historical volatility...")
                use_nn_iv = False
                sigma = sigma_live
            else:
                print("Training neural network for IV surface...")
                try:
                    model_nn_iv, best_val_loss, df_iv, S0_nn = run_iv_nn_training(
                        ticker=ticker, epochs=100, batch_size=128, lr=0.001,
                        lambda_K=0.001, hidden_dim=64, num_hidden_layers=4,
                        dropout=0.1, use_bn=False, device_str=None, save_path=None,
                        plot_fit=True, patience=10, epsilon=0.0001
                    )
                    T_live = max(days_to_expiry, 1) / 365
                    sigma = get_sigma_iv(model_nn_iv, K, S0_live, T_live)
                    logging.info(f"Using NN-predicted starting IV at S0_live: {sigma:.2%}")
                    nn_iv_surface = model_nn_iv
                    iv_model = IVModel(model_nn_iv)
                except Exception as e:
                    logging.error(f"Neural network training failed: {e}")
                    logging.info("Falling back to historical volatility")
                    traceback.print_exc()
                    use_nn_iv = False
                    iv_model = None
                    sigma = sigma_live
        
        print(f"Using volatility: {sigma:.2%}")
        
        # Create S0 grid
        s0_list = list(np.arange(s0_min, s0_max + s0_step, s0_step))
        if S0_live not in s0_list:
            s0_list.append(S0_live)
        s0_list = sorted(set(s0_list))
        
        print(f"S0 grid: {s0_list}")
        
        # Setup parameters
        master_rng = RNGManager(42)
        intervals_per_day = 4
        total_points = days_to_expiry * intervals_per_day
        
        heston_params = {
            "v0": sigma ** 2,
            "kappa": 2.0,
            "theta": sigma ** 2,
            "xi": 0.3,
            "rho": -0.7
        } if run_heston else None
        
        # Run Black-Scholes model (sequential, GPU)
        if run_bs:
            print(f"\n=== Running Black-Scholes Model (GPU) ===")
            start_time = time.time()
            all_records_bs = compute_multiple_S0_gpu_batch(
                s0_list, K, r, sigma, option_type, intervals_per_day, total_points, num_simulations,
                use_heston=False, heston_params=None,
                nn_hidden=nn_hidden, nn_epochs=nn_epochs, nn_lr=nn_lr, nn_layers=nn_layers, nn_dropout=nn_dropout,
                verbose=False, european_approximation=False, use_control_variate=True, iv_model=None, seed=master_rng.get_child_seed()
            )
            df_bs = pd.DataFrame(all_records_bs)
            bs_time = time.time() - start_time
            print(f"Black-Scholes completed in {bs_time:.2f} seconds")
            
            if show_plots:
                plot_option_curves(df_bs, s0_list, S0_live, K, sigma, r, option_type, ticker, "Black-Scholes (GPU)")
        
        # Run Heston model (sequential, GPU)
        if run_heston:
            print(f"\n=== Running Heston Model (GPU) ===")
            start_time = time.time()
            all_records_heston = []
            for S0 in tqdm(s0_list, desc="Heston Pricing"):
                records = compute_curve_worker_gpu(
                    S0, K, r, sigma, option_type, master_rng.get_child_seed(),
                    intervals_per_day, total_points, num_simulations, False, True, heston_params,
                    nn_hidden=nn_hidden, nn_epochs=nn_epochs, nn_lr=nn_lr,
                    nn_layers=nn_layers, nn_dropout=nn_dropout,
                    verbose=False, european_approximation=False, use_control_variate=True,
                    iv_model=None
                )
                all_records_heston.extend(records)
            df_heston = pd.DataFrame(all_records_heston)
            heston_time = time.time() - start_time
            print(f"Heston completed in {heston_time:.2f} seconds")
            
            if show_plots:
                plot_option_curves(df_heston, s0_list, S0_live, K, sigma, r, option_type, ticker, "Heston (GPU)")
        
        # Run Neural Network IV model (sequential, GPU)
        if use_nn_iv and iv_model is not None:
            print(f"\n=== Running Neural Network IV Model (GPU) ===")
            start_time = time.time()
            # Get IVs for plotting
            ivs_for_plot = []
            T_plot = days_to_expiry / 365
            for S0 in s0_list:
                try:
                    iv_val = get_sigma_iv(nn_iv_surface, K, S0, T_plot)
                    ivs_for_plot.append(iv_val)
                except:
                    ivs_for_plot.append(sigma)
            
            all_records_nn = []
            for S0 in tqdm(s0_list, desc="NN IV Pricing"):
                records = compute_curve_worker_gpu(
                    S0, K, r, None, option_type, master_rng.get_child_seed(),
                    intervals_per_day, total_points, num_simulations, False, False, None,
                    nn_hidden=nn_hidden, nn_epochs=nn_epochs, nn_lr=nn_lr,
                    nn_layers=nn_layers, nn_dropout=nn_dropout,
                    verbose=False, european_approximation=False, use_control_variate=False,
                    iv_model=iv_model
                )
                all_records_nn.extend(records)
            df_nn = pd.DataFrame(all_records_nn)
            nn_time = time.time() - start_time
            print(f"Neural Network IV completed in {nn_time:.2f} seconds")
            
            if show_plots:
                plot_option_curves_nn(df_nn, s0_list, S0_live, K, ivs_for_plot, r, option_type, ticker, "Neural Network IV (GPU)")
        
        print("\n=== GPU Pricing Complete ===")
        check_gpu_memory()
        
    except Exception as e:
        logging.error(f"Fatal error in main: {e}")
        traceback.print_exc()
    finally:
        # Final GPU memory cleanup
        clear_gpu_memory()

if __name__ == "__main__":
    main()
