# -------------------------------------------------------------------------
# Imports and Configuration
# -------------------------------------------------------------------------

import datetime
import logging
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Dict, List, Any
import copy

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
import plotly.graph_objects as go
from tqdm import tqdm
import plotly.io as pio
from numpy.random import default_rng
from torch.utils.data import TensorDataset, DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
pio.renderers.default = "browser"

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
# Neural Networks for LSM with Best Weight Saving
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
    x = S / K
    tau = T - t_current
    # Expand tau to a column matching x
    tau_sqrt = float(np.sqrt(max(tau, 1e-6)))
    tau_col = np.full_like(x, tau_sqrt, dtype=np.float64)

    features = np.column_stack([
        np.ones_like(x, dtype=np.float64),
        x.astype(np.float64),
        (x**2).astype(np.float64),
        (x**3).astype(np.float64),
        np.maximum(x - 1, 0).astype(np.float64),
        tau_col,                      # was scalar -> now vector
        (x * tau_col).astype(np.float64)
    ])
    return features

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
# Path Simulation with Enhanced Precision
# -------------------------------------------------------------------------

def simulate_heston_paths_antithetic(
    S0: float, r: float, T: float, v0: float, kappa: float, theta: float, xi: float, rho: float,
    num_simulations: int, num_time_steps: int, rng: np.random.Generator
) -> np.ndarray:
    dt = T / num_time_steps
    M = num_simulations // 2 * 2
    S = np.zeros((num_time_steps + 1, M), dtype=np.float64)
    v = np.zeros((num_time_steps + 1, M), dtype=np.float64)
    S[0] = S0
    v[0] = v0
    
    for t in range(1, num_time_steps + 1):
        z1_half = rng.standard_normal(M // 2)
        z2_half = rng.standard_normal(M // 2)
        z1 = np.concatenate([z1_half, -z1_half])
        z2 = np.concatenate([z2_half, -z2_half])
        
        w1 = z1
        w2 = rho * z1 + np.sqrt(1 - rho ** 2) * z2
        v_prev = np.maximum(v[t - 1], 0)
        v[t] = v_prev + kappa * (theta - v_prev) * dt + xi * np.sqrt(v_prev * dt) * w2
        v[t] = np.maximum(v[t], 0)
        S[t] = S[t - 1] * np.exp((r - 0.5 * v_prev) * dt + np.sqrt(v_prev * dt) * w1)
    
    if num_simulations % 2 != 0:
        S_odd = np.zeros((num_time_steps + 1, 1), dtype=np.float64)
        v_odd = np.zeros((num_time_steps + 1, 1), dtype=np.float64)
        S_odd[0] = S0
        v_odd[0] = v0
        for t in range(1, num_time_steps + 1):
            z1 = rng.standard_normal(1)
            z2 = rng.standard_normal(1)
            w1 = z1
            w2 = rho * z1 + np.sqrt(1 - rho ** 2) * z2
            v_prev = np.maximum(v_odd[t - 1], 0)
            v_odd[t] = v_prev + kappa * (theta - v_prev) * dt + xi * np.sqrt(v_prev * dt) * w2
            v_odd[t] = np.maximum(v_odd[t], 0)
            S_odd[t] = S_odd[t - 1] * np.exp((r - 0.5 * v_prev) * dt + np.sqrt(v_prev * dt) * w1)
        S = np.concatenate([S, S_odd], axis=1)
    
    return S

# Import from NN training module
try:
    from NN_training_stock_iv import run_iv_nn_training, get_sigma_iv
    NN_MODULE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: NN_training_stock_iv module not found: {e}. Neural network IV features will be disabled.")
    run_iv_nn_training = None
    get_sigma_iv = None
    NN_MODULE_AVAILABLE = False

class IVModel:
    def __init__(self, nn_model: torch.nn.Module):
        self.model = nn_model.eval()
        self.device = next(self.model.parameters()).device
        
        # FIX: Proper scaler access
        if hasattr(nn_model, 'scaler') and nn_model.scaler is not None:
            self.m_scale = nn_model.scaler.m_scale
            self.tau_scale = nn_model.scaler.tau_scale
        else:
            raise ValueError("Model does not have a fitted scaler")

    def get_volatility_batch(self, K: float, S_batch: np.ndarray, tau: float) -> np.ndarray:
        tau = max(float(tau), 1e-6)
        S_batch = np.asarray(S_batch, dtype=np.float64)
        
        # Validate inputs
        if K <= 0:
            raise ValueError(f"K must be positive, got {K}")
        if np.any(S_batch <= 0):
            raise ValueError("All S_batch values must be positive")
        
        m = np.log(np.maximum(K, 1e-8) / np.maximum(S_batch, 1e-8))
        m_norm = m / self.m_scale
        tau_norm = tau / self.tau_scale
        X = np.column_stack([m_norm, np.full_like(m_norm, tau_norm)])
        
        with torch.no_grad():
            inp = torch.from_numpy(X).float().to(self.device)
            sig = self.model(inp).squeeze(1).clamp_min(1e-6).detach().cpu().numpy()
        
        # Clear GPU memory
        del inp
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return sig.astype(np.float64)

def simulate_local_vol_paths_antithetic(
    S0: float, r: float, T: float, num_simulations: int, num_time_steps: int,
    iv_model: IVModel, K: float, rng: np.random.Generator
) -> np.ndarray:
    dt = T / num_time_steps
    M = num_simulations // 2 * 2
    S = np.zeros((num_time_steps + 1, M), dtype=np.float64)
    S[0] = S0
    
    Z_half = rng.standard_normal((num_time_steps, M // 2))
    Z = np.concatenate([Z_half, -Z_half], axis=1)

    for t in range(1, num_time_steps + 1):
        S_prev = S[t - 1]
        tau_t = max(T - (t - 1) * dt, 1e-6)
        sigmas = iv_model.get_volatility_batch(K, S_prev, tau_t)
        drift = (r - 0.5 * sigmas**2) * dt
        diffusion = sigmas * np.sqrt(dt)
        S[t] = S_prev * np.exp(drift + diffusion * Z[t - 1])
    
    if num_simulations % 2 != 0:
        S_odd = np.zeros((num_time_steps + 1, 1), dtype=np.float64)
        S_odd[0] = S0
        Z_odd = rng.standard_normal((num_time_steps, 1))
        for t in range(1, num_time_steps + 1):
            S_prev = S_odd[t - 1]
            tau_t = max(T - (t - 1) * dt, 1e-6)
            sigmas = iv_model.get_volatility_batch(K, S_prev, tau_t)
            drift = (r - 0.5 * sigmas**2) * dt
            diffusion = sigmas * np.sqrt(dt)
            S_odd[t] = S_prev * np.exp(drift + diffusion * Z_odd[t - 1])
        S = np.concatenate([S, S_odd], axis=1)
    
    return S

# -------------------------------------------------------------------------
# Advanced Option Pricer with Enhanced Training
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
        verbose: bool = False,
        iv_model: Optional[IVModel] = None,
        use_streaming: bool = True,
        chunk_size: int = 500,
        european_approximation: bool = False,
        use_control_variate: bool = True
    ):
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
        self.verbose = verbose
        self.iv_model = iv_model
        self.use_streaming = use_streaming
        self.chunk_size = chunk_size
        self.european_approximation = european_approximation
        self.use_control_variate = use_control_variate

    def _payoff(self, S: np.ndarray) -> np.ndarray:
        if self.option_type == "call":
            return np.maximum(S - self.K, 0)
        else:
            return np.maximum(self.K - S, 0)

    def price_european_streaming(
        self,
        S0: float,
        T: float,
        num_simulations: int = 10000,
        num_time_steps: int = 50
    ) -> float:
        discount_factor = np.exp(-self.r * T)
        
        def simulator_func(batch_size, S0, r, T, num_time_steps, iv_model, K, rng_manager):
            rng = rng_manager.get_child_rng()
            
            if iv_model is not None:
                S = simulate_local_vol_paths_antithetic(S0, r, T, batch_size, num_time_steps, iv_model, K, rng)
            elif self.use_heston and self.heston_params is not None:
                S = simulate_heston_paths_antithetic(
                    S0, r, T,
                    self.heston_params["v0"], self.heston_params["kappa"],
                    self.heston_params["theta"], self.heston_params["xi"], self.heston_params["rho"],
                    batch_size, num_time_steps, rng
                )
            else:
                if self.sigma is None:
                    raise ValueError("sigma is None: provide sigma, iv_model, or heston configuration")
                dt = T / num_time_steps
                drift = (r - 0.5 * self.sigma ** 2) * dt
                diffusion = self.sigma * np.sqrt(dt)
                M = batch_size // 2 * 2
                Z_half = rng.standard_normal((num_time_steps, M // 2))
                Z = np.concatenate([Z_half, -Z_half], axis=1)
                S = np.zeros((num_time_steps + 1, M), dtype=np.float64)
                S[0] = S0
                for t in range(1, num_time_steps + 1):
                    S[t] = S[t - 1] * np.exp(drift + diffusion * Z[t - 1])
                
                if batch_size % 2 != 0:
                    S_odd = np.zeros((num_time_steps + 1, 1), dtype=np.float64)
                    S_odd[0] = S0
                    Z_odd = rng.standard_normal((num_time_steps, 1))
                    for t in range(1, num_time_steps + 1):
                        S_odd[t] = S_odd[t-1] * np.exp(drift + diffusion * Z_odd[t-1])
                    S = np.concatenate([S, S_odd], axis=1)
            
            payoffs = self._payoff(S[-1]) * discount_factor
            return payoffs.astype(np.float64)
        
        mean, stderr, n_done = monte_carlo_price_streaming(
            simulator_func,
            num_simulations,
            self.chunk_size,
            S0, self.r, T, num_time_steps, self.iv_model, self.K, self.rng_manager
        )
        
        if self.verbose:
            print(f"European streaming MC: {mean:.4f} ± {stderr:.4f} (n={n_done})")
        return mean

    def price_american_enhanced_lsm(
        self,
        S0: float,
        T: float,
        num_simulations: int = 10000,
        num_time_steps: int = 50
    ) -> float:
        """Enhanced LSM with proper weight saving and target scaling"""
        if S0 <= 0 or self.K <= 0 or T <= 0:
            raise ValueError("S0, K, T must be positive.")
        if self.r < 0:
            raise ValueError("r must be non-negative.")
        if num_simulations <= 0 or num_time_steps <= 0:
            raise ValueError("num_simulations and num_time_steps must be positive integers.")

        rng = self.rng_manager.get_child_rng()
        torch.manual_seed(self.rng_manager.get_child_seed())

        dt = T / num_time_steps
        M = num_simulations // 2 * 2

        # Generate all paths with antithetic variance reduction
        if self.iv_model is not None:
            S = simulate_local_vol_paths_antithetic(S0, self.r, T, M, num_time_steps, self.iv_model, self.K, rng)
        elif self.use_heston and self.heston_params is not None:
            S = simulate_heston_paths_antithetic(
                S0, self.r, T,
                self.heston_params["v0"], self.heston_params["kappa"],
                self.heston_params["theta"], self.heston_params["xi"], self.heston_params["rho"],
                M, num_time_steps, rng
            )
        else:
            if self.sigma is None:
                raise ValueError("sigma is None: provide sigma, iv_model, or heston configuration")
            drift = (self.r - 0.5 * self.sigma ** 2) * dt
            diffusion = self.sigma * np.sqrt(dt)
            Z_half = rng.standard_normal((num_time_steps, M // 2))
            Z = np.concatenate([Z_half, -Z_half], axis=1)
            S = np.zeros((num_time_steps + 1, M), dtype=np.float64)
            S[0] = S0
            for t in range(1, num_time_steps + 1):
                S[t] = S[t - 1] * np.exp(drift + diffusion * Z[t - 1])

        # Collect all training data across timesteps
        features_all = []
        targets_all = []
        cashflows = self._payoff(S[-1]).astype(np.float64)
        exercised = np.zeros(M, dtype=bool)
        discount = np.exp(-self.r * dt)

        # First pass: collect training data
        for t in range(num_time_steps - 1, 0, -1):
            cashflows *= discount
            itm = (self._payoff(S[t]) > 0) & (~exercised)
            
            if not np.any(itm):
                continue
            
            X = S[t, itm]
            Y = cashflows[itm]
            features = create_regression_features(X, self.K, self.r, T, t * dt)
            
            # FIXED: Proper reshaping to ensure consistent dimensions
            if len(X) > 0:  # Only append if we have data
                # Ensure features is always 2D
                if features.ndim == 1:
                    features = features.reshape(1, -1)
                
                # Ensure Y is always 2D column vector
                if Y.ndim == 0:
                    Y = np.array([[Y]])
                elif Y.ndim == 1:
                    Y = Y.reshape(-1, 1)
                
                # Only append if dimensions match
                if features.shape[0] == Y.shape[0]:
                    features_all.append(features)
                    targets_all.append(Y)
        
        if not features_all:
            return cashflows.mean()
        
        # FIXED: More robust array concatenation with validation
        try:
            # Ensure all feature arrays have the same number of columns
            feature_shapes = [f.shape for f in features_all]
            if len(set(s[1] for s in feature_shapes)) > 1:
                # If different number of features, standardize to the most common shape
                most_common_cols = max(
                    set(s[1] for s in feature_shapes),
                    key=lambda c: sum(1 for s in feature_shapes if s[1] == c)
                )
                # Filter features and targets in sync using original pairing
                paired = [(f, y) for f, y in zip(features_all, targets_all) if f.shape[1] == most_common_cols]
                if not paired:
                    return cashflows.mean()
                features_all, targets_all = zip(*paired)
                features_all = list(features_all)
                targets_all = list(targets_all)
            
            if not features_all:  # Check again after filtering
                return cashflows.mean()
                
            X_all = np.vstack(features_all)
            Y_all = np.vstack(targets_all)
            
        except ValueError as e:
            if self.verbose:
                print(f"Warning: Array concatenation failed: {e}. Using simple average.")
            return cashflows.mean()
        
        # Scale targets for better training stability
        Y_mean = Y_all.mean()
        Y_std = Y_all.std()
        if Y_std > 0:
            Y_all_scaled = (Y_all - Y_mean) / Y_std
        else:
            Y_all_scaled = Y_all - Y_mean
            Y_std = 1.0
        
        # Normalize features
        feat_mean = X_all.mean(axis=0)
        feat_std = X_all.std(axis=0)
        feat_std[feat_std == 0] = 1
        X_all_norm = (X_all - feat_mean) / feat_std
        
        device = torch.device("cpu")
        net = SingleLSMNet(input_dim=X_all.shape[1], hidden_dim=self.nn_hidden).to(device)
        
        # Create DataLoader for efficient training
        dataset = TensorDataset(
            torch.from_numpy(X_all_norm).float(),
            torch.from_numpy(Y_all_scaled).float()
        )
        loader = DataLoader(dataset, batch_size=min(256, len(dataset)), shuffle=True)
        
        opt = optim.Adam(net.parameters(), lr=self.nn_lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5, min_lr=1e-6)
        
        # Save best weights during training
        best_loss = float('inf')
        best_state_dict = None
        patience_counter = 0
        
        for epoch in range(self.nn_epochs):
            total_loss = 0.0
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                pred = net(batch_X)
                loss = nn.MSELoss()(pred, batch_y)
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            scheduler.step(avg_loss)
            
            # Save best weights
            if avg_loss < best_loss - 1e-6:
                best_loss = avg_loss
                best_state_dict = copy.deepcopy(net.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 8:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch+1}, restoring best weights")
                    break
        
        # Restore best weights
        if best_state_dict is not None:
            net.load_state_dict(best_state_dict)

        # Second pass: use trained network for exercise decisions
        cashflows = self._payoff(S[-1]).astype(np.float64)
        exercised = np.zeros(M, dtype=bool)
        
        for t in range(num_time_steps - 1, 0, -1):
            cashflows *= discount
            itm = (self._payoff(S[t]) > 0) & (~exercised)
            
            if not np.any(itm):
                continue
            
            X = S[t, itm]
            features = create_regression_features(X, self.K, self.r, T, t * dt)
            
            # Ensure features have the same shape as training data
            if features.shape[1] != X_all.shape[1]:
                if self.verbose:
                    print(f"Warning: Feature dimension mismatch at t={t}. Skipping exercise decision.")
                continue
            
            features_norm = (features - feat_mean) / feat_std
            
            with torch.no_grad():
                X_tensor = torch.from_numpy(features_norm).float().to(device)
                continuation_scaled = net(X_tensor).cpu().numpy().flatten()
                continuation = continuation_scaled * Y_std + Y_mean

            # FIX: exercise decision must be inside the loop
            immediate = self._payoff(X)
            to_exercise = immediate > continuation

            idx_itm = np.where(itm)[0]
            ex_idx = idx_itm[to_exercise]
            cashflows[ex_idx] = immediate[to_exercise]
            exercised[ex_idx] = True

        return cashflows.mean()

    def price_american_with_control_variate(
        self,
        S0: float,
        T: float,
        num_simulations: int = 10000,
        num_time_steps: int = 50
    ) -> float:
        american_price = self.price_american_enhanced_lsm(S0, T, num_simulations, num_time_steps)
        
        if not self.use_control_variate or self.sigma is None:
            return american_price
        
        european_mc = self.price_european_streaming(S0, T, num_simulations, num_time_steps)
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
        if self.use_streaming and self.european_approximation:
            if self.verbose:
                print("WARNING: Using European approximation for American option (streaming mode)")
            return self.price_european_streaming(S0, T, num_simulations, num_time_steps)

        if self.use_control_variate and self.sigma is not None:
            return self.price_american_with_control_variate(S0, T, num_simulations, num_time_steps)
        else:
            return self.price_american_enhanced_lsm(S0, T, num_simulations, num_time_steps)

    def compute_curve_for_S0(
        self,
        S0: float,
        intervals_per_day: int,
        total_points: int,
        num_simulations: int,
        plot_paths: bool
    ) -> List[Dict[str, Any]]:
        records = []
        for i in range(total_points, 0, -1):
            d = i / intervals_per_day
            T = d / 365
            steps = max(10, min(130, int(np.ceil(d))))
            
            est_price = self.price_american_option(S0, T, num_simulations, steps, plot_paths)
            records.append({'S0': S0, 'Days to Expiry': d, 'Option Value': est_price})
        return records

# -------------------------------------------------------------------------
# Enhanced Worker Function with Full Reproducibility
# -------------------------------------------------------------------------

def compute_curve_worker_enhanced(
    S0, K, r, sigma, option_type, worker_seed,
    intervals_per_day, total_points, num_simulations, plot_paths, use_heston, heston_params,
    nn_hidden=128, nn_epochs=25, nn_lr=1e-3, verbose=False, european_approximation=False,
    use_control_variate=True
):
    try:
        # Full reproducibility in worker
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        
        rng_manager = RNGManager(worker_seed)
        pricer = AdvancedOptionPricer(
            K, r, sigma, option_type, rng_manager, use_heston, heston_params,
            nn_hidden=nn_hidden, nn_epochs=nn_epochs, nn_lr=nn_lr, verbose=verbose,
            european_approximation=european_approximation, use_control_variate=use_control_variate
        )
        return pricer.compute_curve_for_S0(S0, intervals_per_day, total_points, num_simulations, plot_paths)
    except Exception as e:
        logging.error(f"Error in enhanced worker for S0={S0}: {e}")
        return []

# -------------------------------------------------------------------------
# Plotting Functions
# -------------------------------------------------------------------------

def plot_option_curves(df, s0_list, S0_live, K, sigma, r, option_type, ticker, model_name):
    fig = go.Figure()
    for S0 in s0_list:
        curve = df[df['S0'] == S0]
        fig.add_trace(go.Scatter(
            x=curve['Days to Expiry'],
            y=curve['Option Value'],
            mode='lines',
            name=f"S0 = ${S0}" + (" (Live)" if int(S0) == int(S0_live) else ""),
            line=dict(width=4 if int(S0) == int(S0_live) else 2, dash='solid' if int(S0) == int(S0_live) else 'dot'),
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
    fig = go.Figure()
    for idx, S0 in enumerate(s0_list):
        curve = df[df['S0'] == S0]
        iv_label = f"IV={ivs_for_plot[idx]:.2%}"
        fig.add_trace(go.Scatter(
            x=curve['Days to Expiry'],
            y=curve['Option Value'],
            mode='lines',
            name=f"S0 = ${S0} ({iv_label})" + (" (Live)" if int(S0) == int(S0_live) else ""),
            line=dict(width=4 if int(S0) == int(S0_live) else 2, dash='solid' if int(S0) == int(S0_live) else 'dot'),
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
                 f"<br><sup>{ticker} | K=${K} | r={r:.2%}</sup>",
            x=0.5,
            xanchor='center'
        ),
        legend=dict(
            title="Spot Price (S0) & IV",
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

# -------------------------------------------------------------------------
# Input Function
# -------------------------------------------------------------------------

def get_user_inputs():
    print("=== Enhanced American Option Pricer ===")
    ticker = input("Enter ticker symbol: ").strip().upper()
    expiry = input("Enter option expiry date (YYYY-MM-DD): ").strip()
    K = float(input("Enter strike price: ").strip())
    r = float(input("Enter risk-free rate (e.g., 0.03 for 3%): ").strip())
    option_type = input("Enter option type (call/put): ").strip().lower()
    num_simulations = int(input("Enter number of Monte Carlo simulations: ").strip())
    num_time_steps = int(input("Enter number of time steps: ").strip())
    seed = int(input("Enter random seed: ").strip())
    s0_start = int(input("Enter start of S0 grid: ").strip())
    s0_end = int(input("Enter end of S0 grid: ").strip())
    s0_step = int(input("Enter step for S0 grid: ").strip())
    intervals_per_day = int(input("Enter intervals per day (e.g., 4): ").strip())
    model = input("Model to run (bs/heston/both): ").strip().lower()
    max_workers = int(input("Enter max workers for parallel processing: ").strip())
    iv_input = input("Enter implied volatility (leave blank to auto-fetch, type 'nn' for neural net IV local vol): ").strip()
    iv = iv_input if iv_input else None
    
    euro_approx = input("Use European approximation for faster streaming but lower accuracy? (y/n): ").strip().lower() == 'y'
    control_variate = True  # Always enabled for best accuracy
    
    greeks_input = input("Enter Greeks as 5 space-separated values (Delta Gamma Vega Theta Rho), or leave blank to auto-calc: ").strip()
    if greeks_input:
        greeks = [float(x) for x in greeks_input.split()]
    else:
        greeks = None

    return {
        'ticker': ticker, 'expiry': expiry, 'K': K, 'r': r, 'option_type': option_type,
        'num_simulations': num_simulations, 'num_time_steps': num_time_steps,
        'seed': seed, 's0_start': s0_start, 's0_end': s0_end, 's0_step': s0_step,
        'max_workers': max_workers, 'verbose': True, 'plot_paths': False,
        'model': model, 'iv': iv, 'greeks': greeks, 'intervals_per_day': intervals_per_day,
        'european_approximation': euro_approx, 'control_variate': control_variate
    }

# -------------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------------

def main() -> None:
    print("Enhanced American Option Pricer Started")
    args = get_user_inputs()
    
    try:
        ticker = args['ticker']
        expiry = datetime.datetime.strptime(args['expiry'], "%Y-%m-%d").date()
        today = datetime.date.today()
        days_to_expiry = (expiry - today).days
        expiry_str = expiry.strftime("%Y-%m-%d")
        K = args['K']
        r = args['r']
        option_type = args['option_type']
        num_simulations = args['num_simulations']
        num_time_steps = args['num_time_steps']
        seed = args['seed']
        intervals_per_day = args['intervals_per_day']
        total_points = max(1, days_to_expiry) * intervals_per_day
        s0_start = args['s0_start']
        s0_end = args['s0_end']
        s0_step = args['s0_step']
        s0_list = list(range(s0_start, s0_end + 1, s0_step))
        MAX_WORKERS = args['max_workers']
        verbose = args['verbose']
        european_approximation = args['european_approximation']
        control_variate = args['control_variate']

        master_rng = RNGManager(seed)

        start_time = time.time()
        try:
            S0_live, sigma_live = MarketDataFetcher.get_live_quote(ticker)
        except Exception as e:
            logging.error(f"Failed to fetch market data: {e}")
            return

        if S0_live not in s0_list:
            s0_list.append(int(S0_live))
        s0_list = sorted(set(s0_list))

        heston_params = {
            "v0": None, "kappa": 2.0, "theta": None, "xi": 0.3, "rho": -0.7
        }

        use_nn_iv = args['iv'] is not None and args['iv'].lower() == 'nn'
        nn_iv_surface = None
        iv_model = None
        
        if use_nn_iv:
            if not NN_MODULE_AVAILABLE or run_iv_nn_training is None or get_sigma_iv is None:
                print("Error: Neural network IV features are not available. NN_training_stock_iv module not found.")
                print("Falling back to historical volatility...")
                use_nn_iv = False
                iv_model = None
                sigma = sigma_live
            else:
                print("Training neural net for IV surface...")
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
                    use_nn_iv = False
                    iv_model = None
                    sigma = sigma_live
        elif args['iv'] is not None:
            sigma = float(args['iv'])
            logging.info(f"Using user-supplied implied volatility: {sigma:.2%}")
        else:
            sigma_iv = get_live_iv(ticker, expiry_str, K, option_type)
            if not np.isnan(sigma_iv):
                logging.info(f"Using live implied volatility: {sigma_iv:.2%}")
                sigma = sigma_iv
            else:
                logging.info(f"Falling back to historical volatility: {sigma_live:.2%}")
                sigma = sigma_live

        heston_params["v0"] = sigma**2
        heston_params["theta"] = sigma**2

        model_choice = args['model']
        run_bs = model_choice in ("bs", "both")
        run_heston = model_choice in ("heston", "both")

        T_live = max(days_to_expiry, 1) / 365
        if args['greeks'] is not None:
            greeks = dict(zip(['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'], args['greeks']))
            logging.info("Using user-supplied Greeks:")
            for k, v in greeks.items():
                logging.info(f"  {k}: {v:.4f}")
        else:
            greeks = BlackScholesGreeks.greeks(S0_live, K, T_live, r, sigma, option_type)
            logging.info("Black-Scholes Greeks at S0_live:")
            for k, v in greeks.items():
                logging.info(f"  {k}: {v:.4f}")

        # Enhanced Black-Scholes / Local Vol
        if run_bs:
            if use_nn_iv and iv_model is not None:
                records_bs = []
                pricer = AdvancedOptionPricer(
                    K, r, sigma=None, option_type=option_type,
                    rng_manager=master_rng, use_heston=False, heston_params=None,
                    nn_hidden=128, nn_epochs=25, nn_lr=1e-3, verbose=verbose,
                    iv_model=iv_model, use_streaming=True, chunk_size=500,
                    european_approximation=european_approximation,
                    use_control_variate=control_variate
                )
                for S0 in tqdm(s0_list, desc="S0 curves (Enhanced Local Vol NN)"):
                    try:
                        records_bs.extend(
                            pricer.compute_curve_for_S0(S0, intervals_per_day, total_points, num_simulations, False)
                        )
                    except Exception as e:
                        logging.error(f"Enhanced Local Vol error for S0={S0}: {e}")
                
                df_bs = pd.DataFrame(records_bs)
                if not df_bs.empty:
                    logging.info("Sample Enhanced Local Vol (NN) results:")
                    logging.info(df_bs.head(10))
                    ivs_for_plot = [get_sigma_iv(nn_iv_surface, K, S0, T_live) for S0 in s0_list]
                    plot_option_curves_nn(df_bs, s0_list, S0_live, K, ivs_for_plot, r, option_type, ticker, "Enhanced Local Vol (NN)")
                else:
                    logging.warning("No valid results from Enhanced Local Vol (NN)")
            else:
                records_bs = []
                args_bs = []
                for S0 in s0_list:
                    worker_seed = master_rng.get_child_seed()
                    args_bs.append(
                        (S0, K, r, sigma, option_type, worker_seed,
                         intervals_per_day, total_points, num_simulations, False, False, None,
                         128, 25, 1e-3, verbose, european_approximation, control_variate)
                    )
                try:
                    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                        futures = [executor.submit(compute_curve_worker_enhanced, *arg) for arg in args_bs]
                        for future in tqdm(as_completed(futures), total=len(futures), desc="S0 curves (Enhanced BS)"):
                            records_bs.extend(future.result())
                    df_bs = pd.DataFrame(records_bs)
                    if not df_bs.empty:
                        logging.info("Sample Enhanced Black-Scholes results:")
                        logging.info(df_bs.head(10))
                        plot_option_curves(df_bs, s0_list, S0_live, K, sigma, r, option_type, ticker, "Enhanced Black-Scholes")
                    else:
                        logging.warning("No valid results from Enhanced Black-Scholes")
                except Exception as e:
                    logging.error(f"Error in Enhanced Black-Scholes batch: {e}")

        # Enhanced Heston
        if run_heston:
            records_heston = []
            args_heston = []
            for S0 in s0_list:
                worker_seed = master_rng.get_child_seed()
                sigma_S0 = sigma
                heston_params_local = heston_params.copy()
                heston_params_local["v0"] = sigma_S0**2
                heston_params_local["theta"] = sigma_S0**2
                args_heston.append(
                    (S0, K, r, sigma_S0, option_type, worker_seed,
                     intervals_per_day, total_points, num_simulations, False, True, heston_params_local,
                     128, 25, 1e-3, verbose, european_approximation, control_variate)
                )
            try:
                with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    futures = [executor.submit(compute_curve_worker_enhanced, *arg) for arg in args_heston]
                    for future in tqdm(as_completed(futures), total=len(futures), desc="S0 curves (Enhanced Heston)"):
                        records_heston.extend(future.result())
                df_heston = pd.DataFrame(records_heston)
                if not df_heston.empty:
                    logging.info("Sample Enhanced Heston results:")
                    logging.info(df_heston.head(10))
                    
                    if use_nn_iv and nn_iv_surface is not None:
                        ivs_for_plot = [get_sigma_iv(nn_iv_surface, K, S0, T_live) for S0 in s0_list]
                        plot_option_curves_nn(df_heston, s0_list, S0_live, K, ivs_for_plot, r, option_type, ticker, "Enhanced Heston")
                    else:
                        plot_option_curves(df_heston, s0_list, S0_live, K, sigma, r, option_type, ticker, "Enhanced Heston")
                else:
                    logging.warning("No valid results from Enhanced Heston")
            except Exception as e:
                logging.error(f"Error in Enhanced Heston batch: {e}")

        elapsed = time.time() - start_time
        logging.info(f"Enhanced pricing completed in: {elapsed:.2f} seconds")

    except Exception as e:
        logging.error(f"Fatal error in enhanced main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()



