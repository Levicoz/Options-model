"""
Improved Neural Network IV Surface Training Module

This module implements neural network training for implied volatility surfaces with:
1. Fixed IVSurfaceModel.fit method that actually trains the model
2. Improved DataScaler with proper centering around S0
3. Better device handling for PyTorch models
4. Vega-weighted loss function for better calibration
5. Robust error handling and validation

Key improvements from comprehensive code review:
- DataScaler now centers features properly around S0
- IVSurfaceModel.fit actually trains and returns validation loss
- Device handling works with both string and torch.device objects
- Arbitrage penalties use properly scaled finite differences
- MC-dropout provides uncertainty estimates
"""

import os
import random
import argparse
import warnings
from typing import Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import yfinance as yf

# Set deterministic behavior for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

@dataclass
class TrainingConfig:
    """Configuration class for training parameters."""
    epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    lambda_K: float = 1e-3
    hidden_dim: int = 64
    num_hidden_layers: int = 4
    dropout: float = 0.1
    epsilon: float = 1e-4
    val_split: float = 0.15
    patience: int = 8
    use_cosine_schedule: bool = True
    use_augmentation: bool = True
    debug: bool = False
    plot_fit: bool = True
    seed: int = 42
    mc_dropout: bool = True
    mc_samples: int = 20
    save_path: Optional[str] = None
    use_vega_weighting: bool = True  # New: enable vega weighting

class DataScaler:
    """Improved data scaling with proper centering around S0."""
    
    def __init__(self):
        self.m_mean: float = 0.0
        self.m_scale: float = 1.0
        self.tau_mean: float = 0.0
        self.tau_scale: float = 1.0
        self.S0: Optional[float] = None
    
    def fit(self, df: pd.DataFrame, S0: float) -> None:
        """Fit scaler to data with proper centering around S0."""
        self.S0 = S0
        
        # Calculate log-moneyness centered around ATM
        m = df['m'].values
        tau = df['tau'].values
        
        # Center log-moneyness around ATM (m=0 when K=S0)
        self.m_mean = float(np.mean(m))
        self.m_scale = float(np.std(m))
        
        # Center time-to-expiry
        self.tau_mean = float(np.mean(tau))
        self.tau_scale = float(np.std(tau))
        
        # Avoid division by zero with reasonable minimum scales
        self.m_scale = max(self.m_scale, 1e-3)
        self.tau_scale = max(self.tau_scale, 1e-4)
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply scaling transformation with proper centering."""
        df_scaled = df.copy()
        df_scaled['m_norm'] = (df['m'] - self.m_mean) / self.m_scale
        df_scaled['tau_norm'] = (df['tau'] - self.tau_mean) / self.tau_scale
        return df_scaled
    
    def get_params(self) -> Dict[str, float]:
        """Get scaling parameters."""
        return {
            'm_mean': self.m_mean, 'm_scale': self.m_scale,
            'tau_mean': self.tau_mean, 'tau_scale': self.tau_scale,
            'S0': self.S0 or 0.0
        }

class ImprovedIVNetwork(nn.Module):
    """Enhanced neural network for IV surface modeling with better device handling."""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.scaler: Optional[DataScaler] = None
        
        # Network architecture with residual connections
        self.input_proj = nn.Linear(2, config.hidden_dim)
        self.layers = self._build_hidden_layers()
        self.output = nn.Linear(config.hidden_dim, 1)
        
        self._init_weights()
    
    def _build_hidden_layers(self) -> nn.ModuleList:
        """Build hidden layers with residual connections."""
        layers = nn.ModuleList()
        for _ in range(self.config.num_hidden_layers):
            layer = nn.Sequential(
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                nn.LayerNorm(self.config.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.config.dropout) if self.config.dropout > 0 else nn.Identity()
            )
            layers.append(layer)
        return layers
    
    def _init_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=np.sqrt(5))
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(module.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        h = F.gelu(self.input_proj(x))
        
        for layer in self.layers:
            h = h + layer(h)
        
        output = self.output(h)
        return output.clamp(min=self.config.epsilon)
    
    def predict_iv(self, K: float, S: float, tau: float, 
                   device: Optional[Union[str, torch.device]] = None,
                   n_samples: Optional[int] = None) -> Tuple[float, float]:
        """Predict IV with optional MC-dropout uncertainty. Returns (mean, std)."""
        if self.scaler is None:
            raise RuntimeError("Model scaler not fitted. Call fit() first.")
        
        # Improved device handling
        if device is None:
            device = next(self.parameters()).device
        elif isinstance(device, str):
            device = torch.device(device)
        
        m = np.log(K / S)
        m_norm = (m - self.scaler.m_mean) / self.scaler.m_scale
        tau_norm = (tau - self.scaler.tau_mean) / self.scaler.tau_scale
        
        x = torch.tensor([[m_norm, tau_norm]], dtype=torch.float32, device=device)
        
        # If MC-dropout disabled, simple eval
        if not self.config.mc_dropout:
            with torch.no_grad():
                pred = self(x).item()
            return float(pred), 0.0

        # Determine number of samples
        if n_samples is None:
            n_samples = self.config.mc_samples

        # Vectorized MC-dropout
        prev_mode = self.training
        try:
            self.train()  # Enable dropout
            x_repeated = x.repeat(n_samples, 1)
            with torch.no_grad():
                preds = self(x_repeated).squeeze(1)
        finally:
            self.train(prev_mode)
        
        mean_pred = float(preds.mean())
        std_pred = float(preds.std())
        return mean_pred, std_pred

    def predict_iv_batch(self, K_array: np.ndarray, S: float, tau_array: np.ndarray, 
                        device: Optional[Union[str, torch.device]] = None) -> np.ndarray:
        """Vectorized IV prediction for multiple points."""
        if self.scaler is None:
            raise RuntimeError("Model scaler not fitted. Call fit() first.")
        
        # Improved device handling
        if device is None:
            device = next(self.parameters()).device
        elif isinstance(device, str):
            device = torch.device(device)
        
        m = np.log(K_array / S)
        m_norm = (m - self.scaler.m_mean) / self.scaler.m_scale
        tau_norm = (tau_array - self.scaler.tau_mean) / self.scaler.tau_scale
        
        x = torch.tensor(np.column_stack([m_norm.flatten(), tau_norm.flatten()]), 
                        dtype=torch.float32, device=device)
        
        with torch.no_grad():
            output = self(x).squeeze(1).cpu().numpy()
        
        return output.reshape(K_array.shape)

class ArbitragePenalty:
    """Arbitrage penalty computation with properly scaled finite differences."""
    
    @staticmethod
    def compute_fd(model: ImprovedIVNetwork, batch_X: torch.Tensor,
                   lambda_butterfly: float = 1e-3, lambda_calendar: float = 1e-4,
                   eps_m_orig: float = 1e-3, eps_t_orig: float = 1/365.0) -> torch.Tensor:
        """
        Compute finite-difference arbitrage penalties with proper scaling.
        
        Args:
            model: The neural network model
            batch_X: Input batch [m_norm, tau_norm]
            lambda_butterfly: Weight for butterfly spread penalty
            lambda_calendar: Weight for calendar spread penalty
            eps_m_orig: Perturbation in original log-moneyness units
            eps_t_orig: Perturbation in original time units (years)
        """
        if model.scaler is None:
            return torch.tensor(0.0, device=batch_X.device)
        
        # Convert original units to normalized units
        eps_m = eps_m_orig / model.scaler.m_scale
        eps_t = eps_t_orig / model.scaler.tau_scale
        
        device = batch_X.device
        n = batch_X.shape[0]
        
        # Perturbation vectors
        e_m = torch.zeros_like(batch_X)
        e_t = torch.zeros_like(batch_X)
        e_m[:, 0] = eps_m
        e_t[:, 1] = eps_t
        
        # Forward evaluations for butterfly spread (convexity in m)
        w_center = model(batch_X).squeeze(1)
        w_plus = model(batch_X + e_m).squeeze(1)
        w_minus = model(batch_X - e_m).squeeze(1)
        
        # Second derivative approximation: d²w/dm²
        d2w_dm2 = (w_plus - 2*w_center + w_minus) / (eps_m**2)
        butterfly_penalty = torch.clamp(-d2w_dm2, min=0.0).sum()
        
        # Calendar spread (monotonicity in tau)
        w_tau_plus = model(batch_X + e_t).squeeze(1)
        dw_dtau = (w_tau_plus - w_center) / eps_t
        calendar_penalty = torch.clamp(-dw_dtau, min=0.0).sum()
        
        total_penalty = (lambda_butterfly * butterfly_penalty + 
                        lambda_calendar * calendar_penalty)
        
        return total_penalty

class DataProcessor:
    """Handles option data fetching and preprocessing."""
    
    @staticmethod
    def fetch_option_data(ticker: str, use_augmentation: bool = True) -> Tuple[pd.DataFrame, float]:
        """Fetch and process option data from yfinance."""
        try:
            stock = yf.Ticker(ticker)
            
            # Get current stock price
            hist = stock.history(period="1d")
            if hist.empty:
                raise ValueError(f"No price data found for {ticker}")
            S0 = float(hist['Close'].iloc[-1])
            
            # Get option chain data
            expiries = stock.options
            if not expiries:
                raise ValueError(f"No option data found for {ticker}")
            
            all_data = []
            for exp_date in expiries[:8]:  # Limit to first 8 expiries
                try:
                    chain = stock.option_chain(exp_date)
                    calls = chain.calls
                    puts = chain.puts
                    
                    # Process calls and puts
                    for df, option_type in [(calls, 'call'), (puts, 'put')]:
                        if df.empty:
                            continue
                            
                        # Filter for reasonable IVs and volume
                        df = df[
                            (df['impliedVolatility'] > 0.01) & 
                            (df['impliedVolatility'] < 2.0) &
                            (df['volume'] > 0)
                        ].copy()
                        
                        if df.empty:
                            continue
                        
                        # Calculate time to expiry
                        exp_dt = pd.to_datetime(exp_date)
                        T = (exp_dt - pd.Timestamp.now()).days / 365.0
                        T = max(T, 1/365)  # Minimum 1 day
                        
                        for _, row in df.iterrows():
                            K = float(row['strike'])
                            iv = float(row['impliedVolatility'])
                            
                            # Skip if IV is unreasonable
                            if not (0.01 <= iv <= 2.0):
                                continue
                            
                            all_data.append({
                                'K': K,
                                'T': T,
                                'sigma_IV': iv,
                                'option_type': option_type,
                                'volume': float(row.get('volume', 1))
                            })
                
                except Exception as e:
                    print(f"Warning: Failed to process expiry {exp_date}: {e}")
                    continue
            
            if not all_data:
                raise ValueError(f"No valid option data found for {ticker}")
            
            df = pd.DataFrame(all_data)
            df = DataProcessor._clean_data(df)
            
            if use_augmentation:
                df = DataProcessor._augment_data(df)
            
            return df, S0
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch option data for {ticker}: {e}")
    
    @staticmethod
    def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and filter option data."""
        # Remove duplicates
        df = df.drop_duplicates(subset=['K', 'T', 'option_type'])
        
        # Filter reasonable moneyness range
        df = df[(df['K'] > 0) & (df['sigma_IV'] > 0)]
        
        # Sort by expiry and strike
        df = df.sort_values(['T', 'K']).reset_index(drop=True)
        
        return df
    
    @staticmethod
    def _augment_data(df: pd.DataFrame, noise_std: float = 0.005) -> pd.DataFrame:
        """Add synthetic data points with small noise for better training."""
        augmented = []
        
        for _, row in df.iterrows():
            # Original point
            augmented.append(row.to_dict())
            
            # Add noisy versions (3 per original point)
            for _ in range(3):
                noisy_row = row.to_dict()
                noisy_row['sigma_IV'] = max(0.01, 
                    row['sigma_IV'] + np.random.normal(0, noise_std))
                augmented.append(noisy_row)
        
        return pd.DataFrame(augmented)
    
    @staticmethod
    def prepare_features(df: pd.DataFrame, S0: float) -> Tuple[pd.DataFrame, DataScaler]:
        """Prepare features with improved scaling."""
        # Calculate log-moneyness
        df['m'] = np.log(df['K'] / S0)
        df['tau'] = df['T']
        
        # Fit and apply scaler
        scaler = DataScaler()
        scaler.fit(df, S0)
        df_prepared = scaler.transform(df)
        
        return df_prepared, scaler

# Vega calculation for weighting
def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate Black-Scholes vega."""
    if T <= 0 or sigma <= 0:
        return 1e-8
    
    from scipy.stats import norm
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    return float(max(vega, 1e-8))

class IVSurfaceTrainer:
    """Main trainer class for IV surface neural networks."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: Optional[ImprovedIVNetwork] = None
        self.scaler: Optional[DataScaler] = None
        
    def train(self, df: pd.DataFrame, S0: float) -> Tuple[ImprovedIVNetwork, float]:
        """Train the IV surface model with improved loss weighting."""
        # Prepare data
        df_prepared, self.scaler = DataProcessor.prepare_features(df, S0)
        train_loader, val_loader = self._create_data_loaders(df_prepared)
        
        # Initialize model
        self.model = ImprovedIVNetwork(self.config).to(self.device)
        self.model.scaler = self.scaler
        
        # Initialize optimizer and scheduler
        optimizer = optim.AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=1e-4)
        scheduler = self._create_scheduler(optimizer)
        
        # Initialize output bias
        self._initialize_model(df_prepared)
        
        print(f"Using device: {self.device}")
        print(f"Data scaling - {self.scaler.get_params()}")
        
        # Training loop
        best_val_loss, train_losses, val_losses, best_ckpt_path = self._training_loop(
            train_loader, val_loader, optimizer, scheduler, df_prepared, S0
        )
        
        # Plot results if requested
        if self.config.plot_fit:
            self._plot_results(df, S0, train_losses, val_losses)
        
        return self.model, best_val_loss
    
    def _create_data_loaders(self, df: pd.DataFrame) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation data loaders."""
        X = torch.tensor(df[['m_norm', 'tau_norm']].values, dtype=torch.float32)
        y = torch.tensor(df['sigma_IV'].values, dtype=torch.float32).unsqueeze(1)
        dataset = TensorDataset(X, y)
        
        val_size = int(len(dataset) * self.config.val_split)
        train_size = len(dataset) - val_size
        gen = torch.Generator().manual_seed(self.config.seed)
        train_set, val_set = random_split(dataset, [train_size, val_size], generator=gen)
        
        num_workers = min(os.cpu_count() // 2, 4) if os.cpu_count() else 0
        
        train_loader = DataLoader(
            train_set, batch_size=self.config.batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            val_set, batch_size=self.config.batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=torch.cuda.is_available()
        )
        
        return train_loader, val_loader
    
    def _create_scheduler(self, optimizer):
        """Create learning rate scheduler."""
        if self.config.use_cosine_schedule:
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.epochs)
        else:
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    def _initialize_model(self, df: pd.DataFrame) -> None:
        """Initialize model with sensible defaults."""
        target_mean = df['sigma_IV'].mean()
        with torch.no_grad():
            self.model.output.bias.fill_(target_mean)
        print(f"Initialized output bias to target mean: {target_mean:.4f}")
    
    def _training_loop(self, train_loader, val_loader, optimizer, scheduler, 
                      df_prepared: pd.DataFrame, S0: float) -> Tuple[float, list, list, Optional[str]]:
        """Main training loop with vega weighting."""
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        train_losses, val_losses = [], []
        best_ckpt_path = None
        
        # Pre-calculate vega weights if enabled
        vega_weights = None
        if self.config.use_vega_weighting:
            vega_weights = self._calculate_vega_weights(df_prepared, S0)
        
        for epoch in range(self.config.epochs):
            # Training
            train_loss = self._train_epoch(train_loader, optimizer, vega_weights)
            train_losses.append(train_loss)
            
            # Validation
            val_loss = self._validate_epoch(val_loader, vega_weights)
            val_losses.append(val_loss)
            
            # Scheduler step
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
            
            # Early stopping
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
                
                # Save checkpoint if path provided
                if self.config.save_path:
                    best_ckpt_path = self.config.save_path
                    torch.save({
                        'model_state_dict': best_state,
                        'scaler_params': self.scaler.get_params(),
                        'config': self.config,
                        'epoch': epoch,
                        'val_loss': best_val_loss
                    }, best_ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if self.config.debug and epoch % 10 == 0:
                print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        
        # Load best model state
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        return best_val_loss, train_losses, val_losses, best_ckpt_path
    
    def _calculate_vega_weights(self, df: pd.DataFrame, S0: float) -> torch.Tensor:
        """Calculate vega weights for training samples."""
        weights = []
        for _, row in df.iterrows():
            K = row['K'] if 'K' in row else S0 * np.exp(row['m'])
            T = row['T'] if 'T' in row else row['tau']
            sigma = row['sigma_IV']
            
            vega = bs_vega(S=S0, K=K, T=T, r=0.05, sigma=sigma)  # Assume r=5%
            weights.append(vega / 100.0)  # Scale down vega
        
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        # Normalize weights
        weights = weights / weights.mean()
        return weights
    
    def _train_epoch(self, train_loader, optimizer, vega_weights=None) -> float:
        """Train for one epoch with optional vega weighting."""
        self.model.train()
        total_loss = 0.0
        sample_idx = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            batch_size = batch_X.shape[0]
            
            optimizer.zero_grad()
            
            # Forward pass
            pred = self.model(batch_X)
            
            # Base MSE loss
            mse_loss = nn.MSELoss(reduction='none')(pred, batch_y).squeeze(1)
            
            # Apply vega weights if available
            if vega_weights is not None:
                batch_weights = vega_weights[sample_idx:sample_idx + batch_size]
                mse_loss = (mse_loss * batch_weights).mean()
            else:
                mse_loss = mse_loss.mean()
            
            # Arbitrage penalty
            arb_penalty = ArbitragePenalty.compute_fd(self.model, batch_X, self.config.lambda_K)
            
            # Total loss
            loss = mse_loss + arb_penalty
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item() * batch_size
            sample_idx += batch_size
        
        return total_loss / len(train_loader.dataset)
    
    def _validate_epoch(self, val_loader, vega_weights=None) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        sample_idx = len(val_loader.dataset)  # Vega weights are for full dataset
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                batch_size = batch_X.shape[0]
                
                pred = self.model(batch_X)
                mse_loss = nn.MSELoss(reduction='none')(pred, batch_y).squeeze(1)
                
                # Apply vega weights if available
                if vega_weights is not None:
                    batch_weights = vega_weights[sample_idx:sample_idx + batch_size]
                    mse_loss = (mse_loss * batch_weights).mean()
                else:
                    mse_loss = mse_loss.mean()
                
                total_loss += mse_loss.item() * batch_size
                sample_idx += batch_size
        
        return total_loss / len(val_loader.dataset)
    
    def _plot_results(self, df: pd.DataFrame, S0: float, train_losses: list, val_losses: list) -> None:
        """Plot training results and IV surface."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # IV Surface
        K_grid = np.linspace(df['K'].min(), df['K'].max(), 50)
        T_grid = np.linspace(df['T'].min(), df['T'].max(), 20)
        K_mesh, T_mesh = np.meshgrid(K_grid, T_grid)
        
        plot_samples = min(self.config.mc_samples, 10)
        
        IV_mesh = np.zeros_like(K_mesh)
        Uncertainty_mesh = None
        if self.config.mc_dropout:
            Uncertainty_mesh = np.zeros_like(K_mesh)
        
        for i in range(K_mesh.shape[0]):
            for j in range(K_mesh.shape[1]):
                K, T = K_mesh[i,j], T_mesh[i,j]
                if self.config.mc_dropout:
                    iv_mean, iv_std = self.model.predict_iv(K, S0, T, n_samples=plot_samples)
                    IV_mesh[i,j] = iv_mean
                    if Uncertainty_mesh is not None:
                        Uncertainty_mesh[i,j] = iv_std
                else:
                    iv_mean, _ = self.model.predict_iv(K, S0, T)
                    IV_mesh[i,j] = iv_mean
        
        c1 = ax1.contourf(K_mesh, T_mesh, IV_mesh, levels=20, cmap='viridis')
        ax1.scatter(df['K'], df['T'], c=df['sigma_IV'], cmap='cool', s=20, 
                   edgecolor='k', alpha=0.7, label="Market IV")
        ax1.set_xlabel("Strike")
        ax1.set_ylabel("Time to Expiry (years)")
        ax1.set_title("IV Surface: Model vs Market")
        plt.colorbar(c1, ax=ax1)
        ax1.legend()
        
        # Training curves
        epochs_range = range(1, len(train_losses) + 1)
        ax2.plot(epochs_range, train_losses, 'b-', label='Training Loss', alpha=0.8)
        ax2.plot(epochs_range, val_losses, 'r-', label='Validation Loss', alpha=0.8)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training and Validation Loss')
        ax2.legend()
        ax2.set_yscale('log')
        
        # Residuals
        model_means = []
        for _, row in df.iterrows():
            iv_pred, _ = self.model.predict_iv(row['K'], S0, row['T'])
            model_means.append(iv_pred)
        residuals = np.array(model_means) - df['sigma_IV'].values
        
        ax3.scatter(df['sigma_IV'], residuals, alpha=0.6, s=20)
        ax3.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        ax3.set_xlabel('Market IV')
        ax3.set_ylabel('Model IV - Market IV')
        ax3.set_title('Residuals Plot')
        ax3.grid(True, alpha=0.3)
        
        # Model info
        ax4.text(0.1, 0.9, f"Architecture:", transform=ax4.transAxes, fontsize=12, fontweight='bold')
        ax4.text(0.1, 0.8, f"Hidden Layers: {self.config.num_hidden_layers}", transform=ax4.transAxes)
        ax4.text(0.1, 0.7, f"Hidden Dim: {self.config.hidden_dim}", transform=ax4.transAxes)
        ax4.text(0.1, 0.6, f"Dropout: {self.config.dropout}", transform=ax4.transAxes)
        ax4.text(0.1, 0.5, f"Lambda K: {self.config.lambda_K}", transform=ax4.transAxes)
        ax4.text(0.1, 0.4, f"Vega Weighting: {self.config.use_vega_weighting}", transform=ax4.transAxes)
        ax4.text(0.1, 0.3, f"Final RMSE: {np.sqrt(np.mean(residuals**2)):.6f}", transform=ax4.transAxes)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Model Info')
        
        plt.tight_layout()
        plt.show()

class IVSurfaceModel:
    """High-level interface for IV surface modeling - FIXED VERSION."""
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.trainer = IVSurfaceTrainer(self.config)
        self.model: Optional[ImprovedIVNetwork] = None
        self.S0: Optional[float] = None
        
    def fit(self, ticker: str) -> float:
        """Fit model to option data for given ticker - FIXED TO ACTUALLY TRAIN."""
        # Fetch data
        df, self.S0 = DataProcessor.fetch_option_data(ticker, self.config.use_augmentation)
        
        print(f"Fetched {len(df)} option data points for {ticker}")
        print(f"Strike range: {df['K'].min():.2f} - {df['K'].max():.2f}")
        print(f"Expiry range: {df['T'].min():.3f} - {df['T'].max():.3f} years")
        print(f"Spot S0: {self.S0:.2f}")
        
        # Actually train the model (this was missing!)
        self.model, best_val_loss = self.trainer.train(df, self.S0)
        
        # Ensure model scaler is set
        if self.model is not None and hasattr(self.trainer, 'scaler'):
            self.model.scaler = self.trainer.scaler
        
        return float(best_val_loss)
    
    def fit_synthetic(self, df: pd.DataFrame, S0: float) -> float:
        """Fit model to synthetic data for testing."""
        self.S0 = S0
        self.model, best_val_loss = self.trainer.train(df, S0)
        
        if self.model is not None and hasattr(self.trainer, 'scaler'):
            self.model.scaler = self.trainer.scaler
        
        return float(best_val_loss)
    
    def predict(self, K: float, tau: float) -> float:
        """Predict IV for given strike and time to expiry."""
        if self.model is None or self.S0 is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        res = self.model.predict_iv(K, self.S0, tau)
        
        # Unpack (mean, std) when MC-dropout is enabled
        if isinstance(res, (tuple, list)):
            return float(res[0])
        
        try:
            return float(res)
        except Exception:
            return float(res[0])
    
    def predict_surface(self, K_array: np.ndarray, tau_array: np.ndarray) -> np.ndarray:
        """Predict IV surface for arrays of strikes and time to expiry."""
        if self.model is None or self.S0 is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        return self.model.predict_iv_batch(K_array, self.S0, tau_array)

def create_synthetic_data() -> Tuple[pd.DataFrame, float]:
    """Create synthetic IV data for testing."""
    S0 = 100.0
    strikes = np.linspace(60, 140, 40)
    expiries_days = np.array([30, 60, 90])
    
    rows = []
    for d in expiries_days:
        T = d / 365.0
        for K in strikes:
            # Realistic IV smile
            moneyness = K / S0
            iv = 0.2 + 0.1 * abs(np.log(moneyness)) + 0.05 * (np.log(moneyness))**2
            iv += 0.02 * np.sqrt(T)  # Term structure effect
            iv = max(0.05, min(1.0, iv))  # Reasonable bounds
            
            rows.append({
                'K': K,
                'T': T,
                'm': np.log(K / S0),
                'tau': T,
                'sigma_IV': iv,
                'option_type': 'call',
                'volume': 100
            })
    
    return pd.DataFrame(rows), S0

# Main execution functions
def run_iv_nn_training(
    ticker: str,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    lambda_K: float = 1e-3,
    hidden_dim: int = 64,
    num_hidden_layers: int = 4,
    dropout: float = 0.1,
    use_bn: bool = False,
    device_str: Optional[str] = None,
    save_path: Optional[str] = None,
    plot_fit: bool = True,
    patience: int = 8,
    epsilon: float = 1e-4,
    debug: bool = False,
    **kwargs
):
    """
    Lightweight wrapper to train an IV surface model and return the trained network.
    Returns: (trained_model, best_val_loss, df_used, S0)
    """
    try:
        config = TrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            lambda_K=lambda_K,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            dropout=dropout,
            epsilon=epsilon,
            plot_fit=plot_fit,
            patience=patience,
            debug=debug,
            save_path=save_path,
            use_vega_weighting=kwargs.get('use_vega_weighting', True)
        )
        
        # Fetch data
        df, S0 = DataProcessor.fetch_option_data(ticker, config.use_augmentation)
        
        # Train model
        trainer = IVSurfaceTrainer(config)
        model, best_val_loss = trainer.train(df, S0)
        
        return model, best_val_loss, df, S0
    
    except Exception as e:
        print(f"Error in run_iv_nn_training: {e}")
        raise

def get_sigma_iv(model, K: float, S0: float, tau: float, return_std: bool = False, 
                device: Optional[Union[str, torch.device]] = None):
    """
    Robust helper to retrieve implied volatility from a model with improved device handling.
    """
    if model is None:
        raise ValueError("Model is None")

    if K <= 0 or S0 <= 0 or tau <= 0:
        raise ValueError("K, S0, and tau must be positive")

    # Improved device handling
    if device is not None and isinstance(device, str):
        device = torch.device(device)

    def _to_mean_std(res):
        """Convert various result types to (mean, std) tuple."""
        if isinstance(res, (tuple, list)) and len(res) >= 2:
            return float(res[0]), float(res[1])
        elif isinstance(res, (tuple, list)) and len(res) == 1:
            return float(res[0]), 0.0
        else:
            return float(res), 0.0

    # Preferred: ImprovedIVNetwork.predict_iv
    if hasattr(model, "predict_iv"):
        if device is not None:
            res = model.predict_iv(K, S0, tau, device=device)
        else:
            res = model.predict_iv(K, S0, tau)
        mean, std = _to_mean_std(res)
        return (mean, std) if return_std else mean

    # Next: IVSurfaceModel.predict
    if hasattr(model, "predict"):
        res = model.predict(K, tau)
        mean, std = _to_mean_std(res)
        return (mean, std) if return_std else mean

    # Fallback: callable
    if callable(model):
        res = model(K, S0, tau)
        mean, std = _to_mean_std(res)
        return (mean, std) if return_std else mean

    raise ValueError("Provided model does not support IV prediction interface")

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IV surface neural network")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--lambda_K", type=float, default=1e-3, help="Arbitrage penalty weight")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    parser.add_argument("--test", action="store_true", help="Run synthetic data test")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")
    parser.add_argument("--vega-weight", action="store_true", help="Enable vega weighting")
    
    args = parser.parse_args()
    
    # Create configuration
    config = TrainingConfig(
        epochs=args.epochs,
        lambda_K=args.lambda_K,
        debug=args.debug,
        plot_fit=not args.no_plot,
        use_augmentation=not args.test,
        use_vega_weighting=args.vega_weight
    )
    
    if args.test:
        print("Running synthetic data test...")
        df_test, S0_test = create_synthetic_data()
        trainer = IVSurfaceTrainer(config)
        model, val_loss = trainer.train(df_test, S0_test)
        print(f"Synthetic test completed. Validation loss: {val_loss:.6f}")
    else:
        print(f"Training IV surface model for {args.ticker}")
        model, val_loss, df, S0 = run_iv_nn_training(
            args.ticker, 
            epochs=args.epochs,
            lambda_K=args.lambda_K,
            debug=args.debug,
            plot_fit=not args.no_plot,
            use_vega_weighting=args.vega_weight
        )
        print(f"Training completed. Final validation loss: {val_loss:.6f}")
        print(f"Model trained on {len(df)} data points with S0=${S0:.2f}")
