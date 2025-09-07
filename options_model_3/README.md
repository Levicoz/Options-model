# Options Pricing with Neural Networks and Heston Calibration

This project implements advanced options pricing models using neural networks for implied volatility surface modeling and Heston stochastic volatility model calibration.

## Key Features

- **Neural Network IV Surface Training**: Train deep neural networks to model implied volatility surfaces with arbitrage constraints
- **Heston Model Calibration**: Calibrate Heston stochastic volatility parameters using neural network IV surfaces
- **GPU-Accelerated Pricing**: Fast Monte Carlo option pricing with GPU acceleration
- **Robust Error Handling**: Comprehensive error handling and fallback mechanisms

## Recent Improvements

Based on comprehensive code review, the following critical improvements have been implemented:

### 1. Fixed IVSurfaceModel.fit Method
- **Issue**: The `fit` method only fetched data but never actually trained the model
- **Fix**: Complete implementation that properly trains the neural network and returns validation loss

### 2. Improved DataScaler
- **Issue**: Only divided by standard deviation without centering features
- **Fix**: Proper centering around S0 (spot price) and normalization for both log-moneyness and time-to-expiry

### 3. Vega-Weighted Loss Function
- **Issue**: Raw price errors biased calibration toward tiny OTM options
- **Fix**: Implemented vega-weighted implied volatility loss for better calibration quality

### 4. Better Device Handling
- **Issue**: Device type mismatches between string and torch.device objects
- **Fix**: Robust device conversion and error handling

### 5. Enhanced Heston Calibration
- **Issue**: Price-relative errors caused instability with small option prices
- **Fix**: Implied volatility inversion with vega weighting and robust error handling

## Installation

```bash
pip install torch numpy pandas scipy yfinance plotly tqdm matplotlib
```

## Quick Start

### 1. Train IV Surface Neural Network

```python
from NN_training_stock_iv import run_iv_nn_training

# Train on real market data
model, val_loss, df, S0 = run_iv_nn_training(
    ticker="AAPL",
    epochs=50,
    hidden_dim=64,
    num_hidden_layers=4
)
```

### 2. Calibrate Heston Parameters

```python
from heston_calibration import calibrate_heston_from_iv_surface

# Calibrate Heston model using trained IV surface
heston_params = calibrate_heston_from_iv_surface(
    nn_model=model,
    ticker="AAPL", 
    S0=150.0,
    r=0.05
)
```

### 3. GPU Option Pricing

```python
from option_model_2_gpu import AdvancedOptionPricer, RNGManager

# Create GPU-accelerated pricer
rng = RNGManager(42)
pricer = AdvancedOptionPricer(
    K=150, r=0.05, sigma=0.2, option_type='call', rng_manager=rng
)

# Price American option
price = pricer.price_american_option(S0=150, T=0.25, num_simulations=10000)
```

## Project Structure

```
Options-model/
├── NN_training_stock_iv.py       # Neural network IV surface training
├── heston_calibration.py         # Heston model calibration
├── option_model_2_gpu.py         # GPU-accelerated option pricing
├── options_model_2.py            # CPU-based option pricing
├── test_improvements.py          # Comprehensive test suite
└── README.md                     # This file
```

## Key Classes and Functions

### Neural Network Training (`NN_training_stock_iv.py`)
- `IVSurfaceModel`: High-level interface for IV surface modeling
- `ImprovedIVNetwork`: Deep neural network with residual connections
- `DataScaler`: Improved data normalization with proper centering
- `ArbitragePenalty`: Finite-difference arbitrage constraints

### Heston Calibration (`heston_calibration.py`)
- `HestonCalibrator`: Main calibration class with regime detection
- `MarketRegimeDetector`: Adaptive parameter bounds based on market conditions
- `CalibrationPoint`: Data structure for calibration points
- Vega-weighted implied volatility objective function

### GPU Option Pricing (`option_model_2_gpu.py`)
- `AdvancedOptionPricer`: GPU-accelerated American option pricing
- `SingleLSMNet`: Neural network for Longstaff-Schwartz method
- Bandwidth-optimized path simulation
- Cached LSM networks for better performance

## Testing

Run the comprehensive test suite:

```bash
python test_improvements.py
```

Tests verify:
- DataScaler improvements
- IVSurfaceModel.fit functionality
- Device handling robustness
- Heston calibration improvements
- Vega-weighted loss computation

## Performance Improvements

The recent optimizations address critical performance bottlenecks:

1. **Memory bandwidth utilization**: Better GPU memory access patterns
2. **LSM network caching**: Reuse networks across pricing calls
3. **Adaptive epochs/steps**: Reduce computation for short-dated options
4. **Vectorized operations**: Minimize CPU-GPU transfer overhead

Expected performance gains:
- 7-10x faster option pricing (42 minutes → 5-8 minutes)
- Better GPU memory utilization (0.3GB → 1.5-2GB)
- More stable calibration with vega weighting

## Mathematical Background

### Implied Volatility Surface Modeling
- Log-moneyness features: `m = log(K/S₀)`
- Time-to-expiry: `τ = T`
- Arbitrage constraints via finite differences
- Residual neural networks with layer normalization

### Heston Model
- Stochastic volatility: `dv = κ(θ - v)dt + ξ√v dW₂`
- Stock price: `dS = rS dt + √v S dW₁`
- Correlation: `dW₁ dW₂ = ρ dt`
- Feller condition: `2κθ ≥ ξ²`

### Calibration Objective
Vega-weighted implied volatility MSE:
```
L = Σᵢ wᵢ vega(Kᵢ,Tᵢ) [IV_model(Kᵢ,Tᵢ) - IV_market(Kᵢ,Tᵢ)]²
```

## Contributing

1. Run tests: `python test_improvements.py`
2. Check code quality with type hints and docstrings
3. Validate against synthetic data before real market testing

## License

This project is for educational and research purposes.
