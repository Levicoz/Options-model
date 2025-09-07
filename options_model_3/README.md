# Options Pricing with Neural Networks and Heston Calibration

This project implements advanced options pricing models using neural networks for implied volatility surface modeling and Heston stochastic volatility model calibration.

## Key Features

- **Neural Network IV Surface Training**: Train deep neural networks to model implied volatility surfaces with arbitrage constraints
- **Heston Model Calibration**: Calibrate Heston stochastic volatility parameters using neural network IV surfaces
- **GPU-Accelerated Pricing**: Fast Monte Carlo option pricing with GPU acceleration
- **Robust Error Handling**: Comprehensive error handling and fallback mechanisms

## Core Components

### Neural Network IV Surface Modeling
- **Deep Neural Networks**: Multi-layer networks with residual connections for modeling complex volatility surfaces
- **Arbitrage Constraints**: Built-in penalties to ensure no-arbitrage conditions across strikes and maturities
- **Data Normalization**: Proper scaling and centering of log-moneyness and time-to-expiry features
- **Vega Weighting**: Optional vega-weighted loss functions for market-consistent fitting

### Heston Model Calibration
- **Stochastic Volatility**: Full Heston model implementation with correlation and mean reversion
- **Market Regime Detection**: Adaptive parameter bounds based on current market volatility conditions
- **Multiple Optimizers**: Fallback optimization strategies for robust parameter estimation
- **Implied Volatility Objective**: Numerically stable objective functions using implied volatility rather than prices

### GPU-Accelerated Pricing
- **American Options**: Longstaff-Schwartz Monte Carlo with neural network basis functions
- **Memory Optimization**: Bandwidth-optimized path simulation for faster GPU execution
- **Adaptive Parameters**: Dynamic adjustment of simulation parameters based on option characteristics
- **Caching Systems**: Reusable neural network components to avoid redundant computations

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies
- Python 3.8+
- PyTorch (CPU or GPU)
- NumPy, Pandas, SciPy
- YFinance for market data
- Plotly/Matplotlib for visualization

### Optional GPU Support
For GPU acceleration, install PyTorch with CUDA support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Usage Examples

### Basic Neural Network Training
```python
from NN_training_stock_iv import IVSurfaceModel, TrainingConfig

# Configure training parameters
config = TrainingConfig(
    epochs=50,
    hidden_dim=64,
    num_hidden_layers=4,
    lambda_K=1e-3  # Arbitrage penalty weight
)

# Create and train model
model = IVSurfaceModel(config)
val_loss = model.fit("AAPL")  # Train on Apple options
print(f"Validation loss: {val_loss:.6f}")

# Predict implied volatilities
strikes = [140, 150, 160]  # Strike prices
maturities = [0.25, 0.5]   # Time to expiry in years
iv_surface = model.predict(strikes, maturities)
```

### Heston Model Calibration
```python
from heston_calibration import HestonCalibrator

# Initialize calibrator
calibrator = HestonCalibrator()

# Calibrate to market data
result = calibrator.calibrate_to_market_data(
    ticker="AAPL",
    S0=150.0,
    r=0.05
)

# Access calibrated parameters
heston_params = result['params']
print(f"Mean reversion: κ = {heston_params.kappa:.4f}")
print(f"Long-term variance: θ = {heston_params.theta:.4f}")
print(f"Vol of vol: σ = {heston_params.sigma:.4f}")
print(f"Correlation: ρ = {heston_params.rho:.4f}")
```

### GPU-Accelerated Option Pricing
```python
from option_model_2_gpu import AdvancedOptionPricer, RNGManager

# Setup GPU-accelerated pricer
rng = RNGManager(seed=42)
pricer = AdvancedOptionPricer(
    K=150,           # Strike price
    r=0.05,          # Risk-free rate
    sigma=0.2,       # Volatility
    option_type='call',
    rng_manager=rng
)

# Price American option with GPU acceleration
price = pricer.price_american_option(
    S0=150,                # Current stock price
    T=0.25,               # Time to expiry (3 months)
    num_simulations=10000, # Monte Carlo paths
    num_time_steps=50     # Discretization steps
)

print(f"American call option price: ${price:.2f}")
```

### Integrated Workflow
```python
# 1. Train neural network on market data
model = IVSurfaceModel(TrainingConfig(epochs=100))
model.fit("AAPL")

# 2. Calibrate Heston model using trained IV surface
calibrator = HestonCalibrator()
heston_result = calibrator.calibrate_from_iv_surface(
    nn_model=model,
    ticker="AAPL",
    S0=150.0,
    r=0.05
)

# 3. Price options using calibrated parameters
heston_params = heston_result['params']
# Use parameters for advanced pricing models...
```

## Project Structure

```
options_model_3/
├── NN_training_stock_iv.py       # Neural network IV surface training
├── heston_calibration.py         # Heston model calibration
├── option_model_2_gpu.py         # GPU-accelerated option pricing
├── options_model_2.py            # CPU-based option pricing
├── test_improvements.py          # Comprehensive test suite
├── demo_improvements.py          # Example usage demonstrations
├── quick_validation.py           # Quick validation script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Core Classes and Functions

### Neural Network Training (`NN_training_stock_iv.py`)
- `IVSurfaceModel`: High-level interface for IV surface modeling
- `ImprovedIVNetwork`: Deep neural network with residual connections and layer normalization
- `DataScaler`: Advanced data normalization with proper feature centering
- `ArbitragePenalty`: Finite-difference constraints for no-arbitrage conditions
- `TrainingConfig`: Configuration management for training parameters

### Heston Calibration (`heston_calibration.py`)
- `HestonCalibrator`: Main calibration class with regime detection and multiple optimizers
- `MarketRegimeDetector`: Adaptive parameter bounds based on market volatility conditions
- `CalibrationPoint`: Structured data for calibration target points
- `HestonParams`: Parameter validation and management
- Vega-weighted implied volatility objective functions

### GPU Option Pricing (`option_model_2_gpu.py`)
- `AdvancedOptionPricer`: GPU-accelerated American option pricing with Longstaff-Schwartz
- `SingleLSMNet`: Neural network basis functions for regression
- `RNGManager`: Reproducible random number generation for Monte Carlo
- `PathSimulator`: Optimized path generation for various stochastic models
- Bandwidth-optimized memory layouts for GPU performance

## Testing and Validation

Run the comprehensive test suite:

```bash
python test_improvements.py
```

The test suite validates:
- Neural network training functionality
- Data scaling and normalization accuracy
- Device handling across CPU/GPU configurations
- Heston calibration convergence
- Vega-weighted loss computation
- Arbitrage constraint enforcement
- GPU memory optimization (when available)

### Example Demonstrations

Run interactive examples:
```bash
python demo_improvements.py
```

Quick validation:
```bash
python quick_validation.py
```

## Performance Characteristics

### GPU Acceleration
- Utilizes CUDA for Monte Carlo path simulation
- Optimized memory bandwidth usage
- Adaptive batch sizing for different GPU configurations
- Automatic fallback to CPU if GPU unavailable

### Numerical Stability
- Vega-weighted objective functions prevent small-price instabilities
- Robust parameter bounds with market regime detection
- Multiple optimization fallback strategies
- Comprehensive error handling and validation

### Scalability
- Vectorized operations for large option portfolios
- Cached neural network components for repeated pricing
- Adaptive simulation parameters based on option characteristics
- Memory-efficient data structures

## Mathematical Framework

### Implied Volatility Surface Modeling
The neural network models implied volatility as a function of log-moneyness and time-to-expiry:
- **Features**: Log-moneyness `m = log(K/S₀)` and time-to-expiry `τ = T`
- **Architecture**: Deep residual networks with layer normalization
- **Constraints**: No-arbitrage conditions enforced via finite difference penalties
- **Objective**: MSE loss with optional vega weighting for market consistency

### Heston Stochastic Volatility Model
The Heston model describes asset price and volatility dynamics:
- **Volatility process**: `dv = κ(θ - v)dt + ξ√v dW₂`
- **Asset price**: `dS = rS dt + √v S dW₁`
- **Correlation**: `dW₁ dW₂ = ρ dt`
- **Feller condition**: `2κθ ≥ ξ²` ensures positive volatility

### Calibration Methodology
Parameter estimation uses vega-weighted implied volatility errors:
```
Objective = Σᵢ vega(Kᵢ,Tᵢ) × [IV_model(Kᵢ,Tᵢ) - IV_market(Kᵢ,Tᵢ)]²
```
This approach provides numerical stability and market-consistent parameter estimates.

### American Option Pricing
Uses the Longstaff-Schwartz method with neural network basis functions:
- **Path simulation**: Monte Carlo with antithetic variates
- **Continuation value**: Neural network regression on in-the-money paths
- **Exercise decision**: Comparison of immediate vs. continuation value
- **GPU optimization**: Vectorized operations and optimized memory layouts

## Contributing

1. **Code Quality**: Follow PEP 8 style guidelines and include comprehensive docstrings
2. **Testing**: Run the full test suite before submitting changes: `python test_improvements.py`
3. **Documentation**: Update README and code comments for new features
4. **Validation**: Test against both synthetic and real market data
5. **Performance**: Profile changes that may affect computational performance

## License

This project is intended for educational and research purposes. Please review the licensing terms before using in commercial applications.

## References

- Heston, S.L. (1993). "A Closed-Form Solution for Options with Stochastic Volatility"
- Longstaff, F.A. & Schwartz, E.S. (2001). "Valuing American Options by Simulation"
- Gatheral, J. (2006). "The Volatility Surface: A Practitioner's Guide"
