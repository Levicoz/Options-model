# Machine learning + Heston calibration for option pricing with GPU acceleration (Python/PyTorch)

**This folder contains the most advanced option pricing model in the repository.**  
It combines neural networks, Heston stochastic volatility calibration, and GPU-accelerated Monte Carlo methods.

---

## Features

- **Implied Volatility Surface Modeling**
  - Deep neural networks for IV surfaces using log-moneyness and time-to-expiry features.
  - No-arbitrage penalties and vega-weighted loss for robust fitting.

- **Heston Model Calibration**
  - Full stochastic volatility model with adaptive optimization.
  - Vega-weighted calibration for numerical stability.

- **GPU-Accelerated American Option Pricing**
  - Monte Carlo pricing (Longstaff–Schwartz) with neural network basis.
  - CUDA-enabled; automatic fallback to CPU.

- **Engineering & Validation**
  - Modular code, quick validation scripts, and caching for repeated tasks.

---

## Installation

From your main repo directory:

```bash
pip install -r options_model_3/requirements.txt
```

**Dependencies**

* Python 3.8+
* PyTorch (CPU or GPU)
* NumPy, Pandas, SciPy
* yfinance (market data)
* Matplotlib or Plotly (visualization)

**Optional GPU support**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Getting Started

To run a full workflow demo (train NN, calibrate Heston, price options):

```bash
python options_model_3/demo_improvements.py
```

---

## Quick Usage Examples

### Neural Network IV Surface Training

```python
from NN_training_stock_iv import IVSurfaceModel, TrainingConfig

config = TrainingConfig(epochs=50, hidden_dim=64, num_hidden_layers=4, lambda_K=1e-3)
model = IVSurfaceModel(config)
val_loss = model.fit("AAPL")
iv_surface = model.predict([140, 150, 160], [0.25, 0.5])  # strikes, maturities
```

### Heston Model Calibration

```python
from heston_calibration import HestonCalibrator
calibrator = HestonCalibrator()
result = calibrator.calibrate_to_market_data(ticker="AAPL", S0=150.0, r=0.05)
heston_params = result['params']
```

### GPU-Accelerated American Option Pricing

```python
from option_model_2_gpu import AdvancedOptionPricer, RNGManager
rng = RNGManager(seed=42)
pricer = AdvancedOptionPricer(K=150, r=0.05, sigma=0.2, option_type='call', rng_manager=rng)
price = pricer.price_american_option(S0=150, T=0.25, num_simulations=10000, num_time_steps=50)
```

---

## Folder Structure

```
options_model_3/
├── NN_training_stock_iv.py       # Neural network IV surface training
├── heston_calibration.py         # Heston model calibration
├── option_model_2_gpu.py         # GPU-accelerated option pricing
├── options_model_2.py            # CPU-based option pricing utilities
├── test_improvements.py          # Test suite
├── demo_improvements.py          # Example demonstrations
├── quick_validation.py           # Quick validation checks
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Validation & Testing

Run the test suite for this model:

```bash
python options_model_3/test_improvements.py
```

---

## Notes & Limitations

* **Market data sensitivity**: Calibration/fitting depend on data quality.
* **Calibration stability**: Multiple optimizers and robust bounds included.
* **Performance**: GPU acceleration improves throughput, but varies by hardware.
* **Research/educational use**: Validate before production or trading.

---

## Contributing

* Follow PEP 8 and document non-trivial functions.
* Add tests and run `python options_model_3/test_improvements.py` before PRs.
* Update documentation for new modules/API changes.

---

## References

* Heston, S. L. (1993). *A Closed-Form Solution for Options with Stochastic Volatility*.
* Longstaff, F. A., & Schwartz, E. S. (2001). *Valuing American Options by Simulation*.
* Gatheral, J. (2006). *The Volatility Surface: A Practitioner’s Guide*.

---

## License

Educational and research use only. See main repo license for details.

