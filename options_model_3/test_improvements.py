"""
Comprehensive Test Suite for Options Pricing Improvements

This test suite validates all the critical fixes implemented based on the 
comprehensive code review:

1. DataScaler centering around S0
2. IVSurfaceModel.fit method actually training
3. Device handling improvements
4. Vega-weighted loss functions
5. Heston calibration with vega-weighted IV objective
6. GPU performance optimizations
7. Arbitrage constraints with proper scaling

Run this to ensure all improvements are working correctly.
"""

import sys
import os
import warnings
import time
from typing import Dict, Any, Tuple, List
import traceback

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our modules
try:
    from NN_training_stock_iv import (
        DataScaler, IVSurfaceModel, TrainingConfig, 
        create_synthetic_data, get_sigma_iv
    )
    from heston_calibration import (
        HestonParams, HestonCalibrator, MarketData, CalibrationConfig,
        create_synthetic_heston_data, calibrate_heston_to_data
    )
    # GPU module import is optional
    try:
        from option_model_2_gpu import (
            simulate_bs_paths_torch_bandwidth_optimized,
            create_regression_features_torch_vectorized,
            compute_multiple_S0_gpu_batch
        )
        GPU_AVAILABLE = True
    except ImportError as e:
        print(f"GPU module not available: {e}")
        GPU_AVAILABLE = False
        
except ImportError as e:
    print(f"Failed to import modules: {e}")
    print("Make sure all improved modules are in the same directory")
    sys.exit(1)

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class TestResults:
    """Container for test results."""
    
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.failures: List[Dict[str, Any]] = []
        self.timings: Dict[str, float] = {}
    
    def add_result(self, test_name: str, passed: bool, duration: float, 
                   error: str = None, details: Dict[str, Any] = None):
        """Add test result."""
        self.tests_run += 1
        self.timings[test_name] = duration
        
        if passed:
            self.tests_passed += 1
            print(f"✓ {test_name} ({duration:.3f}s)")
        else:
            self.tests_failed += 1
            print(f"✗ {test_name} ({duration:.3f}s) - {error}")
            self.failures.append({
                'test_name': test_name,
                'error': error,
                'details': details or {}
            })
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Tests run: {self.tests_run}")
        print(f"Tests passed: {self.tests_passed}")
        print(f"Tests failed: {self.tests_failed}")
        print(f"Success rate: {100.0 * self.tests_passed / self.tests_run:.1f}%")
        
        if self.failures:
            print(f"\nFAILURES ({len(self.failures)}):")
            for failure in self.failures:
                print(f"  - {failure['test_name']}: {failure['error']}")
        
        print(f"\nTIMINGS:")
        for test_name, duration in sorted(self.timings.items(), key=lambda x: x[1], reverse=True):
            print(f"  {test_name}: {duration:.3f}s")

def run_test(test_name: str, test_func, results: TestResults, *args, **kwargs):
    """Run a single test with timing and error handling."""
    start_time = time.time()
    try:
        test_func(*args, **kwargs)
        duration = time.time() - start_time
        results.add_result(test_name, True, duration)
        return True
    except Exception as e:
        duration = time.time() - start_time
        error_msg = str(e)
        if len(error_msg) > 100:
            error_msg = error_msg[:97] + "..."
        results.add_result(test_name, False, duration, error_msg)
        if results.tests_failed <= 3:  # Only print details for first few failures
            print(f"    Details: {traceback.format_exc()}")
        return False

def test_data_scaler_centering():
    """Test that DataScaler properly centers around S0."""
    # Create test data
    S0 = 100.0
    strikes = np.array([80, 90, 100, 110, 120])
    df = pd.DataFrame({
        'K': strikes,
        'T': [0.25] * len(strikes),
        'm': np.log(strikes / S0),
        'tau': [0.25] * len(strikes),
        'sigma_IV': [0.2] * len(strikes)
    })
    
    # Test scaler
    scaler = DataScaler()
    scaler.fit(df, S0)
    
    # Check that scaling parameters are reasonable
    assert abs(scaler.m_mean) < 1e-10, f"m_mean should be ~0 for symmetric strikes, got {scaler.m_mean}"
    assert scaler.m_scale > 0, f"m_scale should be positive, got {scaler.m_scale}"
    assert scaler.S0 == S0, f"S0 should be stored correctly, got {scaler.S0}"
    
    # Test transformation
    df_scaled = scaler.transform(df)
    assert 'm_norm' in df_scaled.columns, "m_norm column missing after transform"
    assert 'tau_norm' in df_scaled.columns, "tau_norm column missing after transform"
    
    # Check that ATM (m=0) gets mapped to close to 0 after normalization
    atm_idx = np.argmin(np.abs(df['m']))
    atm_m_norm = df_scaled.iloc[atm_idx]['m_norm']
    assert abs(atm_m_norm) < 0.1, f"ATM m_norm should be close to 0, got {atm_m_norm}"

def test_iv_surface_model_fit():
    """Test that IVSurfaceModel.fit actually trains the model."""
    # Create synthetic data
    df, S0 = create_synthetic_data()
    
    # Create model with small config for fast test
    config = TrainingConfig(
        epochs=5,
        batch_size=32,
        hidden_dim=32,
        num_hidden_layers=2,
        plot_fit=False,
        debug=False,
        patience=10  # Disable early stopping for short test
    )
    
    model = IVSurfaceModel(config)
    
    # Test that fit returns validation loss
    val_loss = model.fit_synthetic(df, S0)
    
    assert isinstance(val_loss, (int, float)), f"fit should return numeric validation loss, got {type(val_loss)}"
    assert val_loss > 0, f"Validation loss should be positive, got {val_loss}"
    assert model.model is not None, "Model should be trained after fit"
    assert model.S0 == S0, "S0 should be stored"
    
    # Test that we can make predictions
    test_K, test_tau = 100.0, 0.25
    pred_iv = model.predict(test_K, test_tau)
    assert isinstance(pred_iv, (int, float)), f"predict should return numeric IV, got {type(pred_iv)}"
    assert 0.01 < pred_iv < 1.0, f"Predicted IV should be reasonable, got {pred_iv}"

def test_device_handling():
    """Test improved device handling in neural networks."""
    # Create simple test data
    df, S0 = create_synthetic_data()
    
    config = TrainingConfig(
        epochs=2,
        batch_size=16,
        hidden_dim=16,
        num_hidden_layers=1,
        plot_fit=False,
        debug=False
    )
    
    model = IVSurfaceModel(config)
    model.fit_synthetic(df, S0)
    
    # Test different device specifications
    test_cases = [
        None,  # Default device
        'cpu',  # String device
        torch.device('cpu'),  # torch.device object
    ]
    
    if torch.cuda.is_available():
        test_cases.extend(['cuda', torch.device('cuda')])
    
    for device in test_cases:
        try:
            pred_iv = get_sigma_iv(model.model, K=100.0, S0=S0, tau=0.25, device=device)
            assert isinstance(pred_iv, (int, float)), f"Prediction failed for device {device}"
            assert 0.01 < pred_iv < 1.0, f"Unreasonable IV for device {device}: {pred_iv}"
        except Exception as e:
            raise AssertionError(f"Device handling failed for {device}: {e}")

def test_vega_weighting():
    """Test vega-weighted loss function implementation."""
    # Create synthetic data with varying strikes
    df, S0 = create_synthetic_data()
    
    # Test with vega weighting enabled vs disabled
    config_vega = TrainingConfig(
        epochs=3,
        batch_size=32,
        hidden_dim=16,
        plot_fit=False,
        use_vega_weighting=True
    )
    
    config_no_vega = TrainingConfig(
        epochs=3,
        batch_size=32,
        hidden_dim=16,
        plot_fit=False,
        use_vega_weighting=False
    )
    
    # Train both models
    model_vega = IVSurfaceModel(config_vega)
    model_no_vega = IVSurfaceModel(config_no_vega)
    
    loss_vega = model_vega.fit_synthetic(df, S0)
    loss_no_vega = model_no_vega.fit_synthetic(df, S0)
    
    # Both should train successfully
    assert loss_vega > 0, "Vega-weighted model should train"
    assert loss_no_vega > 0, "Non-vega-weighted model should train"
    
    # Predictions should be reasonable for both
    test_K, test_tau = 95.0, 0.25
    pred_vega = model_vega.predict(test_K, test_tau)
    pred_no_vega = model_no_vega.predict(test_K, test_tau)
    
    assert 0.01 < pred_vega < 1.0, f"Vega-weighted prediction unreasonable: {pred_vega}"
    assert 0.01 < pred_no_vega < 1.0, f"Non-vega prediction unreasonable: {pred_no_vega}"

def test_heston_calibration_vega_weighted():
    """Test Heston calibration with vega-weighted IV objective."""
    # Create synthetic data
    true_params = HestonParams(kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5, v0=0.04)
    df = create_synthetic_heston_data(true_params, add_noise=False)
    
    # Test calibration with vega weighting
    config = CalibrationConfig(
        use_vega_weighting=True,
        n_mc_paths=10000,  # Smaller for faster test
        max_iterations=100,
        verbose=False,
        plot_results=False
    )
    
    calibrated_params, summary = calibrate_heston_to_data(df, S0=100.0, config=config)
    
    # Check that calibration completed
    assert isinstance(calibrated_params, HestonParams), "Should return HestonParams object"
    assert 'error' in summary, "Summary should contain error"
    assert summary['error'] < 1.0, f"Calibration error too high: {summary['error']}"
    
    # Check that parameters are reasonable
    assert 0.1 < calibrated_params.kappa < 10.0, f"Unreasonable kappa: {calibrated_params.kappa}"
    assert 0.001 < calibrated_params.theta < 0.5, f"Unreasonable theta: {calibrated_params.theta}"
    assert 0.01 < calibrated_params.sigma < 2.0, f"Unreasonable sigma: {calibrated_params.sigma}"
    assert -0.99 < calibrated_params.rho < 0.99, f"Unreasonable rho: {calibrated_params.rho}"
    assert 0.001 < calibrated_params.v0 < 0.5, f"Unreasonable v0: {calibrated_params.v0}"

def test_arbitrage_penalties():
    """Test arbitrage penalty computation with proper scaling."""
    from NN_training_stock_iv import ArbitragePenalty, ImprovedIVNetwork
    
    # Create simple model
    config = TrainingConfig(hidden_dim=16, num_hidden_layers=1)
    model = ImprovedIVNetwork(config)
    
    # Create fake scaler (needed for penalty computation)
    from NN_training_stock_iv import DataScaler
    scaler = DataScaler()
    scaler.m_mean = 0.0
    scaler.m_scale = 0.2
    scaler.tau_mean = 0.1
    scaler.tau_scale = 0.05
    scaler.S0 = 100.0
    model.scaler = scaler
    
    # Create test batch
    batch_size = 10
    batch_X = torch.randn(batch_size, 2)  # [m_norm, tau_norm]
    
    # Compute penalty (should not crash)
    penalty = ArbitragePenalty.compute_fd(model, batch_X)
    
    assert isinstance(penalty, torch.Tensor), "Penalty should be a tensor"
    assert penalty.numel() == 1, "Penalty should be scalar"
    assert penalty.item() >= 0, f"Penalty should be non-negative, got {penalty.item()}"

def test_gpu_optimizations():
    """Test GPU optimization functions if available."""
    if not GPU_AVAILABLE:
        print("  Skipping GPU tests - module not available")
        return
    
    if not torch.cuda.is_available():
        print("  Skipping GPU tests - CUDA not available")
        return
    
    # Test bandwidth-optimized simulation
    S0 = 100.0
    r = 0.05
    sigma = 0.2
    T = 0.25
    n_paths = 1000
    n_steps = 50
    
    device = torch.device('cuda')
    
    # Test path simulation
    paths = simulate_bs_paths_torch_bandwidth_optimized(
        S0, r, sigma, T, n_paths, n_steps, device
    )
    
    assert paths.shape == (n_paths, n_steps + 1), f"Wrong path shape: {paths.shape}"
    assert torch.all(paths > 0), "All stock prices should be positive"
    assert torch.allclose(paths[:, 0], torch.full((n_paths,), S0, device=device)), "Initial prices should be S0"
    
    # Test regression features
    K_values = torch.tensor([90.0, 100.0, 110.0], device=device)
    S_T = paths[:, -1]  # Final stock prices
    
    features = create_regression_features_torch_vectorized(S_T, K_values, device)
    
    assert features.shape[0] == n_paths, f"Wrong feature shape: {features.shape}"
    assert features.shape[1] > 0, "Should have some features"
    
    print(f"  GPU paths shape: {paths.shape}")
    print(f"  GPU features shape: {features.shape}")

def test_performance_benchmarks():
    """Run performance benchmarks to ensure optimizations work."""
    # Test data scaling performance
    large_df = pd.DataFrame({
        'K': np.random.uniform(80, 120, 10000),
        'T': np.random.uniform(0.1, 2.0, 10000),
        'm': np.random.normal(0, 0.2, 10000),
        'tau': np.random.uniform(0.1, 2.0, 10000),
        'sigma_IV': np.random.uniform(0.1, 0.5, 10000)
    })
    
    # Time data scaling
    start_time = time.time()
    scaler = DataScaler()
    scaler.fit(large_df, S0=100.0)
    df_scaled = scaler.transform(large_df)
    scaling_time = time.time() - start_time
    
    assert scaling_time < 1.0, f"Data scaling too slow: {scaling_time:.3f}s"
    assert len(df_scaled) == len(large_df), "Scaling should preserve data size"
    
    # Test NN training speed (small model for quick test)
    small_config = TrainingConfig(
        epochs=3,
        batch_size=128,
        hidden_dim=32,
        num_hidden_layers=2,
        plot_fit=False,
        debug=False
    )
    
    df_small, S0 = create_synthetic_data()
    model = IVSurfaceModel(small_config)
    
    start_time = time.time()
    model.fit_synthetic(df_small, S0)
    training_time = time.time() - start_time
    
    assert training_time < 30.0, f"Training too slow: {training_time:.3f}s"
    
    print(f"  Data scaling (10k points): {scaling_time:.3f}s")
    print(f"  NN training (3 epochs): {training_time:.3f}s")

def test_numerical_stability():
    """Test numerical stability edge cases."""
    # Test with extreme parameter values
    extreme_df = pd.DataFrame({
        'K': [1e-6, 1e6],  # Very small and large strikes
        'T': [1e-6, 10.0],  # Very small and large times
        'm': [-10.0, 10.0],  # Extreme log-moneyness
        'tau': [1e-6, 10.0],
        'sigma_IV': [0.001, 0.999]  # Extreme volatilities
    })
    
    # Test scaler with extreme values
    scaler = DataScaler()
    try:
        scaler.fit(extreme_df, S0=100.0)
        df_scaled = scaler.transform(extreme_df)
        assert not np.any(np.isnan(df_scaled[['m_norm', 'tau_norm']].values)), "Scaling produced NaN"
        assert not np.any(np.isinf(df_scaled[['m_norm', 'tau_norm']].values)), "Scaling produced Inf"
    except Exception as e:
        raise AssertionError(f"Scaler failed with extreme values: {e}")
    
    # Test Heston parameter validation
    try:
        # Valid parameters
        valid_params = HestonParams(kappa=1.0, theta=0.04, sigma=0.2, rho=-0.5, v0=0.04)
        assert valid_params.feller_condition(), "Valid params should satisfy Feller"
        
        # Invalid parameters should raise errors
        try:
            invalid_params = HestonParams(kappa=-1.0, theta=0.04, sigma=0.2, rho=-0.5, v0=0.04)
            assert False, "Negative kappa should raise error"
        except ValueError:
            pass  # Expected
        
        try:
            invalid_params = HestonParams(kappa=1.0, theta=0.04, sigma=0.2, rho=1.5, v0=0.04)
            assert False, "Invalid rho should raise error"
        except ValueError:
            pass  # Expected
            
    except Exception as e:
        raise AssertionError(f"Parameter validation failed: {e}")

def test_memory_efficiency():
    """Test memory efficiency improvements."""
    # This test checks that we don't have obvious memory leaks
    import gc
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Run several training cycles
    for i in range(3):
        df, S0 = create_synthetic_data()
        config = TrainingConfig(
            epochs=2,
            batch_size=32,
            hidden_dim=16,
            plot_fit=False,
            debug=False
        )
        model = IVSurfaceModel(config)
        model.fit_synthetic(df, S0)
        
        # Force garbage collection
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    print(f"  Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB (+{memory_increase:.1f}MB)")
    
    # Allow some memory increase but not excessive
    assert memory_increase < 500, f"Excessive memory increase: {memory_increase:.1f}MB"

def main():
    """Run all tests."""
    print("Running comprehensive test suite for Options Pricing Improvements")
    print("="*60)
    
    results = TestResults()
    
    # Define test suite
    tests = [
        ("DataScaler Centering", test_data_scaler_centering),
        ("IVSurfaceModel.fit Training", test_iv_surface_model_fit),
        ("Device Handling", test_device_handling),
        ("Vega Weighting", test_vega_weighting),
        ("Heston Calibration", test_heston_calibration_vega_weighted),
        ("Arbitrage Penalties", test_arbitrage_penalties),
        ("GPU Optimizations", test_gpu_optimizations),
        ("Performance Benchmarks", test_performance_benchmarks),
        ("Numerical Stability", test_numerical_stability),
        ("Memory Efficiency", test_memory_efficiency),
    ]
    
    # Run tests
    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        run_test(test_name, test_func, results)
    
    # Print final summary
    results.print_summary()
    
    # Return exit code
    return 0 if results.tests_failed == 0 else 1

if __name__ == "__main__":
    exit_code = main()
    print(f"\nTest suite completed with exit code: {exit_code}")
    sys.exit(exit_code)
