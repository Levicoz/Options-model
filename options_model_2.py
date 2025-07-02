# -------------------------------------------------------------------------
# Imports and Configuration
# -------------------------------------------------------------------------

# Standard library
import argparse
import datetime
import logging
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Dict, List, Any

# Third-party
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from tqdm import tqdm
import plotly.io as pio

# --- Logging and Plotly configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
pio.renderers.default = "browser"

# -------------------------------------------------------------------------
# Utility Classes and Stubs
# -------------------------------------------------------------------------

# --- Black-Scholes Greeks Calculation ---
class BlackScholesGreeks:
    @staticmethod
    def greeks(S, K, T, r, sigma, option_type='call'):
        """Return Delta, Gamma, Vega, Theta (per day), Rho for a European option (Vega/Rho per 1% change, per year)."""
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
        # Convert to standard conventions:
        theta = theta / 365                # Theta: per day
        vega = vega / 100                  # Vega: per 1% vol change, per year
        rho = rho / 100                    # Rho: per 1% rate change, per year
        return {'Delta': delta, 'Gamma': gamma, 'Vega': vega, 'Theta': theta, 'Rho': rho}

# --- Exotic Option Pricer Stub ---
class ExoticOptionPricer:
    @staticmethod
    def price_barrier_option(*args, **kwargs):
        """Stub for barrier option pricing (to be implemented)."""
        print("Barrier option pricing not yet implemented.")
        return np.nan

# --- Model Calibration Stub ---
class ModelCalibrator:
    @staticmethod
    def calibrate_heston_to_market(*args, **kwargs):
        """Stub for Heston model calibration (to be implemented)."""
        print("Heston calibration not yet implemented.")
        return None

# --- Web Dashboard Stub ---
def launch_dashboard():
    """Stub for launching a web dashboard (e.g., with Dash or Streamlit)."""
    print("Web dashboard feature not yet implemented.")

# -------------------------------------------------------------------------
# Data Fetching and Preprocessing
# -------------------------------------------------------------------------

def get_live_iv(ticker: str, expiry: str, strike: float, option_type: str = "call") -> float:
    """
    Fetches the implied volatility for a given ticker, expiry (YYYY-MM-DD), strike, and option type.
    Returns np.nan if not found or if IV is out of a reasonable range.
    """
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
        print(f"Using IV={iv:.2%} for strike={closest_strike}, expiry={expiry}")
        if np.isnan(iv) or iv < 0.01 or iv > 2.0:
            print("IV is NaN or out of reasonable range, falling back to historical volatility.")
            return np.nan
        return float(iv)
    except Exception as e:
        print(f"Could not fetch IV: {e}")
        return np.nan

# --- Neural Network for LSM Regression ---
class ContNet(nn.Module):
    """Simple feedforward neural network for LSM regression."""
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# --- Market Data Fetcher ---
class MarketDataFetcher:
    """Fetches live market data and computes historical volatility."""
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
# Heston Path Simulator
# -------------------------------------------------------------------------

def simulate_heston_paths(
    S0: float, r: float, T: float, v0: float, kappa: float, theta: float, xi: float, rho: float,
    num_simulations: int, num_time_steps: int, seed: int = 42
) -> np.ndarray:
    """Simulate Heston model price paths."""
    np.random.seed(seed)
    dt = T / num_time_steps
    S = np.zeros((num_time_steps + 1, num_simulations))
    v = np.zeros((num_time_steps + 1, num_simulations))
    S[0] = S0
    v[0] = v0
    for t in range(1, num_time_steps + 1):
        z1 = np.random.standard_normal(num_simulations)
        z2 = np.random.standard_normal(num_simulations)
        w1 = z1
        w2 = rho * z1 + np.sqrt(1 - rho ** 2) * z2
        v_prev = np.maximum(v[t - 1], 0)
        v[t] = v_prev + kappa * (theta - v_prev) * dt + xi * np.sqrt(v_prev * dt) * w2
        v[t] = np.maximum(v[t], 0)
        S[t] = S[t - 1] * np.exp((r - 0.5 * v_prev) * dt + np.sqrt(v_prev * dt) * w1)
    return S

# -------------------------------------------------------------------------
# Option Pricer (LSM with Neural Network Regression)
# -------------------------------------------------------------------------

class OptionPricer:
    """
    Option pricer using LSM with neural network regression.
    Note: lsm_poly_degree is not used when using neural network regression.
    """
    def __init__(
        self,
        K: float,
        r: float,
        sigma: Optional[float],
        option_type: str = 'call',
        lsm_poly_degree: int = 2,
        seed: int = 42,
        use_heston: bool = False,
        heston_params: Optional[Dict[str, Any]] = None,
        nn_hidden: int = 32,
        nn_epochs: int = 10,
        nn_lr: float = 1e-3,
        verbose: bool = False
    ):
        self.K = K
        self.r = r
        self.sigma = sigma
        self.option_type = option_type
        self.lsm_poly_degree = lsm_poly_degree
        self.seed = seed
        self.use_heston = use_heston
        self.heston_params = heston_params
        self.nn_hidden = nn_hidden
        self.nn_epochs = nn_epochs
        self.nn_lr = nn_lr
        self.verbose = verbose

    def _payoff(self, S: np.ndarray) -> np.ndarray:
        """Payoff function for call or put."""
        if self.option_type == "call":
            return np.maximum(S - self.K, 0)
        else:
            return np.maximum(self.K - S, 0)

    def price_american_option(
        self,
        S0: float,
        T: float,
        num_simulations: int = 10000,
        num_time_steps: int = 50,
        plot_paths: bool = False
    ) -> float:
        """
        Price an American option using LSM with neural network regression.
        """
        # --- Input validation ---
        if S0 <= 0 or self.K <= 0 or T <= 0 or (self.sigma is None and not self.use_heston):
            raise ValueError("S0, K, T, and sigma must be positive.")
        if self.r < 0:
            raise ValueError("r must be non-negative.")
        if num_simulations <= 0 or num_time_steps <= 0:
            raise ValueError("num_simulations and num_time_steps must be positive integers.")
        if self.lsm_poly_degree < 0 or not isinstance(self.lsm_poly_degree, int):
            raise ValueError("lsm_poly_degree must be a non-negative integer.")
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'.")

        # --- Set all seeds for reproducibility ---
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)

        dt = T / num_time_steps
        discount_factor = np.exp(-self.r * dt)

        # --- Simulate price paths ---
        M = num_simulations // 2 * 2
        if self.use_heston and self.heston_params is not None:
            S = simulate_heston_paths(
                S0, self.r, T,
                self.heston_params["v0"], self.heston_params["kappa"],
                self.heston_params["theta"], self.heston_params["xi"], self.heston_params["rho"],
                M, num_time_steps, self.seed
            )
        else:
            drift = (self.r - 0.5 * self.sigma ** 2) * dt
            diffusion = self.sigma * np.sqrt(dt)
            Z = np.random.standard_normal((num_time_steps, M // 2))
            Z = np.concatenate([Z, -Z], axis=1)
            S = np.zeros((num_time_steps + 1, M))
            S[0] = S0
            for t in range(1, num_time_steps + 1):
                S[t] = S[t - 1] * np.exp(drift + diffusion * Z[t - 1])

        # --- Optional: Plot simulated paths ---
        if plot_paths:
            plt.figure(figsize=(10, 6))
            for i in range(min(100, M)):
                plt.plot(np.linspace(0, T, num_time_steps + 1), S[:, i], alpha=0.5)
            plt.title("Simulated Stock Price Paths")
            plt.xlabel("Time to Maturity")
            plt.ylabel("Stock Price")
            plt.grid()
            plt.show()

        # --- LSM Regression with Neural Network ---
        cashflows = self._payoff(S[-1])
        exercised = np.zeros(M, dtype=bool)
        discount = np.exp(-self.r * dt)

        for t in range(num_time_steps - 1, 0, -1):
            cashflows *= discount
            itm = (self._payoff(S[t]) > 0) & (~exercised)
            if not np.any(itm):
                continue
            X = S[t, itm]
            Y = cashflows[itm]
            Xs = (X - X.mean()) / X.std() if X.std() > 0 else X - X.mean()
            Xs = Xs.reshape(-1, 1)
            Y = Y.reshape(-1, 1)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            net = ContNet(hidden=self.nn_hidden).to(device)
            opt = optim.Adam(net.parameters(), lr=self.nn_lr)
            X_tensor = torch.from_numpy(Xs).float().to(device)
            Y_tensor = torch.from_numpy(Y).float().to(device)
            for _ in range(self.nn_epochs):
                pred = net(X_tensor)
                loss = nn.MSELoss()(pred, Y_tensor)
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                continuation = net(X_tensor).cpu().numpy().flatten()
            immediate = self._payoff(X)
            to_exercise = immediate > continuation
            idx_itm = np.where(itm)[0]
            ex_idx = idx_itm[to_exercise]
            cashflows[ex_idx] = immediate[to_exercise]
            exercised[ex_idx] = True

        # --- Price statistics ---
        est_price = cashflows.mean()
        std_price = cashflows.std()
        lower = max(0, est_price - std_price)
        upper = est_price + std_price
        zero_prob = np.mean(cashflows == 0)
        if self.verbose:
            logging.info(f"Probability option expires worthless: {zero_prob:.2%}")
            logging.info(
                f"Estimated American {self.option_type} price: ${est_price:.4f} "
                f"(S0={S0}, K={self.K}, T={T}, r={self.r}, sigma={self.sigma}, "
                f"simulations={num_simulations}, steps={num_time_steps}, heston={self.use_heston})"
            )
            logging.info(
                f"One standard deviation range: "
                f"${lower:.4f} to ${upper:.4f}"
            )
            logging.info(f"Mean: ${est_price:.4f}")
            logging.info(f"Std Dev: ${std_price:.4f}")
            logging.info(f"Min: ${cashflows.min():.4f}")
            logging.info(f"Max: ${cashflows.max():.4f}")
            logging.info(f"Probability expires worthless: {zero_prob:.2%}")
        return est_price

    def compute_curve_for_S0(
        self,
        S0: float,
        intervals_per_day: int,
        total_points: int,
        num_simulations: int,
        plot_paths: bool
    ) -> List[Dict[str, Any]]:
        """Compute option value curve for a given S0 over time to expiry."""
        records = []
        for i in range(total_points, 0, -1):
            d = i / intervals_per_day
            T = d / 365
            steps = max(10, min(130, int(np.ceil(d))))
            np.random.seed(self.seed)
            est_price = self.price_american_option(
                S0, T, num_simulations, steps, plot_paths
            )
            records.append({'S0': S0, 'Days to Expiry': d, 'Option Value': est_price})
        return records

# -------------------------------------------------------------------------
# Plotting Utility
# -------------------------------------------------------------------------

class Plotter:
    @staticmethod
    def plot_curves(
        df: pd.DataFrame,
        s0_list: List[float],
        S0_live: float,
        K: float,
        sigma: float,
        r: float,
        option_type: str,
        ticker: str,
        model_name: str
    ) -> None:
        """Plot option value curves."""
        if df.empty:
            print(f"No results to plot for {model_name} (DataFrame is empty).")
            print("df columns:", df.columns)
            return
        required_cols = {'S0', 'Days to Expiry', 'Option Value'}
        if not required_cols.issubset(df.columns):
            print(f"Missing expected columns in DataFrame: {df.columns}")
            return
        fig = go.Figure()
        for S0 in s0_list:
            curve = df[df['S0'] == S0]
            fig.add_trace(go.Scatter(
                x=curve['Days to Expiry'],
                y=curve['Option Value'],
                mode='lines',
                name=f"S0 = ${S0}" + (" (Live)" if S0 == int(S0_live) else ""),
                line=dict(width=4 if S0 == int(S0_live) else 2, dash='solid' if S0 == int(S0_live) else 'dot'),
                hovertemplate=(
                    'S0: $%{text}<br>'
                    'Days to Expiry: %{x:.2f}<br>'
                    'Option Value: %{y:.4f}<extra></extra>'
                ),
                text=[S0]*len(curve)
            ))
        fig.update_layout(
            title=dict(
                text=f"American {option_type.capitalize()} Option Value vs. Days to Expiry<br><sup>{ticker} | K=${K} | σ={sigma:.2f} | r={r:.2%} | Model: {model_name}</sup>",
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
            linecolor='black',
            dtick=1
        )
        fig.show()

# -------------------------------------------------------------------------
# Parallel Worker for Curve Computation
# -------------------------------------------------------------------------

def compute_curve_worker(
    S0, K, r, sigma, option_type, lsm_poly_degree, seed,
    intervals_per_day, total_points, num_simulations, plot_paths, use_heston, heston_params,
    nn_hidden=32, nn_epochs=10, nn_lr=1e-3, verbose=False
):
    """Worker for parallel curve computation with error handling."""
    try:
        pricer = OptionPricer(
            K, r, sigma, option_type, lsm_poly_degree, seed, use_heston, heston_params,
            nn_hidden=nn_hidden, nn_epochs=nn_epochs, nn_lr=nn_lr, verbose=verbose
        )
        return pricer.compute_curve_for_S0(S0, intervals_per_day, total_points, num_simulations, plot_paths)
    except Exception as e:
        logging.error(f"Error in worker for S0={S0}: {e}")
        return []

# -------------------------------------------------------------------------
# Main Execution Block
# -------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="American Option Pricer with LSM and Neural Network Regression")
    parser.add_argument('--ticker', type=str, default='AMD', help='Ticker symbol')
    parser.add_argument('--expiry', type=str, default='2025-08-08', help='Option expiry date (YYYY-MM-DD)')
    parser.add_argument('--K', type=float, default=125, help='Strike price')
    parser.add_argument('--r', type=float, default=0.05, help='Risk-free rate')
    parser.add_argument('--option_type', type=str, default='call', choices=['call', 'put'], help='Option type')
    parser.add_argument('--num_simulations', type=int, default=1000000, help='Number of Monte Carlo simulations')
    parser.add_argument('--num_time_steps', type=int, default=150, help='Number of time steps')
    parser.add_argument('--lsm_poly_degree', type=int, default=2, help='LSM polynomial degree')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed')
    parser.add_argument('--s0_start', type=int, default=110, help='Start of S0 grid')
    parser.add_argument('--s0_end', type=int, default=130, help='End of S0 grid')
    parser.add_argument('--s0_step', type=int, default=2, help='Step for S0 grid')
    parser.add_argument('--max_workers', type=int, default=6, help='Max parallel workers')
    parser.add_argument('--nn_hidden', type=int, default=32, help='Neural net hidden units')
    parser.add_argument('--nn_epochs', type=int, default=10, help='Neural net epochs')
    parser.add_argument('--nn_lr', type=float, default=1e-3, help='Neural net learning rate')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--plot_paths', action='store_true', help='Plot simulated paths')
    parser.add_argument('--model', type=str, default='both', choices=['bs', 'heston', 'both'], help='Model to run')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    try:
        # --- User and Market Parameters ---
        ticker = args.ticker
        expiry = datetime.datetime.strptime(args.expiry, "%Y-%m-%d").date()
        today = datetime.date.today()
        days_to_expiry = (expiry - today).days
        expiry_str = expiry.strftime("%Y-%m-%d")
        K = args.K
        r = args.r
        option_type = args.option_type
        num_simulations = args.num_simulations
        num_time_steps = args.num_time_steps
        lsm_poly_degree = args.lsm_poly_degree
        plot_paths = args.plot_paths
        seed = args.seed
        intervals_per_day = 4
        total_points = days_to_expiry * intervals_per_day
        s0_start = args.s0_start
        s0_end = args.s0_end
        s0_step = args.s0_step
        s0_list = list(range(s0_start, s0_end + 1, s0_step))
        MAX_WORKERS = args.max_workers
        nn_hidden = args.nn_hidden
        nn_epochs = args.nn_epochs
        nn_lr = args.nn_lr
        verbose = args.verbose

        # --- Fetch market data ---
        start_time = time.time()
        try:
            S0_live, sigma_live = MarketDataFetcher.get_live_quote(ticker)
        except Exception as e:
            logging.error(f"Failed to fetch market data: {e}")
            return

        if S0_live not in s0_list:
            s0_list.append(int(S0_live))
        s0_list = sorted(set(s0_list))

        # --- Heston Model Parameters ---
        heston_params = {
            "v0": None,  # will set after sigma is chosen
            "kappa": 2.0,
            "theta": None,  # will set after sigma is chosen
            "xi": 0.3,
            "rho": -0.7
        }

        # --- User Input: Implied Volatility ---
        try:
            manual_iv = input("Enter an implied volatility (as a decimal, e.g. 0.45) or press Enter to auto-fetch: ").strip()
        except Exception:
            manual_iv = ""
        if manual_iv:
            try:
                sigma = float(manual_iv)
                logging.info(f"Using user-supplied implied volatility: {sigma:.2%}")
            except Exception:
                logging.warning("Invalid input. Falling back to auto-fetch.")
                sigma_iv = get_live_iv(ticker, expiry_str, K, option_type)
                if not np.isnan(sigma_iv):
                    logging.info(f"Using live implied volatility: {sigma_iv:.2%}")
                    sigma = sigma_iv
                else:
                    logging.info(f"Falling back to historical volatility: {sigma_live:.2%}")
                    sigma = sigma_live
        else:
            sigma_iv = get_live_iv(ticker, expiry_str, K, option_type)
            if not np.isnan(sigma_iv):
                logging.info(f"Using live implied volatility: {sigma_iv:.2%}")
                sigma = sigma_iv
            else:
                logging.info(f"Falling back to historical volatility: {sigma_live:.2%}")
                sigma = sigma_live

        # --- Finalize Heston Parameters ---
        heston_params["v0"] = sigma**2
        heston_params["theta"] = sigma**2

        # --- Model Selection ---
        model_choice = args.model
        run_bs = model_choice in ("bs", "both")
        run_heston = model_choice in ("heston", "both")

        # --- Advanced Features: Greeks, Exotic Option, Calibration, Dashboard ---
        T_live = days_to_expiry / 365
        greeks = BlackScholesGreeks.greeks(S0_live, K, T_live, r, sigma, option_type)
        logging.info("Black-Scholes Greeks at S0_live:")
        for k, v in greeks.items():
            logging.info(f"  {k}: {v:.4f}")

        ExoticOptionPricer.price_barrier_option()
        ModelCalibrator.calibrate_heston_to_market()
        # launch_dashboard()  # Uncomment to launch dashboard stub

        # --- Estimate Compute Time for One S0 (Black-Scholes) ---
        if run_bs:
            logging.info(f"Estimating compute time using S0 = {S0_live} (Black-Scholes)...")
            start_test = time.time()
            test_pricer = OptionPricer(K, r, sigma, option_type, lsm_poly_degree, seed, use_heston=False)
            _ = test_pricer.compute_curve_for_S0(
                S0_live, intervals_per_day, total_points, num_simulations, plot_paths
            )
            elapsed_single = time.time() - start_test
            logging.info(f"Time for one S0 curve: {elapsed_single:.2f} seconds")
            num_S0 = len(s0_list)
            num_workers = MAX_WORKERS
            est_total = elapsed_single * num_S0 / num_workers
            logging.info(f"Estimated total compute time: {est_total:.2f} seconds ({est_total/60:.1f} minutes)")

        # --- Black-Scholes (Constant Volatility) Curves ---
        if run_bs:
            args_bs = [
                (S0, K, r, sigma, option_type, lsm_poly_degree, seed,
                 intervals_per_day, total_points, num_simulations, plot_paths, False, None,
                 nn_hidden, nn_epochs, nn_lr, verbose)
                for S0 in s0_list
            ]
            records_bs = []
            try:
                with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    futures = [executor.submit(compute_curve_worker, *arg) for arg in args_bs]
                    for future in tqdm(as_completed(futures), total=len(futures), desc="S0 curves (BS)"):
                        records_bs.extend(future.result())
                df_bs = pd.DataFrame(records_bs)
                logging.info("Sample Black-Scholes results:")
                logging.info(df_bs.head(10))
                Plotter.plot_curves(df_bs, s0_list, S0_live, K, sigma, r, option_type, ticker, "Black-Scholes")
            except Exception as e:
                logging.error(f"Error in Black-Scholes batch: {e}")

        # --- Heston Curves ---
        if run_heston:
            args_heston = [
                (S0, K, r, sigma, option_type, lsm_poly_degree, seed,
                 intervals_per_day, total_points, num_simulations, plot_paths, True, heston_params,
                 nn_hidden, nn_epochs, nn_lr, verbose)
                for S0 in s0_list
            ]
            records_heston = []
            try:
                with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    futures = [executor.submit(compute_curve_worker, *arg) for arg in args_heston]
                    for future in tqdm(as_completed(futures), total=len(futures), desc="S0 curves (Heston)"):
                        records_heston.extend(future.result())
                df_heston = pd.DataFrame(records_heston)
                logging.info("Sample Heston results:")
                logging.info(df_heston.head(10))
                Plotter.plot_curves(df_heston, s0_list, S0_live, K, sigma, r, option_type, ticker, "Heston")
            except Exception as e:
                logging.error(f"Error in Heston batch: {e}")

        elapsed = time.time() - start_time
        logging.info(f"Time it took to compute: {elapsed:.2f} seconds")

    except Exception as e:
        logging.error(f"Fatal error in main: {e}")

if __name__ == "__main__":
    main()



