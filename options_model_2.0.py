import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf  # type: ignore
import torch
import torch.nn as nn
import torch.optim as optim
import datetime
import plotly.graph_objects as go  # type: ignore
import pandas as pd  # type: ignore
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import logging
from typing import Optional, Dict, List, Any
import random
import plotly.io as pio

# --- Logging configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
pio.renderers.default = "browser"

# --- Helper: Get Live Implied Volatility ---
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

# --- Heston Path Simulator ---
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

# --- Option Pricer ---
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

        # Set all seeds for reproducibility
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)

        dt = T / num_time_steps
        discount_factor = np.exp(-self.r * dt)

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

        if plot_paths:
            plt.figure(figsize=(10, 6))
            for i in range(min(100, M)):
                plt.plot(np.linspace(0, T, num_time_steps + 1), S[:, i], alpha=0.5)
            plt.title("Simulated Stock Price Paths")
            plt.xlabel("Time to Maturity")
            plt.ylabel("Stock Price")
            plt.grid()
            plt.show()

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

# --- Plotter ---
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

# --- Parallel Worker ---
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

# --- Main Block ---
if __name__ == "__main__":
    ticker = "AMD"
    start_time = time.time()
    S0_live, sigma_live = MarketDataFetcher.get_live_quote(ticker)
    expiry = datetime.date(2025, 6, 13)
    today = datetime.date.today()
    days_to_expiry = (expiry - today).days
    expiry_str = expiry.strftime("%Y-%m-%d")
    K = 124
    r = 0.05
    option_type = 'call'
    num_simulations = 1000000
    num_time_steps = 150
    lsm_poly_degree = 2
    plot_paths = False
    seed = 2025

    intervals_per_day = 4
    total_points = days_to_expiry * intervals_per_day

    s0_start = 110
    s0_end = 130
    s0_step = 2
    s0_list = list(range(s0_start, s0_end + 1, s0_step))
    if S0_live not in s0_list:
        s0_list.append(int(S0_live))
    s0_list = sorted(set(s0_list))

    MAX_WORKERS = 6

    # --- Neural network parameters (configurable) ---
    nn_hidden = 32
    nn_epochs = 10
    nn_lr = 1e-3
    verbose = False

    # --- Heston parameters (set after sigma selection) ---
    heston_params = {
        "v0": None,  # will set after sigma is chosen
        "kappa": 2.0,
        "theta": None,  # will set after sigma is chosen
        "xi": 0.3,
        "rho": -0.7
    }

    # --- User IV input ---
    manual_iv = input("Enter an implied volatility (as a decimal, e.g. 0.45) or press Enter to auto-fetch: ").strip()
    if manual_iv:
        try:
            sigma = float(manual_iv)
            print(f"Using user-supplied implied volatility: {sigma:.2%}")
        except Exception:
            print("Invalid input. Falling back to auto-fetch.")
            sigma_iv = get_live_iv(ticker, expiry_str, K, option_type)
            if not np.isnan(sigma_iv):
                print(f"Using live implied volatility: {sigma_iv:.2%}")
                sigma = sigma_iv
            else:
                print(f"Falling back to historical volatility: {sigma_live:.2%}")
                sigma = sigma_live
    else:
        sigma_iv = get_live_iv(ticker, expiry_str, K, option_type)
        if not np.isnan(sigma_iv):
            print(f"Using live implied volatility: {sigma_iv:.2%}")
            sigma = sigma_iv
        else:
            print(f"Falling back to historical volatility: {sigma_live:.2%}")
            sigma = sigma_live

    # Now update Heston params after sigma is finalized
    heston_params["v0"] = sigma**2
    heston_params["theta"] = sigma**2

    # --- Model selection prompt ---
    print("Which model(s) do you want to run?")
    print("1: Black-Scholes only")
    print("2: Heston only")
    print("3: Both")
    model_choice = input("Enter 1, 2, or 3: ").strip()
    run_bs = model_choice in ("1", "3")
    run_heston = model_choice in ("2", "3")

    # Estimate time for one S0 (Black-Scholes)
    if run_bs:
        print(f"Estimating compute time using S0 = {S0_live} (Black-Scholes)...")
        start_test = time.time()
        test_pricer = OptionPricer(K, r, sigma, option_type, lsm_poly_degree, seed, use_heston=False)
        _ = test_pricer.compute_curve_for_S0(
            S0_live, intervals_per_day, total_points, num_simulations, plot_paths
        )
        elapsed_single = time.time() - start_test
        print(f"Time for one S0 curve: {elapsed_single:.2f} seconds")

        num_S0 = len(s0_list)
        num_workers = MAX_WORKERS
        est_total = elapsed_single * num_S0 / num_workers
        print(f"Estimated total compute time: {est_total:.2f} seconds ({est_total/60:.1f} minutes)\n")

    # --- Black-Scholes (Constant Volatility) Curves ---
    if run_bs:
        args_bs = [
            (S0, K, r, sigma, option_type, lsm_poly_degree, seed,
             intervals_per_day, total_points, num_simulations, plot_paths, False, None,
             nn_hidden, nn_epochs, nn_lr, verbose)
            for S0 in s0_list
        ]
        records_bs = []
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(compute_curve_worker, *arg) for arg in args_bs]
            for future in tqdm(as_completed(futures), total=len(futures), desc="S0 curves (BS)"):
                records_bs.extend(future.result())
        df_bs = pd.DataFrame(records_bs)
        print("Sample Black-Scholes results:")
        print(df_bs.head(10))
        Plotter.plot_curves(df_bs, s0_list, S0_live, K, sigma, r, option_type, ticker, "Black-Scholes")

    # --- Heston Curves ---
    if run_heston:
        args_heston = [
            (S0, K, r, sigma, option_type, lsm_poly_degree, seed,
             intervals_per_day, total_points, num_simulations, plot_paths, True, heston_params,
             nn_hidden, nn_epochs, nn_lr, verbose)
            for S0 in s0_list
        ]
        records_heston = []
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(compute_curve_worker, *arg) for arg in args_heston]
            for future in tqdm(as_completed(futures), total=len(futures), desc="S0 curves (Heston)"):
                records_heston.extend(future.result())
        df_heston = pd.DataFrame(records_heston)
        print("Sample Heston results:")
        print(df_heston.head(10))
        Plotter.plot_curves(df_heston, s0_list, S0_live, K, sigma, r, option_type, ticker, "Heston")

    elapsed = time.time() - start_time
    print(f"\nTime it took to compute: {elapsed:.2f} seconds")



