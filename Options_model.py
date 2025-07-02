import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf # type: ignore
import torch
import torch.nn as nn
import torch.optim as optim
import datetime
import plotly.graph_objects as go # type: ignore
import pandas as pd # type: ignore
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

class ContNet(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x)

def get_live_quote(ticker: str, vol_window: str = "1y"):
    """
    Fetch current spot price and estimate annualized volatility from historical data.
    """
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

def price_american_option(
    S0,
    K,
    T,
    r,
    sigma,
    num_simulations=10000,
    num_time_steps=50,
    option_type="call",
    lsm_poly_degree=2,
    plot_paths=False,
    seed=42
):
    """
    Prices an American option using the Longstaff-Schwartz least-squares Monte Carlo method,
    with a neural network for the continuation value regression.
    Returns: (mean, std, probability expires worthless)
    """
    # --- Input validation ---
    if S0 <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("S0, K, T, and sigma must be positive.")
    if r < 0:
        raise ValueError("r must be non-negative.")
    if num_simulations <= 0 or num_time_steps <= 0:
        raise ValueError("num_simulations and num_time_steps must be positive integers.")
    if lsm_poly_degree < 0 or not isinstance(lsm_poly_degree, int):
        raise ValueError("lsm_poly_degree must be a non-negative integer.")
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")

    np.random.seed(seed)
    dt = T / num_time_steps
    discount_factor = np.exp(-r * dt)

    # --- Simulate stock price paths using vectorization and antithetic variates ---
    M = num_simulations // 2 * 2  # Ensure even number for antithetic variates
    drift = (r - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)
    Z = np.random.standard_normal((num_time_steps, M // 2))
    Z = np.concatenate([Z, -Z], axis=1)  # Antithetic variates

    stock = np.zeros((num_time_steps + 1, M))
    stock[0] = S0
    for t in range(1, num_time_steps + 1):
        stock[t] = stock[t - 1] * np.exp(drift + diffusion * Z[t - 1])

    # --- Plot simulated paths if requested ---
    if plot_paths:
        plt.figure(figsize=(10, 6))
        for i in range(min(100, M)):
            plt.plot(np.linspace(0, T, num_time_steps + 1), stock[:, i], alpha=0.5)
        plt.title("Simulated Stock Price Paths")
        plt.xlabel("Time to Maturity")
        plt.ylabel("Stock Price")
        plt.grid()
        plt.show()

    # --- Payoff function ---
    if option_type == "call":
        payoff = lambda S: np.maximum(S - K, 0)
    else:
        payoff = lambda S: np.maximum(K - S, 0)

    # --- Initialize cashflows at maturity ---
    cashflows = payoff(stock[-1])
    exercised = np.zeros(M, dtype=bool)

    discount = np.exp(-r * dt)
    for t in range(num_time_steps - 1, 0, -1):
        cashflows *= discount

        itm = (payoff(stock[t]) > 0) & (~exercised)
        if not np.any(itm):
            continue

        X = stock[t, itm]
        Y = cashflows[itm]

        # Neural network regression for continuation value
        Xs = (X - X.mean()) / X.std() if X.std() > 0 else X - X.mean()
        Xs = Xs.reshape(-1, 1)
        Y = Y.reshape(-1, 1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = ContNet().to(device)
        opt = optim.Adam(net.parameters(), lr=1e-3)

        X_tensor = torch.from_numpy(Xs).float().to(device)
        Y_tensor = torch.from_numpy(Y).float().to(device)

        for _ in range(10):
            pred = net(X_tensor)
            loss = nn.MSELoss()(pred, Y_tensor)
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            continuation = net(X_tensor).cpu().numpy().flatten()

        immediate = payoff(X)
        to_exercise = immediate > continuation
        idx_itm = np.where(itm)[0]
        ex_idx = idx_itm[to_exercise]

        cashflows[ex_idx] = immediate[to_exercise]
        exercised[ex_idx] = True

    # No extra discount here—cashflows are now at t=0
    est_price = cashflows.mean()
    std_price = cashflows.std()
    zero_prob = np.mean(cashflows == 0)

    return est_price, std_price, zero_prob

def price_from_ticker(
    ticker: str,
    K: float,
    T: float,
    r: float,
    option_type: str = 'call',
    num_simulations: int = 10000,
    num_time_steps: int = 50,
    lsm_poly_degree: int = 2,
    plot_paths: bool = False,
    seed: int = 42,    sigma: float = None
):
    """
    Fetch live S0 and sigma for `ticker`, then price the American option.
    If sigma is provided, use it instead of historical estimate.
    """
    S0, sigma_hist = get_live_quote(ticker)
    if sigma is None:
        sigma_to_use = sigma_hist
    else:
        sigma_to_use = sigma
    print(f"Fetched {ticker}: S0={S0:.2f}, sigma={sigma_to_use:.2%}")
    price, std, zero_prob = price_american_option(
        S0, K, T, r, sigma_to_use,
        num_simulations, num_time_steps,
        option_type, lsm_poly_degree,
        plot_paths, seed
    )
    print(f"Estimated American {option_type} on {ticker}: ${price:.4f} (Std: {std:.4f}, P(Worthless): {zero_prob:.2%})")
    return price

def compute_curve_for_S0(S0, K, r, sigma, num_simulations, intervals_per_day, total_points,
                         option_type, lsm_poly_degree, plot_paths, seed):
    records = []
    for i in range(total_points, 0, -1):
        d = i / intervals_per_day
        T = d / 365
        steps = max(10, min(130, int(np.ceil(d))))
        np.random.seed(seed)
        est_price, std_price, zero_prob = price_american_option(
            S0, K, T, r, sigma,
            num_simulations, steps,
            option_type, lsm_poly_degree,
            plot_paths, seed
        )
        records.append({
            'S0': S0,
            'Days to Expiry': d,
            'Option Value': est_price,
            'Std Dev': std_price,
            'Zero Prob': zero_prob
        })
    return records

