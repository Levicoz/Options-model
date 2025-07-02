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

class MarketDataFetcher:
    @staticmethod
    def get_live_quote(ticker: str, vol_window: str = "1y"):
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

class OptionPricer:
    def __init__(self, K, r, sigma, option_type='call', lsm_poly_degree=2, seed=42):
        self.K = K
        self.r = r
        self.sigma = sigma
        self.option_type = option_type
        self.lsm_poly_degree = lsm_poly_degree
        self.seed = seed

    def price_american_option(self, S0, T, num_simulations=10000, num_time_steps=50, plot_paths=False):
        if S0 <= 0 or self.K <= 0 or T <= 0 or self.sigma <= 0:
            raise ValueError("S0, K, T, and sigma must be positive.")
        if self.r < 0:
            raise ValueError("r must be non-negative.")
        if num_simulations <= 0 or num_time_steps <= 0:
            raise ValueError("num_simulations and num_time_steps must be positive integers.")
        if self.lsm_poly_degree < 0 or not isinstance(self.lsm_poly_degree, int):
            raise ValueError("lsm_poly_degree must be a non-negative integer.")
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'.")

        np.random.seed(self.seed)
        dt = T / num_time_steps
        discount_factor = np.exp(-self.r * dt)

        M = num_simulations // 2 * 2
        drift = (self.r - 0.5 * self.sigma ** 2) * dt
        diffusion = self.sigma * np.sqrt(dt)
        Z = np.random.standard_normal((num_time_steps, M // 2))
        Z = np.concatenate([Z, -Z], axis=1)

        stock = np.zeros((num_time_steps + 1, M))
        stock[0] = S0
        for t in range(1, num_time_steps + 1):
            stock[t] = stock[t - 1] * np.exp(drift + diffusion * Z[t - 1])

        if plot_paths:
            plt.figure(figsize=(10, 6))
            for i in range(min(100, M)):
                plt.plot(np.linspace(0, T, num_time_steps + 1), stock[:, i], alpha=0.5)
            plt.title("Simulated Stock Price Paths")
            plt.xlabel("Time to Maturity")
            plt.ylabel("Stock Price")
            plt.grid()
            plt.show()

        if self.option_type == "call":
            payoff = lambda S: np.maximum(S - self.K, 0)
        else:
            payoff = lambda S: np.maximum(self.K - S, 0)

        cashflows = payoff(stock[-1])
        exercised = np.zeros(M, dtype=bool)

        discount = np.exp(-self.r * dt)
        for t in range(num_time_steps - 1, 0, -1):
            cashflows *= discount

            itm = (payoff(stock[t]) > 0) & (~exercised)
            if not np.any(itm):
                continue

            X = stock[t, itm]
            Y = cashflows[itm]

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

        est_price = cashflows.mean()
        std_price = cashflows.std()
        lower = max(0, est_price - std_price)
        upper = est_price + std_price

        zero_prob = np.mean(cashflows == 0)
        print(f"Probability option expires worthless: {zero_prob:.2%}")

        print(
            f"Estimated American {self.option_type} price: ${est_price:.4f} "
            f"(S0={S0}, K={self.K}, T={T}, r={self.r}, sigma={self.sigma}, "
            f"simulations={num_simulations}, steps={num_time_steps})"
        )
        print(
            f"One standard deviation range: "
            f"${lower:.4f} to ${upper:.4f}"
        )

        print(f"Mean: ${est_price:.4f}")
        print(f"Std Dev: ${std_price:.4f}")
        print(f"Min: ${cashflows.min():.4f}")
        print(f"Max: ${cashflows.max():.4f}")
        print(f"Probability expires worthless: {zero_prob:.2%}")

        return est_price

    def compute_curve_for_S0(self, S0, intervals_per_day, total_points, num_simulations, plot_paths):
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

class Plotter:
    @staticmethod
    def plot_curves(df, s0_list, S0_live, K, sigma, r, option_type, ticker):
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
                text=f"American {option_type.capitalize()} Option Value vs. Days to Expiry<br><sup>{ticker} | K=${K} | σ={sigma:.2f} | r={r:.2%}</sup>",
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

def compute_curve_worker(S0, K, r, sigma, option_type, lsm_poly_degree, seed,
                        intervals_per_day, total_points, num_simulations, plot_paths):
    pricer = OptionPricer(K, r, sigma, option_type, lsm_poly_degree, seed)
    return pricer.compute_curve_for_S0(S0, intervals_per_day, total_points, num_simulations, plot_paths)

if __name__ == "__main__":
    ticker = "AMD"
    start_time = time.time()
    S0_live, sigma_live = MarketDataFetcher.get_live_quote(ticker)
    sigma_override = 0.44
    sigma = sigma_override if sigma_override is not None else sigma_live
    K = 124
    r = 0.05
    option_type = 'call'
    num_simulations = 10000
    num_time_steps = 150
    lsm_poly_degree = 2
    plot_paths = False
    seed = 2025

    expiry = datetime.date(2025, 7, 6)
    today = datetime.date.today()
    days_to_expiry = (expiry - today).days

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

    # Estimate time for one S0
    print(f"Estimating compute time using S0 = {S0_live}...")
    start_test = time.time()
    test_pricer = OptionPricer(K, r, sigma, option_type, lsm_poly_degree, seed)
    _ = test_pricer.compute_curve_for_S0(
        S0_live, intervals_per_day, total_points, num_simulations, plot_paths
    )
    elapsed_single = time.time() - start_test
    print(f"Time for one S0 curve: {elapsed_single:.2f} seconds")

    num_S0 = len(s0_list)
    num_workers = MAX_WORKERS
    est_total = elapsed_single * num_S0 / num_workers
    print(f"Estimated total compute time: {est_total:.2f} seconds ({est_total/60:.1f} minutes)\n")

    args = [
        (S0, K, r, sigma, option_type, lsm_poly_degree, seed,
         intervals_per_day, total_points, num_simulations, plot_paths)
        for S0 in s0_list
    ]

    records = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(compute_curve_worker, *arg) for arg in args]
        for future in tqdm(as_completed(futures), total=len(futures), desc="S0 curves"):
            records.extend(future.result())

    df = pd.DataFrame(records)

    Plotter.plot_curves(df, s0_list, S0_live, K, sigma, r, option_type, ticker)

    elapsed = time.time() - start_time
    print(f"\nTime it took to compute: {elapsed:.2f} seconds")
