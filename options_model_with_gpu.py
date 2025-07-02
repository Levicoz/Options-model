import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yfinance as yf
import datetime
import pandas as pd
import plotly.graph_objects as go
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
    return float(S0), float(sigma)

def price_american_option_gpu(
    S0, K, T, r, sigma,
    num_simulations=10000,
    num_time_steps=50,
    option_type="call",
    device=None,
    seed=42
):
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    dt = T / num_time_steps
    M = num_simulations // 2 * 2  # even for antithetic
    drift = (r - 0.5 * sigma ** 2) * dt
    diffusion = sigma * (dt ** 0.5)

    # Simulate paths on GPU
    Z = torch.randn(num_time_steps, M // 2, device=device)
    Z = torch.cat([Z, -Z], dim=1)
    stock = torch.zeros(num_time_steps + 1, M, device=device)
    stock[0] = S0
    for t in range(1, num_time_steps + 1):
        stock[t] = stock[t - 1] * torch.exp(drift + diffusion * Z[t - 1])

    # Payoff
    if option_type == "call":
        payoff = lambda S: torch.clamp(S - K, min=0)
    else:
        payoff = lambda S: torch.clamp(K - S, min=0)

    cashflows = payoff(stock[-1])
    exercised = torch.zeros(M, dtype=torch.bool, device=device)
    discount = torch.exp(torch.tensor(-r * dt, device=device))

    for t in range(num_time_steps - 1, 0, -1):
        cashflows = cashflows * discount
        itm = (payoff(stock[t]) > 0) & (~exercised)
        if itm.sum() == 0:
            continue

        X = stock[t, itm].unsqueeze(1)
        Y = cashflows[itm].unsqueeze(1)

        # Neural net regression on GPU
        net = ContNet().to(device)
        opt = optim.Adam(net.parameters(), lr=1e-3)
        for _ in range(10):
            pred = net(X)
            loss = nn.MSELoss()(pred, Y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            continuation = net(X).squeeze(1)

        immediate = payoff(X.squeeze(1))
        to_exercise = immediate > continuation
        idx_itm = torch.where(itm)[0]
        ex_idx = idx_itm[to_exercise]

        cashflows[ex_idx] = immediate[to_exercise]
        exercised[ex_idx] = True

    est_price = cashflows.mean().item()
    std_price = cashflows.std().item()
    zero_prob = (cashflows == 0).float().mean().item()
    return est_price, std_price, zero_prob

def compute_curve_for_S0_gpu(S0, K, r, sigma, num_simulations, intervals_per_day, total_points,
                         option_type, plot_paths, seed):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    records = []
    for i in range(total_points, 0, -1):
        d = i / intervals_per_day
        T = d / 365
        steps = max(10, min(130, int(np.ceil(d))))
        est_price, std_price, zero_prob = price_american_option_gpu(
            S0, K, T, r, sigma,
            num_simulations=num_simulations,
            num_time_steps=steps,
            option_type=option_type,
            device=device,
            seed=seed
        )
        records.append({
            'S0': S0,
            'Days to Expiry': d,
            'Option Value': est_price,
            'Std Dev': std_price,
            'Zero Prob': zero_prob
        })
    return records

if __name__ == "__main__":
    # --- User parameters ---
    ticker = "AAPL"
    K = 100
    r = 0.03
    option_type = "call"
    sigma_override = None  # Set to a float to override, or None to use live
    num_simulations = 10000
    intervals_per_day = 2
    expiry = datetime.date.today() + datetime.timedelta(days=90)
    s0_start = 80
    s0_end = 120
    s0_step = 5
    plot_paths = False
    seed = 42
    MAX_WORKERS = 4

    # --- Fetch live quote and volatility ---
    S0_live, sigma_live = get_live_quote(ticker)
    sigma = sigma_override if sigma_override is not None else sigma_live
    today = datetime.date.today()
    days_to_expiry = (expiry - today).days
    total_points = days_to_expiry * int(intervals_per_day)

    # --- S0 sweep ---
    s0_list = list(range(int(s0_start), int(s0_end) + 1, int(s0_step)))
    if int(S0_live) not in s0_list:
        s0_list.append(int(S0_live))
    s0_list = sorted(set(s0_list))

    print(f"Running American {option_type} option analysis for {ticker}")
    print(f"S0 live: {S0_live:.2f}, sigma: {sigma:.4f}, K: {K}, r: {r}, expiry: {expiry}")
    print(f"Simulations: {num_simulations}, Intervals/day: {intervals_per_day}, S0 range: {s0_list}")

    # --- Estimate time for one S0 ---
    print("Estimating time for one S0 curve...")
    t0 = time.time()
    _ = compute_curve_for_S0_gpu(
        S0_live, K, r, sigma, num_simulations, intervals_per_day, total_points,
        option_type, plot_paths, seed
    )
    t1 = time.time()
    print(f"Time for one S0: {t1-t0:.2f} seconds")

    # --- Parallel computation ---
    records = []
    for S0 in tqdm(s0_list, desc="S0 curves"):
        records.extend(compute_curve_for_S0_gpu(
            S0, K, r, sigma, num_simulations, intervals_per_day, total_points,
            option_type, plot_paths, seed
        ))

    df = pd.DataFrame(records)

    # --- Plot results ---
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
        linecolor='black'
    )
    fig.show()

    print("Done.")