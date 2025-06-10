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
import plotly.io as pio

pio.renderers.default = "browser"

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

def simulate_heston_paths(
    S0, v0, r, kappa, theta, xi, rho,
    T, num_steps, num_paths, seed=None
):
    """
    Simulate stock and variance paths under the Heston model.
    Returns: stock_paths, var_paths (shape: [num_steps+1, num_paths])
    """
    if seed is not None:
        np.random.seed(seed)
    dt = T / num_steps
    S = np.zeros((num_steps + 1, num_paths))
    v = np.zeros((num_steps + 1, num_paths))
    S[0] = S0
    v[0] = v0
    for t in range(1, num_steps + 1):
        Z1 = np.random.standard_normal(num_paths)
        Z2 = np.random.standard_normal(num_paths)
        dW1 = Z1
        dW2 = rho * Z1 + np.sqrt(1 - rho ** 2) * Z2  # Correlated Brownian motions

        v_prev = np.maximum(v[t-1], 0)
        v[t] = v_prev + kappa * (theta - v_prev) * dt + xi * np.sqrt(v_prev * dt) * dW2
        v[t] = np.maximum(v[t], 0)  # Full truncation

        S[t] = S[t-1] * np.exp((r - 0.5 * v_prev) * dt + np.sqrt(v_prev * dt) * dW1)
    return S, v

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
    seed=42,
    use_heston=False,
    heston_params=None
):
    """
    Prices an American option using the Longstaff-Schwartz least-squares Monte Carlo method,
    with a neural network for the continuation value regression.
    Supports both constant and stochastic volatility (Heston).
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
    M = num_simulations // 2 * 2  # Ensure even number for antithetic variates

    # --- Simulate stock price paths ---
    if use_heston:
        # Set default Heston parameters if not provided
        if heston_params is None:
            heston_params = {
                "v0": sigma**2,
                "kappa": 2.0,
                "theta": sigma**2,
                "xi": 0.3,
                "rho": -0.7
            }
        stock, var = simulate_heston_paths(
            S0, heston_params["v0"], r, heston_params["kappa"], heston_params["theta"],
            heston_params["xi"], heston_params["rho"],
            T, num_time_steps, M, seed
        )
    else:
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

        for _ in range(20):  # Try 20-50 for better fit
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
    lower = max(0, est_price - std_price)
    upper = est_price + std_price

    zero_prob = np.mean(cashflows == 0)

    return est_price

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
    seed: int = 42,
    sigma: float = None,
    use_heston: bool = False,
    heston_params: dict = None
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
    price = price_american_option(
        S0, K, T, r, sigma_to_use,
        num_simulations, num_time_steps,
        option_type, lsm_poly_degree,
        plot_paths, seed,
        use_heston=use_heston,
        heston_params=heston_params
    )
    print(f"Estimated American {option_type} on {ticker}: ${price:.4f}")
    return price

def compute_curve_for_S0(
    S0, K, r, sigma, num_simulations, intervals_per_day, total_points,
    option_type, lsm_poly_degree, plot_paths, seed,
    use_heston=False, heston_params=None
):
    records = []
    base = datetime.datetime.combine(datetime.date.today(), datetime.time())
    for i in range(1, total_points + 1):
        d = i / intervals_per_day
        T = d / 365

        steps = max(10, min(130, int(np.ceil(d))))
        np.random.seed(seed)
        est_price = price_american_option(
            S0, K, T, r, sigma,
            num_simulations, steps,
            option_type, lsm_poly_degree,
            plot_paths, seed,
            use_heston=use_heston,
            heston_params=heston_params
        )
        date = base + datetime.timedelta(days=d)
        records.append({
            'S0': S0,
            'Days to Expiry': d,
            'Option Value': est_price,
            'Date': date
        })
    return records

def get_live_iv(ticker, expiry, strike, option_type='put'):
    import yfinance as yf
    stock = yf.Ticker(ticker)
    chain = stock.option_chain(expiry)
    df = chain.puts if option_type == 'put' else chain.calls
    row = df[df['strike'] == strike]
    if row.empty:
        raise ValueError("Strike not found in option chain.")
    return row['impliedVolatility'].values[0]

# Example usage:
if __name__ == "__main__":
    # --- USER INPUT ---
    ticker = "TSLA"
    K = 330
    r = 0.0018
    option_type = 'call'  # 'call' or 'put'
    expiry_str = "2025-06-06"  # <--- Enter expiry as string here (YYYY-MM-DD)
    num_simulations = 1000000
    num_time_steps = 160
    lsm_poly_degree = 2
    plot_paths = False
    seed = 2025

    # For compute_curve_for_S0
    intervals_per_day = 12      # How many intervals per day for the curve
    total_points = 100         # How many points in the curve

    # For S0 curve sweep
    s0_curve_min = 270         # Minimum S0 for curve
    s0_curve_max = 300         # Maximum S0 for curve
    s0_curve_steps = 3         # Number of S0 points in the curve
    # --- END USER INPUT ---

    # Convert expiry string to datetime.date
    expiry = datetime.datetime.strptime(expiry_str, "%Y-%m-%d").date()

    start_time = time.time()
    S0_live, sigma_live = get_live_quote(ticker)
    today = datetime.date.today()
    days_to_expiry = (expiry - today).days

    # --- Use live IV from option chain ---
    expiry_str = expiry.strftime("%Y-%m-%d")  # Ensure correct format for yfinance
    try:
        sigma_override = get_live_iv(ticker, expiry_str, K, option_type)
        print(f"Using live IV from option chain: {sigma_override:.2%}")
    except Exception as e:
        print(f"Could not fetch live IV, falling back to historical: {e}")
        sigma_override = sigma_live

    sigma = sigma_override

    # --- Compare constant volatility vs Heston ---
    print("\n--- Constant Volatility Model ---")
    price_bs = price_from_ticker(
        ticker, K, days_to_expiry/365, r, option_type,
        num_simulations, num_time_steps, lsm_poly_degree,
        plot_paths, seed, sigma=sigma, use_heston=False
    )

    print("\n--- Heston Stochastic Volatility Model ---")
    heston_params = {
        "v0": sigma**2,
        "kappa": 2.0,
        "theta": sigma**2,
        "xi": 0.3,
        "rho": -0.7
    }
    price_heston = price_from_ticker(
        ticker, K, days_to_expiry/365, r, option_type,
        num_simulations, num_time_steps, lsm_poly_degree,
        plot_paths, seed, sigma=sigma, use_heston=True, heston_params=heston_params
    )

    print(f"\nConstant Volatility Price: ${price_bs:.4f}")
    print(f"Heston Model Price:        ${price_heston:.4f}")

    # --- S0 Curve Calculation Example ---
    print("\n--- S0 Curve Calculation ---")
    s0_values = np.linspace(s0_curve_min, s0_curve_max, s0_curve_steps)
    option_values = []
    for S0_test in s0_values:
        val = price_american_option(
            S0_test, K, days_to_expiry/365, r, sigma,
            num_simulations, num_time_steps,
            option_type, lsm_poly_degree,
            plot_paths, seed,
            use_heston=use_heston,
            heston_params=heston_params if use_heston else None
        )
        option_values.append(val)
    print("S0 sweep complete.")

    # --- S0 Curve Interactive Plot (Plotly) ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s0_values,
        y=option_values,
        mode='lines+markers',
        name="Option Value",
        line=dict(width=3, color='blue'),
        marker=dict(size=8, color='blue'),
        hovertemplate=(
            'S0: $%{x}<br>'
            'Option Value: %{y:.4f}<extra></extra>'
        )
    ))

    model_name = "Heston Stochastic Volatility" if use_heston else "Constant Volatility"
    fig.update_layout(
        title=dict(
            text=f"American {option_type.capitalize()} Option Value vs. Spot Price (S0)<br>"
                 f"<sup>{ticker} | K=${K} | σ={sigma:.2f} | r={r:.2%} | Expiry: {expiry_str} | Model: {model_name}</sup>",
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title="Spot Price (S0)",
            showgrid=True,
            ticks="outside",
            showline=True,
            linewidth=2,
            linecolor='black'
        ),
        yaxis=dict(
            title="Option Value",
            showgrid=True,
            ticks="outside",
            showline=True,
            linewidth=2,
            linecolor='black'
        ),
        template="plotly_white",
        legend=dict(
            title="Legend",
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.5)"
        ),
        dragmode='pan'
    )
    fig.show()

    # --- Detailed S0 Curve Calculation for Multiple S0s ---
    print("\n--- Detailed S0 Curve Calculation for Multiple S0s ---")
    all_records = []
    args = [
        (S0_test, K, r, sigma, num_simulations,
         intervals_per_day, total_points,
         option_type, lsm_poly_degree, plot_paths, seed,
         use_heston, heston_params if use_heston else None)
        for S0_test in s0_values
    ]

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(compute_curve_for_S0, *arg) for arg in args]
        for future in as_completed(futures):
            all_records.extend(future.result())

    df = pd.DataFrame(all_records)

    # --- Plotly Interactive Plot for Multiple S0 Curves ---
    fig = go.Figure()
    for S0_test in s0_values:
        curve = df[df['S0'] == S0_test]
        fig.add_trace(go.Scatter(
            x=curve['Days to Expiry'],
            y=curve['Option Value'],
            mode='lines',
            name=f"S0 = ${S0_test:.2f}",
            line=dict(width=4 if np.isclose(S0_test, S0_live, atol=1e-2) else 2, dash='solid' if np.isclose(S0_test, S0_live, atol=1e-2) else 'dot'),
            hovertemplate=(
                'S0: $%{text}<br>'
                'Days to Expiry: %{x:.2f}<br>'
                'Option Value: %{y:.4f}<extra></extra>'
            ),
            text=[f"{S0_test:.2f}"]*len(curve)
        ))

    fig.update_layout(
        title=dict(
            text=f"American {option_type.capitalize()} Option Value vs. Days to Expiry<br>"
                 f"<sup>{ticker} | K=${K} | σ={sigma:.2f} | r={r:.2%} | Expiry: {expiry_str} | Model: {model_name}</sup>",
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

