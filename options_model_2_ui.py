import streamlit as st
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import pandas as pd
import plotly.graph_objects as go

try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# Import your core functions/classes here
from options_model_2 import (
    MarketDataFetcher,
    get_live_iv,
    compute_curve_worker,
)

st.title("American Option Risk Analysis & Pricing")

st.markdown("""
Customize all parameters for American option pricing using Black-Scholes and Heston models.
""")

# --- User Inputs ---
col1, col2 = st.columns(2)
with col1:
    ticker = st.text_input("Ticker", value="AAPL")
    expiry = st.date_input("Expiry Date", value=datetime.date.today() + datetime.timedelta(days=90))
    K = st.number_input("Strike Price (K)", value=100.0)
    r = st.number_input("Risk-Free Rate (r)", value=0.03, format="%.4f")
    option_type = st.selectbox("Option Type", ["call", "put"])
    sigma_override = st.number_input("Volatility (σ, leave 0 for auto)", value=0.0, format="%.4f")
    model = st.selectbox("Model", ["Black-Scholes", "Heston", "Both"])
    verbose = st.checkbox("Verbose Output", value=False)

with col2:
    num_simulations = st.number_input("Number of Simulations", value=10000, step=1000)
    num_time_steps = st.number_input("Number of Time Steps", value=50, step=1)
    seed = st.number_input("Random Seed", value=2025, step=1)
    intervals_per_day = st.number_input("Intervals Per Day", value=2, step=1)
    s0_start = st.number_input("S0 Start", value=80, step=1)
    s0_end = st.number_input("S0 End", value=120, step=1)
    s0_step = st.number_input("S0 Step", value=5, step=1)
    max_workers = st.number_input("Max Parallel Workers", value=4, step=1)
    use_parallel = st.checkbox("Use Parallel Processing", value=False)

st.write("Inputs:", ticker, expiry, K, r, option_type, sigma_override, num_simulations, num_time_steps, seed, intervals_per_day, s0_start, s0_end, s0_step, max_workers, model, verbose)

# Set default LSM polynomial degree
lsm_poly_degree = 2

if st.button("Run Analysis"):
    try:
        S0_live, sigma_live = MarketDataFetcher.get_live_quote(ticker)
        expiry_str = expiry.strftime("%Y-%m-%d")
        sigma = sigma_override if sigma_override > 0 else sigma_live

        # Optionally fetch live IV
        sigma_iv = get_live_iv(ticker, expiry_str, K, option_type)
        if sigma_override == 0 and not pd.isna(sigma_iv):
            sigma = sigma_iv

        today = datetime.date.today()
        days_to_expiry = (expiry - today).days
        total_points = days_to_expiry * int(intervals_per_day)
        s0_list = list(range(int(s0_start), int(s0_end) + 1, int(s0_step)))
        if int(S0_live) not in s0_list:
            s0_list.append(int(S0_live))
        s0_list = sorted(set(s0_list))

        # Heston params (user could expose these as well)
        heston_params = {
            "v0": sigma**2,
            "kappa": 2.0,
            "theta": sigma**2,
            "xi": 0.3,
            "rho": -0.7
        }

        # Default NN hyperparameters
        nn_hidden = 32
        nn_epochs = 10
        nn_lr = 0.001

        args_bs = [
            (S0, K, r, sigma, option_type, lsm_poly_degree, int(seed),
             int(intervals_per_day), total_points, int(num_simulations), False,  # plot_paths is now always False
             False, None, nn_hidden, nn_epochs, nn_lr, verbose)
            for S0 in s0_list
        ]
        args_heston = [
            (S0, K, r, sigma, option_type, lsm_poly_degree, int(seed),
             int(intervals_per_day), total_points, int(num_simulations), False,  # plot_paths is now always False
             True, heston_params, nn_hidden, nn_epochs, nn_lr, verbose)
            for S0 in s0_list
        ]

        # Run pricing
        results = {}
        progress_bar = st.progress(0)
        total_models = 1 if model != "Both" else 2
        total_tasks = len(s0_list) * total_models

        def run_tasks(args_list):
            records = []
            completed = 0
            if use_parallel:
                with ProcessPoolExecutor(max_workers=int(max_workers)) as executor:
                    futures = [executor.submit(compute_curve_worker, *arg) for arg in args_list]
                    for future in as_completed(futures):
                        records.extend(future.result())
                        completed += 1
                        progress_bar.progress(completed / total_tasks)
            else:
                for arg in args_list:
                    records.extend(compute_curve_worker(*arg))
                    completed += 1
                    progress_bar.progress(completed / total_tasks)
            return records

        if model in ("Black-Scholes", "Both"):
            st.info("Running Black-Scholes model...")
            records_bs = run_tasks(args_bs)
            df_bs = pd.DataFrame(records_bs)
            results["Black-Scholes"] = df_bs

        if model in ("Heston", "Both"):
            st.info("Running Heston model...")
            records_heston = run_tasks(args_heston)
            df_heston = pd.DataFrame(records_heston)
            results["Heston"] = df_heston

        progress_bar.empty()

        # Plot results
        for model_name, df in results.items():
            if df.empty:
                st.warning(f"No results to plot for {model_name}.")
                continue
            fig = go.Figure()
            for S0 in s0_list:
                curve = df[df['S0'] == S0]
                fig.add_trace(go.Scatter(
                    x=curve['Days to Expiry'],
                    y=curve['Option Value'],
                    mode='lines',
                    name=f"S0 = ${S0}",
                    text=[S0]*len(curve),
                    hovertemplate=(
                        'S0: $%{text}<br>'
                        'Days to Expiry: %{x:.2f}<br>'
                        'Option Value: %{y:.4f}<extra></extra>'
                    ),
                ))
            fig.update_layout(
                title=f"{model_name} Option Value vs. Days to Expiry",
                xaxis_title="Days to Expiry",
                yaxis_title="Option Value",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.write(f"Sample of computed results for {model_name}:")
            st.dataframe(df.head(20))

        # Download results
        for model_name, df in results.items():
            st.download_button(
                label=f"Download {model_name} Results as CSV",
                data=df.to_csv(index=False),
                file_name=f"{model_name.lower()}_results.csv"
            )

    except Exception as e:
        st.error(f"Error: {e}")


        #cd "c:\Users\lcozo\OneDrive\Desktop\coding stuff\Options model"
        #streamlit run options_model_2_ui.py