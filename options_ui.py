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

from Options_model import get_live_quote, price_american_option, compute_curve_for_S0

st.title("American Option Risk Analysis")

# User inputs with unique keys and generic defaults
ticker = st.text_input("Ticker", value="AAPL", key="ticker_input")
K = st.number_input("Strike Price (K)", value=100.0, key="strike_input")
r = st.number_input("Risk-Free Rate (r)", value=0.03, key="rate_input")
option_type = st.selectbox("Option Type", ["call", "put"], key="type_input")
sigma_override = st.number_input("Volatility (σ, leave 0 for auto)", value=0.0, key="sigma_input")
expiry = st.date_input("Expiry Date", value=datetime.date.today() + datetime.timedelta(days=90), key="expiry_input")
num_simulations = st.number_input("Number of Simulations", value=10000, step=1000, key="sim_input")
intervals_per_day = st.number_input("Intervals Per Day", value=2, key="intervals_input")
s0_start = st.number_input("S0 Start", value=80, key="s0_start_input")
s0_end = st.number_input("S0 End", value=120, key="s0_end_input")
s0_step = st.number_input("S0 Step", value=5, key="s0_step_input")

use_parallel = st.checkbox("Use parallel processing (faster, but may cause issues on Windows)", value=False)

st.write("Inputs:", ticker, K, r, option_type, sigma_override, expiry, num_simulations, intervals_per_day, s0_start, s0_end, s0_step)

if st.button("Run Analysis"):
    S0_live, sigma_live = get_live_quote(ticker)
    sigma = sigma_override if sigma_override > 0 else sigma_live
    today = datetime.date.today()
    days_to_expiry = (expiry - today).days
    total_points = days_to_expiry * int(intervals_per_day)
    s0_list = list(range(int(s0_start), int(s0_end) + 1, int(s0_step)))
    if int(S0_live) not in s0_list:
        s0_list.append(int(S0_live))
    s0_list = sorted(set(s0_list))

    MAX_WORKERS = 6

    args = [
        (S0, K, r, sigma, num_simulations, int(intervals_per_day), total_points,
         option_type, 2, False, 2025)
        for S0 in s0_list
    ]

    records = []
    if use_parallel:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(compute_curve_for_S0, *arg) for arg in args]
            progress_bar = st.progress(0)
            total = len(futures)
            for i, future in enumerate(as_completed(futures)):
                records.extend(future.result())
                progress_bar.progress((i + 1) / total)
            progress_bar.empty()
    else:
        progress_bar = st.progress(0)
        total = len(args)
        for i, arg in enumerate(args):
            records.extend(compute_curve_for_S0(*arg))
            progress_bar.progress((i + 1) / total)
        progress_bar.empty()

    df = pd.DataFrame(records)
    fig = go.Figure()
    for S0 in s0_list:
        curve = df[df['S0'] == S0]
        fig.add_trace(go.Scatter(
            x=curve['Days to Expiry'],
            y=curve['Option Value'],
            mode='lines',
            name=f"S0 = ${S0}",
            customdata=curve[['Std Dev', 'Zero Prob']],
            hovertemplate=(
                'S0: $%{text}<br>'
                'Days to Expiry: %{x:.2f}<br>'
                'Option Value: %{y:.4f}<br>'
                'Std Dev: %{customdata[0]:.4f}<br>'
                'P(Worthless): %{customdata[1]:.2%}<extra></extra>'
            ),
            text=[S0]*len(curve)
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([curve['Days to Expiry'], curve['Days to Expiry'][::-1]]),
            y=pd.concat([curve['Option Value'] + curve['Std Dev'], (curve['Option Value'] - curve['Std Dev'])[::-1]]),
            fill='toself',
            fillcolor='rgba(0,100,80,0.15)',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
    st.plotly_chart(fig, use_container_width=True)
    st.write("Sample of computed results:")
    st.dataframe(df.head(20))