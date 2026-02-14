import streamlit as st
import pandas as pd
import plotly.express as px
import time
import os
from datetime import datetime

st.set_page_config(page_title="Statistical Arbitrage ML Dashboard", layout="wide")
st.set_page_config(page_title="Statistical Arbitrage ML", layout="wide")

st.title("📈 Statistical Arbitrage ML Dashboard")
st.markdown("""
Welcome to your interactive ML-based trading analytics app.
Use the sidebar to:
- View backtest results
- Train new models
- Monitor live pair spreads
- Upload and analyze new datasets
""")
st.divider()

st.success("Navigate using the left sidebar to explore various modules.")


st.title("📈 Statistical Arbitrage ML Dashboard")

# =======================
# Load Data
# =======================
model_results = pd.read_csv("model_comparison_summary.csv")
cost_results = pd.read_csv("transaction_cost_sensitivity.csv")
pair_df = pd.read_csv("pair_data_latest.csv")
rf_bt = pd.read_csv("rf_backtest.csv")
xg_bt = pd.read_csv("xg_backtest.csv")
portfolio = pd.read_csv("portfolio_equity.csv")

# =======================
# Model Performance
# =======================
st.header("💡 Model Performance Comparison")
st.dataframe(model_results)

# =======================
# Transaction Cost Sensitivity
# =======================
st.header("💰 Transaction Cost Sensitivity")
st.dataframe(cost_results)

fig_cost = px.line(cost_results, x="Cost", y="Sharpe", title="Transaction Cost vs Sharpe Ratio", markers=True)
st.plotly_chart(fig_cost, use_container_width=True)

# =======================
# Live Pair Monitor
# =======================
st.header("📡 Live Pair Monitor")

latest_z = round(pair_df["spread_z"].iloc[-1], 2)
latest_spread = round(pair_df["spread"].iloc[-1], 2)
latest_date = pair_df["Date"].iloc[-1]

col1, col2, col3 = st.columns(3)
col1.metric("Latest Date", latest_date)
col2.metric("Latest Z-Score", latest_z)
col3.metric("Latest Spread", latest_spread)

st.info("🔄 Auto-refresh every 60 seconds")
time.sleep(60)
st.rerun()

# =======================
# Live Pair Monitor (safe refresh)
# =======================
import os
from datetime import datetime

st.header("📡 Live Pair Monitor")

pair_file = "pair_data_latest.csv"
if not os.path.exists(pair_file):
    st.warning("pair_data_latest.csv not found. Run stat_arb_ml.py to generate latest pair data.")
else:
    try:
        pair_df = pd.read_csv(pair_file)
        # ensure index/Date present
        if "Date" in pair_df.columns:
            last_date = pair_df["Date"].iloc[-1]
        else:
            # if Date is the index name, try fallback
            last_date = pair_df.iloc[-1, 0]

        latest_z = round(pair_df["spread_z"].iloc[-1], 3)
        latest_spread = round(pair_df["spread"].iloc[-1], 4)
        file_mtime = datetime.fromtimestamp(os.path.getmtime(pair_file)).strftime("%Y-%m-%d %H:%M:%S")

        col1, col2, col3 = st.columns([2, 2, 2])
        col1.metric("Latest Date (pair)", str(last_date))
        col2.metric("Latest Z-Score", latest_z)
        col3.metric("Latest Spread", latest_spread)

        st.write(f"📁 Data last written: **{file_mtime}**")

        # Refresh button — re-run the script so latest CSV is reloaded
        if st.button("Refresh Live Monitor"):
            st.rerun()  # safe in current Streamlit version; if error replace with st.rerun()

    except Exception as e:
        st.error(f"Error reading pair_data_latest.csv: {e}")



# =======================
# Interactive Charts
# =======================
st.header("📊 Interactive Equity Curves")

fig1 = px.line(rf_bt, x="Date", y="cum_return", title="Equity Curve - RandomForest")
fig2 = px.line(xg_bt, x="Date", y="cum_return", title="Equity Curve - XGBoost")

st.plotly_chart(fig1, use_container_width=True)
st.plotly_chart(fig2, use_container_width=True)

# =======================
# Portfolio Average Curve
# =======================
st.header("💼 Portfolio (Top 3 Pairs) Performance")
fig_portfolio = px.line(portfolio, y="portfolio_return", title="Average Portfolio Return (Top 3 Pairs)", color_discrete_sequence=['orange'])
st.plotly_chart(fig_portfolio, use_container_width=True)
