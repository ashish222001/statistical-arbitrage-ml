# stat_arb_ml.py
import warnings
import os
warnings.filterwarnings("ignore")

import time
from itertools import combinations
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import zscore
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import joblib

# ---------------------------
# Parameters (edit as needed)
# ---------------------------
TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
START = "2018-01-01"
END   = "2024-12-31"
TRAIN_RATIO = 0.7
TRANSACTION_COST = 0.0005  # 5 bps per side
ZSCORE_ENTRY = 1.5
ZSCORE_EXIT = 0.5
RANDOM_STATE = 42

# ---------------------------
# Utilities & metrics
# ---------------------------
def sharpe_ratio(returns, period_per_year=252):
    r = returns.dropna()
    if r.std() == 0 or len(r) == 0:
        return 0.0
    return (r.mean() * period_per_year) / (r.std() * np.sqrt(period_per_year))

def max_drawdown(cum_series):
    peak = cum_series.cummax()
    drawdown = (cum_series - peak) / peak
    return drawdown.min()

# ---------------------------
# Robust download function
# ---------------------------
def download_prices(tickers, start, end):
    data = pd.DataFrame()
    for t in tickers:
        print(f"Downloading {t} ...")
        df = yf.download(t, start=start, end=end, progress=False)
        if df.empty:
            raise RuntimeError(f"Failed to download data for {t}")
        if 'Adj Close' in df.columns:
            data[t] = df['Adj Close']
        elif 'Close' in df.columns:
            data[t] = df['Close']
        else:
            raise RuntimeError(f"No usable price found for {t}")
    data = data.ffill().dropna()
    print("✅ Final merged shape:", data.shape)
    return data


# ---------------------------
# Pair selection (OLS residual + ADF)
# ---------------------------
def hedge_ratio_and_spread(x, y):
    x_const = sm.add_constant(x)
    model = sm.OLS(y, x_const).fit()
    beta = model.params[1]
    spread = x - beta * y
    return beta, spread

def adf_pvalue(series):
    # returns ADF p-value (use small lag)
    try:
        res = adfuller(series.dropna(), maxlag=1, autolag=None)
        return res[1]
    except Exception:
        return 1.0

def select_pairs(price_df, top_n=3):
    tickers = list(price_df.columns)
    pairs = list(combinations(tickers, 2))
    stats = []
    for a, b in pairs:
        y = price_df[a].dropna()
        x = price_df[b].dropna()
        # align
        df = pd.concat([y, x], axis=1).dropna()
        if len(df) < 120:
            continue
        beta, spread = hedge_ratio_and_spread(df[a], df[b])
        pval = adf_pvalue(spread)
        stats.append({"pair": (a, b), "beta": beta, "adf_p": pval})
    if not stats:
        return []
    df_stats = pd.DataFrame(stats).sort_values("adf_p")
    return df_stats.head(top_n)

# ---------------------------
# Build dataset for a pair
# ---------------------------
def build_pair_dataset(price_df, ticker_y, ticker_x):
    y = price_df[ticker_y]
    x = price_df[ticker_x]
    beta, spread = hedge_ratio_and_spread(y, x)
    df = pd.DataFrame({"y": y, "x": x, "spread": spread})
    # features
    df["spread_z"] = zscore(df["spread"].fillna(0))
    df["spread_lag1"] = df["spread"].shift(1)
    df["spread_ret"] = df["spread"].pct_change()
    df["ma5"] = df["spread"].rolling(5).mean()
    df["ma20"] = df["spread"].rolling(20).mean()
    df["std20"] = df["spread"].rolling(20).std()
    df = df.dropna()
    # target: binary next-day increase of spread
    df["next_spread_delta"] = df["spread"].shift(-1) - df["spread"]
    df["target"] = (df["next_spread_delta"] > 0).astype(int)
    return df, beta

# ---------------------------
# Train models
# ---------------------------
def train_models(df, feature_cols, train_ratio=TRAIN_RATIO):
    df = df.copy()
    split = int(len(df) * train_ratio)
    X_train = df[feature_cols].iloc[:split].values
    y_train = df["target"].iloc[:split].values
    X_test = df[feature_cols].iloc[split:].values
    y_test = df["target"].iloc[split:].values

    rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    xg = xgb.XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss', max_depth=4, random_state=RANDOM_STATE)
    xg.fit(X_train, y_train)
    xg_pred = xg.predict(X_test)

    print("\n--- Model performance (test set) ---")
    print("RandomForest accuracy:", accuracy_score(y_test, rf_pred))
    print(classification_report(y_test, rf_pred, zero_division=0))
    print("XGBoost accuracy:", accuracy_score(y_test, xg_pred))
    print(classification_report(y_test, xg_pred, zero_division=0))

    return rf, xg, (X_train, X_test, y_train, y_test)

# ---------------------------
# Signals & Backtest
# ---------------------------
def generate_signals(df, model, feature_cols, prob_threshold=0.5):
    d = df.copy()
    proba = model.predict_proba(d[feature_cols])[:, 1]  # prob spread increases
    d["pred_prob"] = proba
    d["pred"] = (d["pred_prob"] > prob_threshold).astype(int)
    # map prediction to spread position:
    # If model predicts spread increases => short spread (position -1), else long spread (+1)
    d["position"] = 0
    # If model predicts spread increases => long spread
    d.loc[d["pred"] == 1, "position"] = 1
    d.loc[d["pred"] == 0, "position"] = -1

    # shift to simulate entering next day
    d["position"] = d["position"].shift(1).fillna(0)
    return d

def backtest_pair(df_signals, beta, y_ticker, x_ticker, price_df, transaction_cost=TRANSACTION_COST):
    df = df_signals.copy()
    # pct returns
    py = price_df[y_ticker].pct_change().reindex(df.index)
    px = price_df[x_ticker].pct_change().reindex(df.index)
    df["ret_y"] = py
    df["ret_x"] = px
    # long spread => long y short beta*x => pnl = ret_y - beta*ret_x
    df["strategy_ret"] = df["position"] * (df["ret_y"] - beta * df["ret_x"])
    df["pos_change"] = df["position"].diff().abs().fillna(0)
    df["strategy_ret"] = df["strategy_ret"] - df["pos_change"] * transaction_cost
    df["cum_return"] = (1 + df["strategy_ret"].fillna(0)).cumprod() - 1
    return df

def transaction_cost_sensitivity(df_pair, beta, y_ticker, x_ticker, prices, model, feature_cols):
    costs = [0.0001, 0.0005, 0.001, 0.002]
    results = []

    for cost in costs:
        df_sign = generate_signals(df_pair, model, feature_cols)
        df_bt = backtest_pair(df_sign, beta, y_ticker, x_ticker, prices, transaction_cost=cost)
        sharpe = sharpe_ratio(df_bt['strategy_ret'])
        final_return = df_bt['cum_return'].iloc[-1]
        mdd = max_drawdown((1 + df_bt['strategy_ret']).cumprod())
        results.append({"Cost": cost, "Sharpe": sharpe, "Final Return": final_return, "Max Drawdown": mdd})

    results_df = pd.DataFrame(results)
    print("\n--- Transaction Cost Sensitivity ---")
    print(results_df)

    # Plot Sharpe vs Cost
    plt.figure(figsize=(8, 5))
    plt.plot(results_df["Cost"], results_df["Sharpe"], marker='o', label="Sharpe Ratio")
    plt.title(f"Sharpe Ratio Sensitivity for {y_ticker}/{x_ticker}")
    plt.xlabel("Transaction Cost per side")
    plt.ylabel("Sharpe Ratio")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return results_df


# ---------------------------
# Orchestration / main
# ---------------------------
def main():
    print("=== Step 1: Downloading prices ===")
    prices = download_prices(TICKERS, START, END)
    print("Prices shape:", prices.shape)
    print(prices.tail())

    print("\n=== Step 2: Selecting cointegrated pairs (ADF on OLS residual) ===")
    selected = select_pairs(prices, top_n=3)
    if selected is None or selected.empty:
        print("No candidate pairs found. Exiting.")
        return
    print(selected)

    # Work on the top pair
    top = selected.iloc[0]["pair"]
    y_ticker, x_ticker = top
    print(f"\n=== Step 3: Building dataset for top pair: {y_ticker} / {x_ticker} ===")
    df_pair, beta = build_pair_dataset(prices, y_ticker, x_ticker)
    # Save latest pair data for dashboard live monitor (CSV with Date column)
    df_pair_reset = df_pair.reset_index()
    df_pair_reset.to_csv("pair_data_latest.csv", index=False)
    print("Saved latest pair data to pair_data_latest.csv")

    df_pair_reset = df_pair.reset_index()
    df_pair_reset.to_csv("pair_data_latest.csv", index=False)
    print(df_pair.head())
    print("Dataset shape:", df_pair.shape)

    # Features & training
    feature_cols = ["spread_lag1", "spread_ret", "ma5", "ma20", "std20"]
    print("\n=== Step 4: Training models ===")
    rf_model, xg_model, _ = train_models(df_pair, feature_cols)

    # Save models (optional)
    joblib.dump(rf_model, "rf_model_toppair.joblib")
    joblib.dump(xg_model, "xg_model_toppair.joblib")
    print("Models saved to rf_model_toppair.joblib and xg_model_toppair.joblib")

    # Generate signals & backtest for both models
    print("\n=== Step 5: Generating signals & backtesting ===")
    df_rf_sign = generate_signals(df_pair, rf_model, feature_cols)
    df_xg_sign = generate_signals(df_pair, xg_model, feature_cols)

    df_rf_bt = backtest_pair(df_rf_sign, beta, y_ticker, x_ticker, prices)
    df_xg_bt = backtest_pair(df_xg_sign, beta, y_ticker, x_ticker, prices)
    # Save backtest data for dashboard visualizations
    df_rf_bt_reset = df_rf_bt.reset_index()
    df_xg_bt_reset = df_xg_bt.reset_index()
    df_rf_bt_reset.to_csv("rf_backtest.csv", index=False)
    df_xg_bt_reset.to_csv("xg_backtest.csv", index=False)

    # ---- Portfolio Mode: Evaluate all top 3 pairs ----
    portfolio_returns = []
    for i, row in selected.iterrows():
      y_ticker, x_ticker = row["pair"]
      print(f"\n--- Evaluating Pair {y_ticker}/{x_ticker} ---")
      df_tmp, beta_tmp = build_pair_dataset(prices, y_ticker, x_ticker)
      df_tmp_sign = generate_signals(df_tmp, rf_model, feature_cols)
      df_tmp_bt = backtest_pair(df_tmp_sign, beta_tmp, y_ticker, x_ticker, prices)
      portfolio_returns.append(df_tmp_bt["strategy_ret"])

    portfolio_df = pd.concat(portfolio_returns, axis=1).mean(axis=1)
    portfolio_cum = (1 + portfolio_df.fillna(0)).cumprod() - 1
    portfolio_cum.to_csv("portfolio_equity.csv", index=False)
    print("✅ Saved portfolio equity curve to portfolio_equity.csv")


    # ---- Performance Summary ----
    print("\n--- Performance Summary ---")
    results = pd.DataFrame([
        {
            "Model": "RandomForest",
            "Sharpe": round(sharpe_ratio(df_rf_bt['strategy_ret']), 3),
            "Max Drawdown": round(max_drawdown((1 + df_rf_bt['strategy_ret']).cumprod()), 3),
            "Final Return": round(df_rf_bt['cum_return'].iloc[-1], 3)
        },
        {
            "Model": "XGBoost",
            "Sharpe": round(sharpe_ratio(df_xg_bt['strategy_ret']), 3),
            "Max Drawdown": round(max_drawdown((1 + df_xg_bt['strategy_ret']).cumprod()), 3),
            "Final Return": round(df_xg_bt['cum_return'].iloc[-1], 3)
        }
    ])
    print(results)

    # ---- Plot both equity curves ----
    plt.figure(figsize=(10, 5))
    plt.plot(df_rf_bt.index, df_rf_bt["cum_return"], label="RandomForest Strategy")
    plt.plot(df_xg_bt.index, df_xg_bt["cum_return"], label="XGBoost Strategy")
    plt.title(f"Equity Curves for {y_ticker}/{x_ticker}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---- Transaction Cost Sensitivity (RandomForest as example) ----
    sensitivity_df = transaction_cost_sensitivity(
    df_pair, beta, y_ticker, x_ticker, prices, rf_model, feature_cols
)
    sensitivity_df.to_csv("transaction_cost_sensitivity.csv", index=False)
    print("✅ Saved transaction cost analysis to transaction_cost_sensitivity.csv")

# ---- Save results ----
    results.to_csv("model_comparison_summary.csv", index=False)
    print("\n✅ Results saved to model_comparison_summary.csv")
    print("Saving files in folder:", os.getcwd())
    print("\n--- Summary ---")
    print(f"Pair: {y_ticker}/{x_ticker}, Beta: {beta:.4f}")
    print(f"Train/test split ratio: {TRAIN_RATIO}")
    print(f"Transaction cost (per side): {TRANSACTION_COST}")

if __name__ == "__main__":
    main()


    




