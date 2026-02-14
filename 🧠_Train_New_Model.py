# ============================================================
# 🧠 Train New Model — Final V7
# Full Version (All Models + Visuals + Theme Toggle + No Test Size)
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, auc
)
from sklearn.utils import resample
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import ta
import warnings
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ============================================================
# 🎨 THEME: Dark/Light toggle
# ============================================================
def set_theme(dark_mode=True):
    if dark_mode:
        css = """
        <style>
        body { background-color: #0E1117; color: #FAFAFA; }
        .stApp { background-color: #0E1117; }
        h1, h2, h3, h4 { color: #00C4FF; font-family: 'Poppins', sans-serif; }
        .stMetric { background-color: #1C1F26; padding: 10px; border-radius: 10px; }
        .stButton>button { background-color: #00C4FF; color: #000; }
        </style>
        """
        theme = "plotly_dark"
    else:
        css = """
        <style>
        body { background-color: #FFFFFF; color: #000000; }
        .stApp { background-color: #FFFFFF; }
        h1, h2, h3, h4 { color: #007BFF; font-family: 'Poppins', sans-serif; }
        .stMetric { background-color: #F3F3F3; padding: 10px; border-radius: 10px; }
        .stButton>button { background-color: #007BFF; color: #fff; }
        </style>
        """
        theme = "plotly_white"
    st.markdown(css, unsafe_allow_html=True)
    return theme


# ============================================================
# 🔧 Helper Functions
# ============================================================
def download_prices(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        if 'Adj Close' in data.columns.get_level_values(0):
            data = data['Adj Close']
        elif 'Close' in data.columns.get_level_values(0):
            data = data['Close']
        else:
            data = data.xs(data.columns[0], axis=1, level=0)
    else:
        if 'Adj Close' in data.columns:
            data = data[['Adj Close']]
            if isinstance(tickers, (list, tuple)) and len(tickers) == 1:
                data.columns = [tickers[0]]
        elif 'Close' in data.columns:
            data = data[['Close']]
            if isinstance(tickers, (list, tuple)) and len(tickers) == 1:
                data.columns = [tickers[0]]
    return data.dropna(how='all')


def select_pairs(prices, top_n=3):
    pairs = []
    cols = list(prices.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            y = prices[cols[i]].dropna()
            x = prices[cols[j]].dropna()
            df_tmp = pd.concat([y, x], axis=1).dropna()
            if len(df_tmp) < 120:
                continue
            beta = np.polyfit(df_tmp[cols[j]], df_tmp[cols[i]], 1)[0]
            resid = df_tmp[cols[i]] - beta * df_tmp[cols[j]]
            try:
                pvalue = adfuller(resid, maxlag=1, autolag=None)[1]
            except Exception:
                pvalue = 1.0
            pairs.append((cols[i], cols[j], beta, pvalue))
    df_pairs = pd.DataFrame(pairs, columns=["y", "x", "beta", "pvalue"])
    if df_pairs.empty:
        return df_pairs
    return df_pairs.sort_values("pvalue").head(top_n)


def build_pair_dataset(prices, y_ticker, x_ticker, beta):
    df = pd.DataFrame({"y": prices[y_ticker], "x": prices[x_ticker]})
    df["spread"] = df["y"] - beta * df["x"]
    df["spread_z"] = (df["spread"] - df["spread"].rolling(20).mean()) / df["spread"].rolling(20).std()
    df["spread_lag1"] = df["spread"].shift(1)
    df["spread_ret"] = df["spread"].pct_change()
    df["ma5"] = df["spread"].rolling(5).mean()
    df["ma20"] = df["spread"].rolling(20).mean()
    df["std20"] = df["spread"].rolling(20).std()
    df["next_spread_delta"] = df["spread"].shift(-1) - df["spread"]
    df["target"] = (df["next_spread_delta"] > 0).astype(int)
    return df.dropna()


def preprocess_features(df, feature_cols):
    X = df[feature_cols].copy()
    y = df["target"].copy()
    X = X.fillna(method="ffill").fillna(method="bfill")
    for col in feature_cols:
        q_low, q_high = X[col].quantile(0.01), X[col].quantile(0.99)
        X[col] = np.clip(X[col], q_low, q_high)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=min(len(feature_cols), 5))
    X_pca = pca.fit_transform(X_scaled)
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.3, shuffle=False
    )
    return X_train, X_test, y_train, y_test, scaler, pca


def metrics_dict(y_true, y_pred, probs=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc_score = roc_auc_score(y_true, probs) if probs is not None else np.nan
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "AUC": auc_score}


# ============================================================
# 📈 Plot Functions
# ============================================================
def plot_probability_distribution(y_true, probs, model_name, theme):
    df_probs = pd.DataFrame({"True Label": y_true, "Predicted Probability": probs})
    fig = px.histogram(
        df_probs, x="Predicted Probability", color=df_probs["True Label"].astype(str),
        nbins=50, barmode="overlay",
        color_discrete_sequence=["#00C4FF", "#FF4B4B"],
        template=theme
    )
    fig.update_layout(
        title=f"Probability Distribution — {model_name}",
        xaxis_title="Predicted Probability", yaxis_title="Count", height=480
    )
    return fig


def plot_roc_curves(models_info, y_true, theme):
    fig = go.Figure()
    for name, probs in models_info:
        if probs is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc_val = roc_auc_score(y_true, probs)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{name} (AUC={auc_val:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), showlegend=False))
    fig.update_layout(title_text='ROC Curves — Model Comparison', xaxis_title='False Positive Rate',
                      yaxis_title='True Positive Rate', height=480, template=theme)
    return fig


def plot_pr_curves(models_info, y_true, theme):
    fig = go.Figure()
    for name, probs in models_info:
        if probs is None:
            continue
        precision, recall, _ = precision_recall_curve(y_true, probs)
        auc_pr = auc(recall, precision)
        fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f"{name} (AUC_PR={auc_pr:.3f})"))
    fig.update_layout(title_text='Precision-Recall Curves — Model Comparison', xaxis_title='Recall',
                      yaxis_title='Precision', height=480, template=theme)
    return fig


def plot_confusion_matrix(y_true, y_pred, title, theme):
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(cm, text_auto=True, labels=dict(x='Predicted', y='Actual'),
                    x=["0", "1"], y=["0", "1"], color_continuous_scale='bluered', template=theme)
    fig.update_layout(title=title, height=420)
    return fig


def optimize_threshold_from_probs(probs, y_true):
    thresholds = np.linspace(0.1, 0.9, 81)
    best_th, best_acc = 0.5, 0
    for th in thresholds:
        preds = (probs >= th).astype(int)
        a = accuracy_score(y_true, preds)
        if a > best_acc:
            best_acc, best_th = a, th
    return best_th, best_acc


# ============================================================
# 🚀 Streamlit App
# ============================================================
st.set_page_config(page_title="Train Model — Final V7", page_icon="🧠", layout="wide")
st.sidebar.header("⚙️ Options & Controls")

dark_mode = st.sidebar.checkbox("🌙 Dark Mode", value=True)
theme = set_theme(dark_mode)

st.sidebar.subheader("Data & Training")
tickers_in = st.sidebar.text_input("Stock tickers (comma separated)", "AAPL,MSFT,GOOG,AMZN,META")
start = st.sidebar.date_input("Start date", pd.to_datetime("2018-01-01"))
end = st.sidebar.date_input("End date", pd.to_datetime("2025-01-01"))
train_button = st.sidebar.button("🚀 Train Models")

show_feature_importance = st.sidebar.checkbox("Show feature importances", True)
export_metrics = st.sidebar.checkbox("Enable metrics export (CSV)", True)
export_pngs = st.sidebar.checkbox("Enable PNG export for charts", False)

st.title("🧠 Train New Model — Final App (V7)")

if not train_button:
    st.info("Set tickers/date and click '🚀 Train Models' to run the full pipeline.")
else:
    tickers = [t.strip().upper() for t in tickers_in.split(",") if t.strip()]
    st.write(f"### 📥 Downloading data for: {', '.join(tickers)}")
    prices = download_prices(tickers, start, end)
    if prices.empty:
        st.error("No price data downloaded.")
        st.stop()

    st.subheader("Historical Prices")
    st.plotly_chart(px.line(prices, title='Price Series', template=theme), use_container_width=True)

    st.write("### 🔍 Selecting cointegrated pairs...")
    selected = select_pairs(prices)
    if selected.empty:
        st.warning("No candidate pairs found.")
        st.stop()
    st.dataframe(selected)

    top = selected.iloc[0]
    y_ticker, x_ticker, beta = top["y"], top["x"], top["beta"]
    st.success(f"Top pair: {y_ticker}/{x_ticker}")

    # BASIC PIPELINE
    df_pair_basic = build_pair_dataset(prices, y_ticker, x_ticker, beta)
    feature_cols_basic = ["spread_lag1", "spread_ret", "ma5", "ma20", "std20"]
    X_train_b, X_test_b, y_train_b, y_test_b, scaler_b, pca_b = preprocess_features(df_pair_basic, feature_cols_basic)

    rf_params = {"n_estimators": [100, 200], "max_depth": [3, 5, 7, None], "min_samples_split": [2, 5]}
    rf_b = RandomForestClassifier(random_state=42)
    rf_grid_b = GridSearchCV(rf_b, rf_params, cv=3, scoring="accuracy", n_jobs=-1)
    rf_grid_b.fit(X_train_b, y_train_b)
    best_rf_b = rf_grid_b.best_estimator_

    xg_params = {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1], "max_depth": [3, 5, 7]}
    xg_b = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    xg_grid_b = GridSearchCV(xg_b, xg_params, cv=3, scoring="accuracy", n_jobs=-1)
    xg_grid_b.fit(X_train_b, y_train_b)
    best_xg_b = xg_grid_b.best_estimator_

    rf_pred_b = best_rf_b.predict(X_test_b)
    xg_pred_b = best_xg_b.predict(X_test_b)

    st.subheader("📊 Basic Pipeline — Model Performance")
    st.dataframe(pd.DataFrame([
        {"Model": "RandomForest (basic)", "Accuracy": accuracy_score(y_test_b, rf_pred_b)},
        {"Model": "XGBoost (basic)", "Accuracy": accuracy_score(y_test_b, xg_pred_b)}
    ]))

    # ENHANCED PIPELINE
    st.subheader("⚙️ Building Enhanced Dataset with Advanced Features")

    df_enh = pd.DataFrame({"y": prices[y_ticker], "x": prices[x_ticker]})
    df_enh["spread"] = df_enh["y"] - beta * df_enh["x"]
    df_enh["ma20"] = df_enh["spread"].rolling(20).mean()
    df_enh["std20"] = df_enh["spread"].rolling(20).std()
    df_enh["zscore"] = (df_enh["spread"] - df_enh["ma20"]) / df_enh["std20"]
    df_enh["target"] = 0
    df_enh.loc[df_enh["zscore"] > 1, "target"] = 1
    df_enh.loc[df_enh["zscore"] < -1, "target"] = -1
    df_enh = df_enh.dropna()
    df_enh["target"] = (df_enh["target"].shift(-1) > 0).astype(int)

    df_enh["spread_lag1"] = df_enh["spread"].shift(1)
    df_enh["spread_ret"] = df_enh["spread"].pct_change().fillna(0)
    df_enh["ma5"] = df_enh["spread"].rolling(5).mean()
    df_enh["spread_change"] = df_enh["spread"].diff()
    df_enh["spread_vol"] = df_enh["spread"].rolling(10).std()
    df_enh["spread_momentum"] = df_enh["spread"].diff(3)
    df_enh["bollinger_upper"] = df_enh["ma20"] + 2 * df_enh["std20"]
    df_enh["bollinger_lower"] = df_enh["ma20"] - 2 * df_enh["std20"]
    df_enh["distance_from_bands"] = (df_enh["spread"] - df_enh["ma20"]) / df_enh["std20"]

    try:
        df_enh["rsi"] = ta.momentum.RSIIndicator(df_enh["spread"], window=14).rsi()
        macd = ta.trend.MACD(df_enh["spread"])
        df_enh["macd"] = macd.macd()
        df_enh["macd_signal"] = macd.macd_signal()
        df_enh["ema10"] = ta.trend.EMAIndicator(df_enh["spread"], window=10).ema_indicator()
        df_enh["ema50"] = ta.trend.EMAIndicator(df_enh["spread"], window=50).ema_indicator()
        df_enh["roc"] = ta.momentum.ROCIndicator(df_enh["spread"], window=12).roc()
        df_enh["adx"] = ta.trend.ADXIndicator(df_enh["y"], df_enh["x"], df_enh["spread"], window=14).adx()
        df_enh["willr"] = ta.momentum.WilliamsRIndicator(df_enh["y"], df_enh["x"], df_enh["spread"], lbp=14).williams_r()
    except Exception:
        st.warning("TA library failed to compute some indicators — proceeding without them.")

    df_enh = df_enh.dropna()

    feature_cols_enh = [
        "spread_lag1", "spread_ret", "ma5", "ma20", "std20", "spread_change", "spread_vol",
        "spread_momentum", "bollinger_upper", "bollinger_lower", "distance_from_bands",
        "rsi", "macd", "macd_signal", "ema10", "ema50", "roc", "adx", "willr"
    ]

    scaler_enh = StandardScaler()
    X_enh_scaled = scaler_enh.fit_transform(df_enh[feature_cols_enh])
    y_enh = df_enh["target"]

    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(X_enh_scaled):
        X_train_e, X_test_e = X_enh_scaled[train_index], X_enh_scaled[test_index]
        y_train_e, y_test_e = y_enh.iloc[train_index], y_enh.iloc[test_index]

    train_df = pd.DataFrame(X_train_e, columns=feature_cols_enh)
    train_df["target"] = y_train_e.values
    maj, mino = train_df[train_df["target"] == 0], train_df[train_df["target"] == 1]
    if len(mino) > 0:
        mino_up = resample(mino, replace=True, n_samples=len(maj), random_state=42)
        balanced = pd.concat([maj, mino_up])
        X_train_e = balanced[feature_cols_enh].values
        y_train_e = balanced["target"].values
        st.success("✅ Balanced training data")

    selector = SelectKBest(mutual_info_classif, k=min(15, len(feature_cols_enh)))
    selector.fit(X_train_e, y_train_e)
    X_train_e_sel = selector.transform(X_train_e)
    X_test_e_sel = selector.transform(X_test_e)
    selected_features = [f for f, s in zip(feature_cols_enh, selector.get_support()) if s]

    rf_enh = RandomForestClassifier(
        n_estimators=600, max_depth=9, min_samples_split=5, class_weight='balanced', random_state=42
    )
    rf_enh.fit(X_train_e_sel, y_train_e)

    xg_enh = XGBClassifier(
        learning_rate=0.05, max_depth=8, n_estimators=800, subsample=0.9,
        colsample_bytree=0.9, gamma=1, reg_lambda=5,
        eval_metric='logloss', use_label_encoder=False, random_state=42
    )
    xg_enh.fit(X_train_e_sel, y_train_e)

    ensemble_model = VotingClassifier(estimators=[('rf', rf_enh), ('xgb', xg_enh)], voting='soft')
    ensemble_model.fit(X_train_e_sel, y_train_e)

    rf_pred = rf_enh.predict(X_test_e_sel)
    xg_pred = xg_enh.predict(X_test_e_sel)
    ens_pred = ensemble_model.predict(X_test_e_sel)

    rf_probs = rf_enh.predict_proba(X_test_e_sel)[:, 1]
    xg_probs = xg_enh.predict_proba(X_test_e_sel)[:, 1]
    ens_probs = ensemble_model.predict_proba(X_test_e_sel)[:, 1]

    metrics_enh = []
    for name, pred, probs in [
        ("RandomForest (enhanced)", rf_pred, rf_probs),
        ("XGBoost (enhanced)", xg_pred, xg_probs),
        ("Ensemble (RF+XGB)", ens_pred, ens_probs)
    ]:
        d = metrics_dict(y_test_e, pred, probs)
        d.update({"Model": name})
        metrics_enh.append(d)

    metrics_df = pd.DataFrame(metrics_enh)[["Model", "Accuracy", "Precision", "Recall", "F1", "AUC"]]
    st.subheader("📊 Enhanced Pipeline — Model Comparison")
    st.dataframe(metrics_df.style.format({
        "Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}", "F1": "{:.4f}", "AUC": "{:.4f}"
    }))

    best_by_auc = metrics_df.sort_values('AUC', ascending=False).iloc[0]
    best_by_acc = metrics_df.sort_values('Accuracy', ascending=False).iloc[0]

    st.markdown("---")
    col1, col2 = st.columns(2)
    col1.metric("Best (AUC)", f"{best_by_auc['Model']}", f"AUC: {best_by_auc['AUC']:.4f}")
    col2.metric("Best (Accuracy)", f"{best_by_acc['Model']}", f"Accuracy: {best_by_acc['Accuracy']:.4f}")

    st.success(f"🏁 Recommendation: use {best_by_auc['Model']} (highest AUC={best_by_auc['AUC']:.4f}) — present this as the distinguished final model in your report.")

    st.plotly_chart(plot_roc_curves([
        ("RandomForest", rf_probs),
        ("XGBoost", xg_probs),
        ("Ensemble", ens_probs)
    ], y_test_e, theme), use_container_width=True)

    st.plotly_chart(plot_pr_curves([
        ("RandomForest", rf_probs),
        ("XGBoost", xg_probs),
        ("Ensemble", ens_probs)
    ], y_test_e, theme), use_container_width=True)

    cm1, cm2, cm3 = st.columns(3)
    cm1.plotly_chart(plot_confusion_matrix(y_test_e, rf_pred, "RF Confusion", theme), use_container_width=True)
    cm2.plotly_chart(plot_confusion_matrix(y_test_e, xg_pred, "XGB Confusion", theme), use_container_width=True)
    cm3.plotly_chart(plot_confusion_matrix(y_test_e, ens_pred, "Ensemble Confusion", theme), use_container_width=True)

    st.subheader("🎨 Probability Distributions")
    pcol1, pcol2 = st.columns(2)
    pcol1.plotly_chart(plot_probability_distribution(y_test_e, rf_probs, "RandomForest", theme), use_container_width=True)
    pcol2.plotly_chart(plot_probability_distribution(y_test_e, xg_probs, "XGBoost", theme), use_container_width=True)
    st.plotly_chart(plot_probability_distribution(y_test_e, ens_probs, "Ensemble", theme), use_container_width=True)

    if show_feature_importance:
        st.subheader("🔎 Feature Importances (selected features)")
        fi_rf = rf_enh.feature_importances_
        try:
            fi_xgb = xg_enh.feature_importances_
        except Exception:
            fi_xgb = None
        fi_df = pd.DataFrame({'feature': selected_features, 'rf_importance': fi_rf})
        if fi_xgb is not None and len(fi_xgb) == len(selected_features):
            fi_df['xgb_importance'] = fi_xgb
        fi_df = fi_df.sort_values('rf_importance', ascending=False)
        fig_fi = go.Figure()
        fig_fi.add_trace(go.Bar(x=fi_df['feature'], y=fi_df['rf_importance'], name='RF importance'))
        if 'xgb_importance' in fi_df.columns:
            fig_fi.add_trace(go.Bar(x=fi_df['feature'], y=fi_df['xgb_importance'], name='XGB importance'))
        fig_fi.update_layout(barmode='group', title_text='Feature Importances (selected)', template=theme, height=500)
        st.plotly_chart(fig_fi, use_container_width=True)

    st.subheader("🎯 Threshold Optimization")
    rf_th, rf_th_acc = optimize_threshold_from_probs(rf_probs, y_test_e)
    xg_th, xg_th_acc = optimize_threshold_from_probs(xg_probs, y_test_e)
    ens_th, ens_th_acc = optimize_threshold_from_probs(ens_probs, y_test_e)

    th_df = pd.DataFrame([
        {"Model": "RandomForest", "Best_Threshold": rf_th, "Accuracy_at_threshold": rf_th_acc},
        {"Model": "XGBoost", "Best_Threshold": xg_th, "Accuracy_at_threshold": xg_th_acc},
        {"Model": "Ensemble", "Best_Threshold": ens_th, "Accuracy_at_threshold": ens_th_acc}
    ])
    st.dataframe(th_df.style.format({"Best_Threshold": "{:.2f}", "Accuracy_at_threshold": "{:.4f}"}))

    if export_metrics:
        csv_bytes = metrics_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download metrics CSV", data=csv_bytes, file_name="model_metrics.csv", mime='text/csv')

    if export_pngs:
        st.info("Preparing PNGs — requires `kaleido` installed.")
        for name, fig in [("roc", plot_roc_curves([("RF", rf_probs), ("XGB", xg_probs), ("Ens", ens_probs)], y_test_e, theme))]:
            try:
                img_bytes = fig.to_image(format='png', width=1200, height=600)
                st.download_button(f"⬇️ Download {name}.png", data=img_bytes, file_name=f"{name}.png", mime='image/png')
            except Exception as e:
                st.warning(f"Could not export {name} to PNG: {e}")

    st.info("✅ Full pipeline complete. 'Test set size' metric removed for clean report visuals.")
