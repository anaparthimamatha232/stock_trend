"""Stock Trend Classifier - Streamlit app.   Run:  streamlit run app.py"""
import json, os
import joblib
import pandas as pd
import streamlit as st
from features import (HORIZONS, PAST_PERIODS, build_features, fetch_stock, past_move)

st.set_page_config(page_title="Stock Trend Classifier", page_icon="📈", layout="wide")

META_PATH = "models/metadata.json"
if not os.path.exists(META_PATH):
    st.error("Models not found. Run the notebook first (it creates the models/ folder).")
    st.stop()
META = json.load(open(META_PATH))


@st.cache_resource
def load_model(h):
    return joblib.load(f"models/model_{h}.joblib")


@st.cache_data(ttl=3600, show_spinner=False)
def get_prices(ticker):
    """Live data from Yahoo Finance; falls back to the saved CSV if offline."""
    try:
        return fetch_stock(ticker + ".NS", start="2005-01-01"), "Yahoo Finance (live)"
    except Exception:
        raw = pd.read_csv("data/stock_data.csv", parse_dates=["Date"])
        df = raw[raw.Ticker == ticker].drop(columns="Ticker").sort_values("Date")
        return df.reset_index(drop=True), "saved CSV file (offline)"


st.title("📈 Stock Trend Classifier")
st.caption("Pick a stock and a period. See what **really happened** (calculated from data) "
           "and what the **ML model predicts next** (UP / DOWN).")

with st.sidebar:
    st.header("Choose")
    ticker = st.selectbox("Stock (NSE)", META["tickers"])
    past_label = st.selectbox("Look back period", list(PAST_PERIODS.keys()), index=2)
    fut_label = st.radio("Predict next…", list(HORIZONS.keys()),
                         format_func=lambda k: {"1D": "1 Day", "1W": "1 Week", "1M": "1 Month"}[k])
    st.caption(f"Model data ends: {META['data_last_date']}")

with st.spinner("Loading prices…"):
    df, source = get_prices(ticker)
if df is None or len(df) < 300:
    st.error("Not enough price data for this stock.")
    st.stop()
st.caption(f"Data source: {source} · latest date {df.Date.iloc[-1]:%d %b %Y}")

# ---------------- Part 1: the past (exact calculation) ----------------
n = PAST_PERIODS[past_label]
mv = past_move(df, n)
st.subheader(f"1) Past: {ticker} over the last {past_label}")
if mv is None:
    st.warning("Not enough history for this period.")
else:
    up = mv["direction"] == "UP"
    c1, c2, c3 = st.columns(3)
    c1.metric("Result", f"{'🟢 ROSE' if up else '🔴 FELL'}", f"{mv['pct']:+.2f}%")
    c2.metric(f"Price on {mv['start_date']:%d %b %Y}", f"₹{mv['start_price']:,.2f}")
    c3.metric(f"Price on {mv['end_date']:%d %b %Y}", f"₹{mv['end_price']:,.2f}")
    seg = df.iloc[-1 - n:]
    st.line_chart(seg.set_index("Date")["Close"], color="#2e9e44" if up else "#d6336c", height=320)
    st.caption("This part is an exact calculation from the closing prices, not a prediction.")

# ---------------- Part 2: the future (ML prediction) ----------------
st.subheader(f"2) Future: ML prediction for the next {fut_label}")
feat = build_features(df).dropna(subset=META["feature_columns"])
latest = feat.iloc[[-1]]
model = load_model(fut_label)
p_up = float(model.predict_proba(latest[META["feature_columns"]])[0, 1])
pred_up = p_up > 0.5
conf = p_up if pred_up else 1 - p_up
m = META["metrics"][fut_label]

c1, c2, c3 = st.columns(3)
c1.metric("Predicted direction", "🟢 UP" if pred_up else "🔴 DOWN")
c2.metric("Model confidence", f"{conf*100:.1f}%")
c3.metric("Model test AUC", f"{m['AUC']:.3f}", help="0.50 = coin flip. Measured on unseen test dates.")
st.progress(p_up, text=f"Probability UP: {p_up*100:.1f}%  |  DOWN: {(1-p_up)*100:.1f}%")

if conf < 0.55:
    st.warning("Low confidence (below 55%): the model is basically unsure for this stock right now.")
st.info(f"Model: **{m['Model']}** · test accuracy {m['Accuracy']*100:.1f}% vs "
        f"{m['Baseline']*100:.1f}% for always guessing the most common class.")

with st.expander("Latest technical indicators used by the model"):
    show = ["rsi14", "macd_hist", "sma20_ratio", "sma50_ratio", "vol_21", "volume_ratio", "dist_52w_high"]
    st.dataframe(latest[show].T.rename(columns={latest.index[0]: "value"}).round(4))

st.error("⚠️ Educational project only. Stock direction is very hard to predict; this is NOT financial advice.")
