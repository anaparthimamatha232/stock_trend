"""
features.py  -  shared code used by the notebook AND the Streamlit app.
Keeping feature engineering in ONE place means training and prediction
always use exactly the same formulas (no mismatch bugs).
"""
import numpy as np
import pandas as pd

# Prediction horizons: name -> number of trading days ahead
HORIZONS = {"1D": 1, "1W": 5, "1M": 21}

# Periods for the "what happened in the past" view: label -> trading days
PAST_PERIODS = {
    "1 Day": 1, "1 Week (5 days)": 5, "1 Month (21 days)": 21,
    "3 Months": 63, "6 Months": 126, "1 Year": 252,
    "3 Years": 756, "5 Years": 1260,
}

FEATURE_COLUMNS = [
    "ret_1", "ret_5", "ret_10", "ret_21", "ret_63",
    "sma5_ratio", "sma10_ratio", "sma20_ratio", "sma50_ratio", "sma200_ratio",
    "ema12_ratio", "ema26_ratio",
    "macd", "macd_signal", "macd_hist",
    "rsi14", "stoch_k", "stoch_d",
    "bb_pctb", "bb_width", "atr14_pct",
    "vol_5", "vol_21", "vol_ratio_5_21",
    "volume_ratio", "obv_slope",
    "dist_52w_high", "dist_52w_low",
    "gap", "hl_range", "dow", "month",
]


# ----------------------------------------------------------------- download
def fetch_stock(ticker, start=None, end=None, period=None, retries=3):
    """Download daily OHLCV for one ticker from Yahoo Finance."""
    import time
    import yfinance as yf
    last_err = None
    for _ in range(retries):
        try:
            kw = dict(progress=False, auto_adjust=True)
            if period:
                df = yf.download(ticker, period=period, **kw)
            else:
                df = yf.download(ticker, start=start, end=end, **kw)
            if df is not None and len(df) > 0:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
                df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
                return df.dropna().sort_values("Date").reset_index(drop=True)
        except Exception as e:  # network hiccup -> retry
            last_err = e
        time.sleep(2)
    raise RuntimeError(f"Could not download {ticker}: {last_err}")


# ----------------------------------------------------------------- features
def _rsi(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_features(df):
    """df: one stock, columns Date/Open/High/Low/Close/Volume. Returns df + features."""
    df = df.sort_values("Date").reset_index(drop=True).copy()
    c, h, l, o, v = df["Close"], df["High"], df["Low"], df["Open"], df["Volume"].astype(float)
    out = df.copy()

    for n in (1, 5, 10, 21, 63):
        out[f"ret_{n}"] = c.pct_change(n)
    for n in (5, 10, 20, 50, 200):
        out[f"sma{n}_ratio"] = c / c.rolling(n).mean() - 1
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    out["ema12_ratio"], out["ema26_ratio"] = c / ema12 - 1, c / ema26 - 1
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    out["macd"], out["macd_signal"], out["macd_hist"] = macd / c, sig / c, (macd - sig) / c

    out["rsi14"] = _rsi(c)
    ll, hh = l.rolling(14).min(), h.rolling(14).max()
    out["stoch_k"] = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    out["stoch_d"] = out["stoch_k"].rolling(3).mean()

    m20, s20 = c.rolling(20).mean(), c.rolling(20).std()
    out["bb_pctb"] = (c - (m20 - 2 * s20)) / (4 * s20).replace(0, np.nan)
    out["bb_width"] = 4 * s20 / m20
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    out["atr14_pct"] = tr.rolling(14).mean() / c

    r1 = c.pct_change()
    out["vol_5"], out["vol_21"] = r1.rolling(5).std(), r1.rolling(21).std()
    out["vol_ratio_5_21"] = out["vol_5"] / out["vol_21"].replace(0, np.nan)
    out["volume_ratio"] = v / v.rolling(20).mean().replace(0, np.nan)
    obv = (np.sign(r1.fillna(0)) * v).cumsum()
    out["obv_slope"] = obv.diff(10) / v.rolling(20).sum().replace(0, np.nan)

    out["dist_52w_high"] = c / c.rolling(252).max() - 1
    out["dist_52w_low"] = c / c.rolling(252).min() - 1
    out["gap"] = o / c.shift() - 1
    out["hl_range"] = (h - l) / c
    out["dow"] = out["Date"].dt.dayofweek
    out["month"] = out["Date"].dt.month
    return out.replace([np.inf, -np.inf], np.nan)


def add_targets(feat):
    """Add target_1D/1W/1M: 1 if Close after h trading days > Close today."""
    feat = feat.copy()
    for name, h in HORIZONS.items():
        fut = feat["Close"].shift(-h)
        feat[f"target_{name}"] = np.where(fut.isna(), np.nan, (fut > feat["Close"]).astype(float))
    return feat


def build_dataset(raw):
    """raw: all stocks stacked, with a 'Ticker' column. Returns the model-ready table."""
    parts = []
    for tkr, g in raw.groupby("Ticker"):
        f = add_targets(build_features(g.drop(columns="Ticker")))
        f.insert(1, "Ticker", tkr)
        parts.append(f)
    data = pd.concat(parts, ignore_index=True)
    return data.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)


# ----------------------------------------------------------------- splitting
def time_split(data, target, h, train_frac=0.70, val_frac=0.15):
    """
    Split by DATE (never random). A gap of h days is removed before each boundary
    so that a training label (which looks h days ahead) never overlaps val/test.
    """
    d = data.dropna(subset=[target])
    dates = np.sort(d["Date"].unique())
    t_end = dates[int(len(dates) * train_frac)]
    v_end = dates[int(len(dates) * (train_frac + val_frac))]
    gap = pd.Timedelta(days=int(h * 1.6) + 2)  # calendar-day cushion for h trading days
    train = d[d["Date"] < t_end - gap]
    val = d[(d["Date"] >= t_end) & (d["Date"] < v_end - gap)]
    test = d[d["Date"] >= v_end]
    return train, val, test


# ----------------------------------------------------------------- past view
def past_move(df, n):
    """Exact calculation: did the stock go UP or DOWN over the last n trading days?"""
    if len(df) <= n:
        return None
    s, e = df.iloc[-1 - n], df.iloc[-1]
    pct = (e["Close"] / s["Close"] - 1) * 100
    return dict(start_date=s["Date"], end_date=e["Date"], start_price=float(s["Close"]),
                end_price=float(e["Close"]), pct=float(pct), direction="UP" if pct > 0 else "DOWN")
