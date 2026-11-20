"""
download_data.py - builds data/stock_data.csv (many NSE stocks, daily, up to TODAY).

    python download_data.py            # real data from Yahoo Finance (needs internet)
    python download_data.py --demo     # SYNTHETIC random data, only to test the code
"""
import sys
import numpy as np
import pandas as pd
from features import fetch_stock

TICKERS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "ITC", "SBIN",
    "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "SUNPHARMA",
    "TITAN", "BAJFINANCE", "NESTLEIND", "ULTRACEMCO", "WIPRO", "HCLTECH", "TECHM",
    "POWERGRID", "NTPC", "ONGC", "TATAMOTORS", "TATASTEEL", "JSWSTEEL", "M&M",
    "ADANIPORTS", "COALINDIA", "DRREDDY", "CIPLA", "BRITANNIA", "GRASIM",
    "HINDALCO", "EICHERMOT", "HEROMOTOCO", "BPCL", "INDUSINDBK",
]
START = "2005-01-01"
OUT = "data/stock_data.csv"


def real():
    frames = []
    for t in TICKERS:
        try:
            df = fetch_stock(t + ".NS", start=START)
            df["Ticker"] = t
            frames.append(df)
            print(f"{t:12s} {len(df):5d} rows  {df.Date.min():%Y-%m-%d} -> {df.Date.max():%Y-%m-%d}")
        except Exception as e:
            print(f"{t:12s} FAILED: {e}")
    return pd.concat(frames, ignore_index=True)


def demo(n_stocks=15, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2008-01-01", pd.Timestamp.today().normalize())
    frames = []
    for t in TICKERS[:n_stocks]:
        r = rng.normal(0.0004, 0.015, len(dates))
        c = 100 * np.exp(np.cumsum(r))
        o = c * (1 + rng.normal(0, 0.003, len(dates)))
        h = np.maximum(o, c) * (1 + rng.uniform(0, 0.01, len(dates)))
        l = np.minimum(o, c) * (1 - rng.uniform(0, 0.01, len(dates)))
        v = rng.integers(1e6, 5e6, len(dates))
        frames.append(pd.DataFrame(dict(Date=dates, Open=o, High=h, Low=l, Close=c, Volume=v, Ticker=t)))
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    raw = demo() if "--demo" in sys.argv else real()
    raw.to_csv(OUT, index=False)
    print(f"\nSaved {len(raw):,} rows, {raw.Ticker.nunique()} stocks -> {OUT}")
    if "--demo" in sys.argv:
        print("WARNING: this is SYNTHETIC data. Do not report results from it.")
