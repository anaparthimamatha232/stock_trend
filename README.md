# Stock Market Trend Classification on Historical Data

A machine-learning project that classifies whether a stock will go **UP or DOWN** over the next **1 day, 1 week or 1 month**, and also shows exactly what the stock did over a past period you choose. Includes a Streamlit web app.

## Project Overview

- **Past view (exact):** pick a stock and a period (1 day … 5 years) and the app calculates from closing prices whether it rose or fell, and by how much.
- **Future view (ML):** the trained model predicts UP / DOWN for the next 1 day / 1 week / 1 month, with a probability.
- This is a **classification** project (direction), not a price-prediction project.

## Dataset

- **Source:** Yahoo Finance via the `yfinance` library (daily OHLCV, split/dividend adjusted)
- **Stocks:** 40 large NSE stocks (list in `download_data.py`)
- **Period:** 2005 → today (downloaded at the time you run `download_data.py`)
- **Size:** roughly 150,000+ daily rows before feature creation
- **Features (32):** returns, SMA/EMA ratios, MACD, RSI, Stochastic, Bollinger %B/width, ATR, volatility, volume ratio, OBV slope, 52-week high/low distance, gap, daily range, day-of-week, month
- **Targets:** `target_1D`, `target_1W` (5 trading days), `target_1M` (21 trading days): 1 if the close after that many days is higher than today's close

## Method

1. Download data and build features using past data only (`features.py`, shared by notebook and app)
2. Create UP/DOWN targets for three horizons
3. **Date-based split** (70% train / 15% validation / 15% test) with a purge gap so labels never leak across sets
4. Train Logistic Regression, Random Forest and Hist Gradient Boosting; choose by validation AUC
5. Re-train best model on train + validation, evaluate once on the test period (accuracy vs majority-class baseline, AUC, precision, recall, F1, confusion matrix)
6. Permutation feature importance
7. Save models + metadata, serve with Streamlit

## Important note on results

Stock direction is very noisy. Honest results are usually **AUC ≈ 0.50–0.58** and accuracy only slightly above the majority-class baseline. Results far above that (e.g. 80–90% accuracy) almost always mean data leakage. Run the notebook and report **your own** numbers; this README intentionally contains none.

## Project Structure

```
stock-trend-classification/
├── data/                 # stock_data.csv is created by download_data.py
├── models/               # created by the notebook (model_1D/1W/1M.joblib, metadata.json)
├── reports/              # charts saved by the notebook
├── features.py           # feature engineering, targets, split, past-move calculation
├── download_data.py      # downloads all stocks into data/stock_data.csv
├── stock_trend_classification.ipynb   # full pipeline
├── app.py                # Streamlit app
├── requirements.txt
└── README.md
```

## How to Run

```bash
pip install -r requirements.txt

# 1. Download real data (needs internet, takes a few minutes)
python download_data.py

# 2. Train: open the notebook and Kernel -> Restart & Run All
jupyter notebook stock_trend_classification.ipynb

# 3. Run the app
streamlit run app.py
```

If Yahoo Finance is unreachable you can test the code with synthetic data: `python download_data.py --demo`. That data is random, so never report results from it.

The app fetches live prices (cached for 1 hour) and falls back to `data/stock_data.csv` if offline.

## Limitations

- Uses only price/volume indicators (no news, fundamentals or market-wide data)
- Past patterns may not repeat; predictions are probabilistic
- **Educational project; not financial advice**

## Future Improvements

- Add NIFTY index and sector features; news sentiment
- Compare with an LSTM sequence model
- Walk-forward backtesting and a simple trading-strategy simulation

## Acknowledgements

Built as part of the NIELIT Chennai PG Program in Data Engineering. Project Guide: Mr. Mayank Singh Gautam.
