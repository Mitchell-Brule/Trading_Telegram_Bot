import yfinance as yf
import ta
import pandas as pd

tickers = ["HD", "AAPL", "MSFT", "GOOG", "AMZN"]
period = "3mo"
interval = "1h"

results = {}

for ticker in tickers:
    data = yf.download(ticker, period=period, interval=interval, auto_adjust=False)

    if data.empty:
        print(f"⚠️ No data for {ticker}")
        continue

    # Ensure close is 1D
    close = data["Close"].squeeze()

    macd_indicator = ta.trend.MACD(close, window_fast=12, window_slow=26, window_sign=9)
    macd = macd_indicator.macd().iloc[-1]
    signal = macd_indicator.macd_signal().iloc[-1]
    hist = macd_indicator.macd_diff().iloc[-1]

    results[ticker] = {"MACD": macd, "Signal": signal, "Histogram": hist}

pd.set_option("display.float_format", "{:.4f}".format)
df_results = pd.DataFrame(results).T
print("\nMost recent MACD values:")
print(df_results)

