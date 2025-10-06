import os
import time
import pickle
import asyncio
import datetime
import threading
from zoneinfo import ZoneInfo
import logging

import yfinance as yf
import ta
import requests
from telegram import Bot
from telegram.request import HTTPXRequest
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from flask import Flask

# === Logging setup ===
logging.basicConfig(
    filename='bot.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
logging.info("Bot started")

# === Environment Variables ===
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
#7481105387:AAHsNaOFEuMuWan2E1Y44VMrWeiZcxBjCAw
CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID"))
#7602575312
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
#be1ef3d5ba614c959c1c7b8b14744eda
NEWS_API_ENDPOINT = "https://newsapi.org/v2/everything"

# === Telegram Setup ===
bot = Bot(token=BOT_TOKEN, request=HTTPXRequest())

async def send_async_message(text):
    await bot.send_message(chat_id=CHAT_ID, text=text)

def send_telegram_message(text):
    try:
        asyncio.run(send_async_message(text))
        logging.info(f"Telegram alert sent: {text}")
    except Exception as e:
        logging.error(f"Telegram send error: {e}")

# === NLTK Sentiment Setup ===
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# === Tickers & Portfolio ===
tickers = ["AAPL","MSFT","AMZN"]
my_stocks = []

# === Alert Persistence ===
alert_file = "alerts.pkl"
if os.path.exists(alert_file):
    with open(alert_file, "rb") as f:
        alerted = pickle.load(f)
    if not isinstance(alerted, dict):
        alerted = {}
else:
    alerted = {}

# === Cache for fundamentals ===
fundamentals_cache = {}

def get_fundamentals(ticker):
    if ticker in fundamentals_cache:
        return fundamentals_cache[ticker]
    try:
        info = yf.Ticker(ticker).info
        pe = info.get("trailingPE", 1000)
        eps_growth = info.get("earningsQuarterlyGrowth", 0)
        dividend_yield = info.get("dividendYield", 0)
        fundamentals_cache[ticker] = (pe, eps_growth, dividend_yield)
        return pe, eps_growth, dividend_yield
    except Exception:
        return 1000, 0, 0

# === MACD + RSI Signal Check ===
def check_signals():
    global alerted
    try:
        data_hourly = yf.download(tickers, period="3mo", interval="1h", group_by="ticker", auto_adjust=True)
        data_daily = yf.download(tickers, period="6mo", interval="1d", group_by="ticker", auto_adjust=True)

        for ticker in tickers:
            try:
                data_h = data_hourly[ticker].copy()
                close_h = data_h["Close"].squeeze()
                volume_h = data_h["Volume"].squeeze()

                data_h["macd"] = ta.trend.macd(close_h)
                data_h["macd_signal"] = ta.trend.macd_signal(close_h)
                data_h["rsi"] = ta.momentum.rsi(close_h, window=14)
                data_h["50ma"] = close_h.rolling(window=50).mean()
                data_h["vol_ma"] = volume_h.rolling(window=20).mean()

                last_price = close_h.iloc[-1]
                last_rsi = data_h["rsi"].iloc[-1]
                last_50ma = data_h["50ma"].iloc[-1]
                last_vol = volume_h.iloc[-1]
                avg_vol = data_h["vol_ma"].iloc[-1]
                trend = "Uptrend" if last_price > last_50ma else "Downtrend"

                macd_cross_up = (data_h["macd"] > data_h["macd_signal"]) & (data_h["macd"].shift(1) <= data_h["macd_signal"].shift(1))
                macd_cross_down = (data_h["macd"] < data_h["macd_signal"]) & (data_h["macd"].shift(1) >= data_h["macd_signal"].shift(1))

                pe, eps_growth, dividend_yield = get_fundamentals(ticker)
                fundamentals_ok = (pe < 40) and (eps_growth > 0.05)

                data_d = data_daily[ticker].copy()
                close_d = data_d["Close"].squeeze()
                macd_d = ta.trend.macd(close_d).iloc[-1]
                macd_signal_d = ta.trend.macd_signal(close_d).iloc[-1]
                rsi_d = ta.momentum.rsi(close_d, window=14).iloc[-1]

                # Buy signal
                if macd_cross_up.iloc[-1] and last_vol > avg_vol and macd_d > macd_signal_d and rsi_d > 50 and fundamentals_ok:
                    if alerted.get(ticker) != "up":
                        msg = f"🚀 BUY SIGNAL: {ticker} | RSI={last_rsi:.2f} | Trend={trend} | PE={pe} | EPS growth={eps_growth:.2f}"
                        send_telegram_message(msg)
                        alerted[ticker] = "up"

                # Sell signal
                elif macd_cross_down.iloc[-1] and last_vol > avg_vol and macd_d < macd_signal_d and rsi_d < 50 and fundamentals_ok:
                    if alerted.get(ticker) != "down":
                        msg = f"⚠️ SELL SIGNAL: {ticker} | RSI={last_rsi:.2f} | Trend={trend} | PE={pe} | EPS growth={eps_growth:.2f}"
                        send_telegram_message(msg)
                        alerted[ticker] = "down"

            except Exception as e:
                logging.error(f"Error processing {ticker}: {e}")

        with open(alert_file, "wb") as f:
            pickle.dump(alerted, f)

    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        send_telegram_message(f"⚠️ Error fetching data: {e}")

# === News Sentiment Alerts ===
def check_news_alerts():
    for ticker in tickers:
        try:
            params = {
                "q": ticker,
                "sortBy": "publishedAt",
                "language": "en",
                "apiKey": NEWS_API_KEY,
                "pageSize": 10
            }
            response = requests.get(NEWS_API_ENDPOINT, params=params, timeout=10)
            data = response.json()
            if "articles" not in data or not data["articles"]:
                continue

            compound_scores = []
            for article in data["articles"]:
                text = (article.get("title") or "") + ". " + (article.get("description") or "")
                if text.strip() and len(text) < 1000:
                    sentiment = sia.polarity_scores(text)
                    compound_scores.append(sentiment["compound"])

            if not compound_scores:
                continue

            avg_sentiment = sum(compound_scores) / len(compound_scores)
            num_positive = len([s for s in compound_scores if s > 0.8])
            num_negative = len([s for s in compound_scores if s < -0.8])
            total_articles = len(compound_scores)

            if avg_sentiment >= 0.85 and num_positive >= 3:
                msg = f"📈 {ticker} VERY POSITIVE sentiment! {num_positive}/{total_articles} articles | Avg sentiment: {avg_sentiment:.2f}"
                send_telegram_message(msg)

            if ticker in my_stocks and avg_sentiment <= 0.15 and num_negative >= 3:
                msg = f"⚠️ {ticker} VERY NEGATIVE sentiment! {num_negative}/{total_articles} articles | Avg sentiment: {avg_sentiment:.2f}"
                send_telegram_message(msg)

        except Exception as e:
            logging.error(f"Error fetching news for {ticker}: {e}")

# === Bot loop ===
def bot_loop():
    pacific = ZoneInfo("America/Los_Angeles")
    run_times = ["06:45", "10:00", "13:05"]

    while True:
        try:
            now = datetime.datetime.now(pacific)
            today = now.date()
            future_times = []

            for t in run_times:
                h, m = map(int, t.split(":"))
                run_dt = datetime.datetime.combine(today, datetime.time(h, m), tzinfo=pacific)
                if run_dt > now:
                    future_times.append(run_dt)

            if not future_times:
                h, m = map(int, run_times[0].split(":"))
                run_dt = datetime.datetime.combine(today + datetime.timedelta(days=1), datetime.time(h, m), tzinfo=pacific)
                future_times.append(run_dt)

            wait_seconds = (min(future_times) - now).total_seconds()
            logging.info(f"Sleeping for {wait_seconds} seconds until next run.")
            time.sleep(wait_seconds)

            check_signals()
            check_news_alerts()

        except Exception as e:
            logging.error(f"Bot crashed: {e}")
            send_telegram_message(f"⚠️ Bot crashed: {e}")
            time.sleep(60)

# === Flask server to keep free tier alive ===
app = Flask(__name__)
from flask import request

@app.route("/test")
def test_alert():
    try:
        message = "🚀 Test alert from your Render Trading Bot — it’s working!"
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": message}
        requests.post(url, json=payload)
        return "✅ Test alert sent to Telegram!", 200
    except Exception as e:
        logging.error(f"Test alert failed: {e}")
        return f"❌ Error: {e}", 500





