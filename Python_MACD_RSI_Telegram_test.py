import os
import time
import pickle
import asyncio
import datetime
from zoneinfo import ZoneInfo
import logging

import yfinance as yf
import ta
import requests
from telegram import Bot
from telegram.request import HTTPXRequest
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# === Environment Variables for Security ===
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN")
CHAT_ID = int(os.getenv("TELEGRAM_CHAT_ID", 0))
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "YOUR_NEWSAPI_KEY")

# === Logging setup ===
logging.basicConfig(
    filename='bot.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
logging.info("Bot started")

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
tickers = [
    "AAPL","MSFT","AMZN","NVDA","TSLA","GOOGL","GOOG","META","BRK-B","JPM",
    "V","MA","UNH","HD","PG","BAC","XOM","CVX","PFE","MRK",
    "VZ","T","NFLX","INTC","CSCO","ORCL","CRM","ABNB","AMD","QCOM",
    "NKE","MCD","SBUX","KO","PEP","COST","WMT","DIS","ABBV","LLY",
    # ... add remaining tickers
]
my_stocks = []  # stocks you own

# === Alert Persistence ===
alert_file = "alerts.pkl"
if os.path.exists(alert_file):
    with open(alert_file, "rb") as f:
        alerted = pickle.load(f)
    if not isinstance(alerted, dict):
        alerted = {}
else:
    alerted = {}

# === Cache for fundamentals to reduce API calls ===
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
        logging.info("Checking signals with volume, multi-timeframe, and fundamentals...")
        data_hourly = yf.download(tickers, period="3mo", interval="1h", group_by="ticker", auto_adjust=True)
        data_daily = yf.download(tickers, period="6mo", interval="1d", group_by="ticker", auto_adjust=True)

        for ticker in tickers:
            try:
                data_h = data_hourly[ticker].copy()
                close_h = data_h["Close"].squeeze()
                volume_h = data_h["Volume"].squeeze()

                # Hourly indicators
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

                # Multi-timeframe confirmation
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
                        logging.info(msg)
                        alerted[ticker] = "up"

                # Sell signal
                elif macd_cross_down.iloc[-1] and last_vol > avg_vol and macd_d < macd_signal_d and rsi_d < 50 and fundamentals_ok:
                    if alerted.get(ticker) != "down":
                        msg = f"⚠️ SELL SIGNAL: {ticker} | RSI={last_rsi:.2f} | Trend={trend} | PE={pe} | EPS growth={eps_growth:.2f}"
                        send_telegram_message(msg)
                        logging.info(msg)
                        alerted[ticker] = "down"

            except Exception as e:
                logging.error(f"Error processing {ticker}: {e}")

        # Save alert states
        with open(alert_file, "wb") as f:
            pickle.dump(alerted, f)

    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        send_telegram_message(f"⚠️ Error fetching data: {e}")

# === News Sentiment Alerts ===
def check_news_alerts():
    logging.info("Checking news sentiment for all tickers...")
    for ticker in tickers:
        try:
            params = {
                "q": ticker,
                "sortBy": "publishedAt",
                "language": "en",
                "apiKey": NEWS_API_KEY,
                "pageSize": 10
            }
            response = requests.get("https://newsapi.org/v2/everything", params=params, timeout=10)
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
                logging.info(msg)

            if ticker in my_stocks and avg_sentiment <= 0.15 and num_negative >= 3:
                msg = f"⚠️ {ticker} VERY NEGATIVE sentiment! {num_negative}/{total_articles} articles | Avg sentiment: {avg_sentiment:.2f}"
                send_telegram_message(msg)
                logging.info(msg)

        except Exception as e:
            logging.error(f"Error fetching news for {ticker}: {e}")

# === Scheduler ===
run_times = ["06:45", "10:00", "13:05"]
pacific = ZoneInfo("America/Los_Angeles")

def seconds_until_next_run():
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
    return (min(future_times) - now).total_seconds()

# === Main Loop with Countdown & Crash Notification ===
while True:
    try:
        wait_seconds = seconds_until_next_run()
        next_run_time = datetime.datetime.now(pacific) + datetime.timedelta(seconds=wait_seconds)
        logging.info(f"Next run at {next_run_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")

        # Countdown loop
        while wait_seconds > 0:
            minutes = int(wait_seconds // 60)
            seconds_remain = int(wait_seconds % 60)
            print(f"\r⏳ Next run at {next_run_time.strftime('%H:%M:%S %Z')} (in {minutes}m {seconds_remain}s) ", end="")
            time.sleep(10)  # update every 10 seconds
            wait_seconds -= 10

        print()  # new line after countdown

        now = datetime.datetime.now(pacific)
        logging.info(f"Running checks at {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        check_signals()
        check_news_alerts()

    except Exception as e:
        err_msg = f"⚠️ Bot crashed: {e}"
        logging.error(err_msg)
        send_telegram_message(err_msg)
        time.sleep(60)


