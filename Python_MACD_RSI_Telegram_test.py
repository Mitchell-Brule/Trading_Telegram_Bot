import yfinance as yf
import ta
import time
import pickle
import os
import asyncio
from telegram import Bot
from telegram.request import HTTPXRequest
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import datetime
from zoneinfo import ZoneInfo
from telegram.error import NetworkError
import nest_asyncio
nest_asyncio.apply()  # allows nested event loops


# === Telegram setup (async-safe, persistent session) ===
bot_token = "7481105387:AAHsNaOFEuMuWan2E1Y44VMrWeiZcxBjCAw"
chat_id = 7602575312

bot = Bot(
    token=bot_token,
    request=HTTPXRequest(connect_timeout=10, read_timeout=20, connection_pool_size=10)
)

async def send_async_message(text):
    """Send a Telegram message asynchronously with retries."""
    for attempt in range(3):
        try:
            await bot.send_message(chat_id=chat_id, text=text)
            print(f"📩 Telegram alert sent: {text}")
            return
        except NetworkError as e:
            print(f"🌐 Telegram network error (try {attempt+1}/3): {e}")
            await asyncio.sleep(3)
        except Exception as e:
            print(f"⚠️ Telegram send error: {e}")
            return

def send_telegram_message(text):
    """Run async Telegram message safely, even inside sync code."""
    try:
        loop = asyncio.get_running_loop()
        asyncio.create_task(send_async_message(text))
    except RuntimeError:
        asyncio.run(send_async_message(text))



# === List of tickers to check ===
tickers = [
    "ABBV", "ABNB", "ABT", "ADBE", "ADI", "ADP", "ADSK", "ADM", "AAPL", "ALGN", "AMAT", "AMGN", "AMD", "AMT", "AMZN", "ANET", "APD", "ASML", "AVB", "AVGO", "AXP",
    "BA", "BAC", "BDX", "BIIB", "BKNG", "BLK", "BMY", "BRK-B", "BX",
    "CAT", "CDNS", "CDW", "CHRW", "CHTR", "CI", "CL", "CLX", "CMCSA", "CMG", "COF", "COP", "COST", "CPB", "CRM", "CRWD", "CSCO", "CTAS", "CVX",
    "DE", "DELL", "DG", "DHR", "DIS", "DOCU", "DUK", "DVN", "DDOG",
    "EMR", "ENPH", "EOG", "EQIX", "ETN", "EW", "EXPE",
    "FAST", "FDX", "FIS", "FTNT",
    "GE", "GILD", "GIS", "GOOG", "GOOGL", "GS",
    "HCA", "HD", "HON", "HPE",
    "IBM", "ILMN", "INTC", "INTU", "ISRG",
    "JCI", "JPM",
    "KHC", "KIM", "KKR", "KLAC", "KMB", "KMI", "KO",
    "LAMR", "LIN", "LLY", "LMT", "LOW", "LRCX",
    "MA", "MAR", "MCD", "MCHP", "MDB", "MDT", "META", "MMC", "MNST", "MO", "MRK", "MS", "MSFT", "MSI", "MU",
    "NFLX", "NKE", "NOC", "NOW", "NTES", "NVDA", "NXPI",
    "OKTA", "ORCL", "ORLY",
    "PANW", "PAYC", "PEP", "PFE", "PG", "PGR", "PLD", "PM", "PSX", "PYPL",
    "QCOM",
    "REGN", "RMD", "ROST", "RTX",
    "SBUX", "SHOP", "SLB", "SNOW", "SNPS", "SO", "SPGI", "SPOT", "SYK",
    "TEAM", "T", "TMO", "TJX", "TMUS", "TSLA", "TSM", "TXN",
    "UNH", "UNP", "UPS",
    "V", "VZ",
    "WMT",
    "XOM", "ZM", "ZS"
]

# === Stocks you own (for sell alerts) ===
my_stocks = []


# === File to store previous alerts ===
alert_file = "alerts.pkl"
if os.path.exists(alert_file):
    try:
        with open(alert_file, "rb") as f:
            alerted = pickle.load(f)
            if not isinstance(alerted, dict):
                alerted = {}
    except Exception:
        alerted = {}
else:
    alerted = {}

# === NLTK setup ===
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()


# === MACD + RSI Signal Check ===
def check_signals():
    global alerted
    try:
        print("🔍 Checking for MACD crosses + RSI with trend filtering...")

        data_dict = yf.download(
            tickers,
            period="6mo",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False
        )

        any_alerts_today = False

        for ticker in tickers:
            try:
                data = data_dict[ticker].copy()
                close = data["Close"]

                # MACD
                macd_indicator = ta.trend.MACD(close, window_fast=12, window_slow=26, window_sign=9)
                data["macd"] = macd_indicator.macd()
                data["macd_signal"] = macd_indicator.macd_signal()
                data["macd_hist"] = macd_indicator.macd_diff()

                # RSI
                data["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()

                # MACD Cross logic
                cross_up = (data["macd"] > data["macd_signal"]) & (data["macd"].shift(1) <= data["macd_signal"].shift(1))
                cross_down = (data["macd"] < data["macd_signal"]) & (data["macd"].shift(1) >= data["macd_signal"].shift(1))

                # Trend filter
                sma = close.rolling(window=20).mean()
                trend_list = ["Uptrend" if close.iloc[i] > sma.iloc[i] else "Downtrend" for i in range(len(data))]

                latest_idx = data.index[-1]

                # === Cross Up (only for stocks you DON'T own) ===
                if cross_up.loc[latest_idx] and ticker not in my_stocks:
                    key = f"{ticker}_{latest_idx.date()}"
                    if alerted.get(key) != "up":
                        msg = f"🔵 MACD CROSS UP: {ticker} RSI = {data['rsi'].loc[latest_idx]:.2f} Trend: {trend_list[-1]}"
                        send_telegram_message(msg)
                        alerted[key] = "up"
                        any_alerts_today = True

                # === Cross Down (only for stocks you DO own) ===
                if cross_down.loc[latest_idx] and ticker in my_stocks:
                    key = f"{ticker}_{latest_idx.date()}"
                    if alerted.get(key) != "down":
                        msg = f"🔻 MACD CROSS DOWN: {ticker} RSI = {data['rsi'].loc[latest_idx]:.2f} Trend: {trend_list[-1]}"
                        send_telegram_message(msg)
                        alerted[key] = "down"
                        any_alerts_today = True

            except Exception as e:
                print(f"❌ Error processing {ticker}: {e}")

        # Save alerts
        with open(alert_file, "wb") as f:
            pickle.dump(alerted, f)

        if not any_alerts_today:
            print("📭 No MACD cross signals today.")
            send_telegram_message("📭 No MACD cross signals today.")

        print("✅ check_signals() complete.\n")

    except Exception as e:
        print(f"❌ Error in check_signals(): {e}")
        send_telegram_message(f"⚠️ Error in check_signals(): {e}")


# === News + Hype Alerts ===
NEWS_API_KEY = "be1ef3d5ba614c959c1c7b8b14744eda"
NEWS_API_ENDPOINT = "https://newsapi.org/v2/everything"

def check_news_alerts():
    print("📰 Checking for aggregated news sentiment...")
    for ticker in tickers[:10]:
        try:
            params = {
                "q": ticker,
                "sortBy": "publishedAt",
                "language": "en",
                "apiKey": NEWS_API_KEY,
                "pageSize": 10
            }
            response = requests.get(NEWS_API_ENDPOINT, params=params, timeout=15)
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
            num_positive = len([s for s in compound_scores if s > 0.6])
            num_negative = len([s for s in compound_scores if s < -0.6])
            total_articles = len(compound_scores)

            if avg_sentiment >= 0.8 and num_positive >= 3:
                msg = (
                    f"📈 {ticker}: strong positive sentiment!\n"
                    f"{num_positive}/{total_articles} articles are bullish.\n"
                    f"Avg sentiment: {avg_sentiment:.2f}"
                )
                send_telegram_message(msg)

            if ticker in my_stocks and avg_sentiment <= 0.3 and num_negative >= 3:
                msg = (
                    f"⚠️ {ticker}: trending negative.\n"
                    f"{num_negative}/{total_articles} articles bearish.\n"
                    f"Avg sentiment: {avg_sentiment:.2f}"
                )
                send_telegram_message(msg)

        except Exception as e:
            print(f"Error fetching news for {ticker}: {e}")


# === Pacific Time Scheduler ===
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

    next_run = min(future_times)
    return (next_run - now).total_seconds()


# === Startup confirmation ===
startup_msg = "✅ Bot started successfully — running first test."
print(startup_msg)
send_telegram_message(startup_msg)

# === Immediate test ===
check_signals()
check_news_alerts()
print("\n✅ Initial test complete. Now waiting for schedule...\n")

print("⏳ Scheduler started. Will run at PST/PDT times:", run_times)

while True:
    wait_seconds = seconds_until_next_run()
    next_run = datetime.datetime.now(pacific) + datetime.timedelta(seconds=wait_seconds)
    print(f"\n🕒 Next run scheduled at: {next_run.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    for remaining in range(int(wait_seconds), 0, -1):
        mins, secs = divmod(remaining, 60)
        print(f"⏳ Time until next check: {mins:02d}:{secs:02d}", end="\r", flush=True)
        time.sleep(1)

    print("\n🚀 Running checks now!\n")
    check_signals()
    check_news_alerts()








