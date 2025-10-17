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
        # If we're already inside an event loop (e.g., async context)
        asyncio.create_task(send_async_message(text))
    except RuntimeError:
        # No running loop, so start a new one
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
# === Daily summary lists ===
daily_cross_ups_new = set()     # MACD cross ups for tickers you don't own
daily_cross_down_owned = set()  # MACD cross downs for tickers you own

# === NLTK setup ===
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

# === MACD + RSI Signal Check ===
# === MACD + RSI Signal Check (fixed with summary + immediate alerts) ===
def check_signals():
    global alerted
    # Daily summary lists
    daily_cross_ups_new = []      # cross ups for tickers you don't own
    daily_cross_down_owned = []   # cross downs for tickers you own

    try:
        print("🔍 Checking MACD + RSI signals...")
        data_dict = yf.download(
            tickers,
            period="6mo",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False
        )

        for ticker in tickers:
            try:
                data = data_dict[ticker].copy()
                if data.empty:
                    continue

                # Localize timezone
                if data.index.tz is None:
                    data.index = data.index.tz_localize('UTC').tz_convert('America/New_York')
                else:
                    data.index = data.index.tz_convert('America/New_York')

                close = data["Close"].squeeze()

                # MACD and RSI
                macd_indicator = ta.trend.MACD(close, window_fast=12, window_slow=26, window_sign=9)
                data["macd"] = macd_indicator.macd()
                data["macd_signal"] = macd_indicator.macd_signal()
                data["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()
                data["50ma"] = close.rolling(window=50).mean()

                last_price = close.iloc[-1]
                last_rsi = data["rsi"].iloc[-1]
                trend = "Uptrend" if last_price > data["50ma"].iloc[-1] else "Downtrend"

                macd_cross_up = (data["macd"] > data["macd_signal"]) & (data["macd"].shift(1) <= data["macd_signal"].shift(1))
                macd_cross_down = (data["macd"] < data["macd_signal"]) & (data["macd"].shift(1) >= data["macd_signal"].shift(1))

                # === BUY SIGNALS (cross up) ===
                if macd_cross_up.iloc[-1] and ticker not in my_stocks and alerted.get(ticker) != "up":
                    msg = f"🔵 MACD CROSS UP: {ticker} RSI = {last_rsi:.2f} Trend: {trend}"
                    send_telegram_message(msg)  # immediate alert
                    daily_cross_ups_new.append(f"{ticker} | RSI={last_rsi:.2f} | Trend={trend}")
                    alerted[ticker] = "up"

                # === SELL SIGNALS (cross down) ONLY for owned stocks ===
                elif macd_cross_down.iloc[-1] and ticker in my_stocks and alerted.get(ticker) != "down":
                    msg = f"🔻 MACD CROSS DOWN: {ticker} RSI = {last_rsi:.2f} Trend: {trend}"
                    send_telegram_message(msg)  # immediate alert
                    daily_cross_down_owned.append(f"{ticker} | RSI={last_rsi:.2f} | Trend={trend}")
                    alerted[ticker] = "down"

            except Exception as e:
                print(f"Error processing {ticker}: {e}")

        # === End-of-run summary ===
        if daily_cross_ups_new:
            summary_up = "📊 MACD CROSS UP SUMMARY (new tickers today):\n" + "\n".join(daily_cross_ups_new)
            send_telegram_message(summary_up)

        if daily_cross_down_owned:
            summary_down = "📊 MACD CROSS DOWN SUMMARY (owned tickers today):\n" + "\n".join(daily_cross_down_owned)
            send_telegram_message(summary_down)

        if not daily_cross_ups_new and not daily_cross_down_owned:
            print("📭 No MACD cross signals today.")

        # === Save alerts ===
        with open(alert_file, "wb") as f:
            pickle.dump(alerted, f)

    except Exception as e:
        print(f"Error checking signals: {e}")


        # Detect if any new alerts happened during this run
        new_alerts_today = False
        for ticker in tickers:
            if macd_cross_up.iloc[-1] and alerted.get(ticker) == "up":
                new_alerts_today = True
            if macd_cross_down.iloc[-1] and alerted.get(ticker) == "down":
                new_alerts_today = True

        # Save alerts
        with open(alert_file, "wb") as f:
            pickle.dump(alerted, f)

        if not new_alerts_today:
            print("📭 No trade signals today.")
            send_telegram_message("📭 No trade signals today.")

    except Exception as e:
        print(f"Error fetching data: {e}")

        # Detect if any new alerts happened during this run
        new_alerts_today = False
        for ticker in tickers:
            if macd_cross_up.iloc[-1] and alerted.get(ticker) == "up":
                new_alerts_today = True
            if macd_cross_down.iloc[-1] and alerted.get(ticker) == "down":
                new_alerts_today = True

        # Save alerts
        with open(alert_file, "wb") as f:
            pickle.dump(alerted, f)

        if not new_alerts_today:
            print("📭 No trade signals today.")
            send_telegram_message("📭 No trade signals today.")


    except Exception as e:
        print(f"Error fetching data: {e}")

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

# === Pacific Time Scheduler (3 runs per day) ===
run_times = ["06:45", "10:00", "13:05"]  # PST/PDT
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

# Run immediate test on startup
check_signals()
check_news_alerts()
print("\n✅ Initial test complete. Now waiting for schedule...\n")


# === Main loop with live countdown ===
print("⏳ Scheduler started. Will run at PST/PDT times:", run_times)


while True:
    wait_seconds = seconds_until_next_run()
    next_run = datetime.datetime.now(pacific) + datetime.timedelta(seconds=wait_seconds)
    print(f"\n🕒 Next run scheduled at: {next_run.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    # Live countdown timer
    for remaining in range(int(wait_seconds), 0, -1):
        mins, secs = divmod(remaining, 60)
        timer_display = f"⏳ Time until next check: {mins:02d}:{secs:02d}"
        print(timer_display, end="\r", flush=True)
        time.sleep(1)

    print("\n🚀 Running checks now!\n")

    # Run your main functions
    check_signals()
    check_news_alerts()






