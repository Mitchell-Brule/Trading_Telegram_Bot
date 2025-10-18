import yfinance as yf
import ta
import time
import pickle
import os
import asyncio
import datetime
from zoneinfo import ZoneInfo
import requests
from telegram import Bot
from telegram.request import HTTPXRequest
from telegram.error import NetworkError
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from flask import Flask, render_template_string
import threading
import nest_asyncio

nest_asyncio.apply()  # allows nested event loops

# === Telegram setup ===
bot_token = "7481105387:AAHsNaOFEuMuWan2E1Y44VMrWeiZcxBjCAw"
chat_id = 7602575312

bot = Bot(
    token=bot_token,
    request=HTTPXRequest(connect_timeout=10, read_timeout=20, connection_pool_size=10)
)

async def send_async_message(text):
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
    try:
        loop = asyncio.get_running_loop()
        asyncio.create_task(send_async_message(text))
    except RuntimeError:
        asyncio.run(send_async_message(text))

# === Flask web log setup ===
app = Flask(__name__)
signal_log = []  # memory of all alerts

@app.route("/")
def index():
    html = """
    <html>
        <head>
            <title>Trading Bot Logs</title>
            <meta http-equiv="refresh" content="30">
            <style>
                body { font-family: Arial; background-color: #111; color: #eee; text-align: center; }
                h1 { color: #4CAF50; }
                table { margin: auto; border-collapse: collapse; width: 90%; }
                td, th { border: 1px solid #444; padding: 8px; }
                tr:nth-child(even) { background-color: #222; }
                .up { color: #4CAF50; }
                .down { color: #F44336; }
            </style>
        </head>
        <body>
            <h1>📈 Trading Bot Signals</h1>
            <p>Last updated: {{ last_update }}</p>
            <table>
                <tr><th>Time</th><th>Ticker</th><th>Type</th><th>RSI</th><th>Trend</th></tr>
                {% for s in signals %}
                    <tr>
                        <td>{{ s['time'] }}</td>
                        <td>{{ s['ticker'] }}</td>
                        <td class="{{ 'up' if 'UP' in s['type'] else 'down' }}">{{ s['type'] }}</td>
                        <td>{{ s['rsi'] }}</td>
                        <td>{{ s['trend'] }}</td>
                    </tr>
                {% endfor %}
            </table>
        </body>
    </html>
    """
    return render_template_string(
        html,
        signals=signal_log[-100:],
        last_update=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

# === Stock lists ===
tickers = [
    "ABBV","ABNB","ABT","ADBE","ADI","ADP","ADSK","ADM","AAPL","ALGN","AMAT","AMGN","AMD","AMT","AMZN","ANET","APD","ASML","AVB","AVGO","AXP",
    "BA","BAC","BDX","BIIB","BKNG","BLK","BMY","BRK-B","BX","CAT","CDNS","CDW","CHRW","CHTR","CI","CL","CLX","CMCSA","CMG","COF","COP","COST","CPB",
    "CRM","CRWD","CSCO","CTAS","CVX","DE","DELL","DG","DHR","DIS","DOCU","DUK","DVN","DDOG","EMR","ENPH","EOG","EQIX","ETN","EW","EXPE",
    "FAST","FDX","FIS","FTNT","GE","GILD","GIS","GOOG","GOOGL","GS","HCA","HD","HON","HPE","IBM","ILMN","INTC","INTU","ISRG","JCI","JPM",
    "KHC","KIM","KKR","KLAC","KMB","KMI","KO","LAMR","LIN","LLY","LMT","LOW","LRCX","MA","MAR","MCD","MCHP","MDB","MDT","META","MMC","MNST",
    "MO","MRK","MS","MSFT","MSI","MU","NFLX","NKE","NOC","NOW","NTES","NVDA","NXPI","OKTA","ORCL","ORLY","PANW","PAYC","PEP","PFE","PG","PGR",
    "PLD","PM","PSX","PYPL","QCOM","REGN","RMD","ROST","RTX","SBUX","SHOP","SLB","SNOW","SNPS","SO","SPGI","SPOT","SYK","TEAM","T","TMO","TJX",
    "TMUS","TSLA","TSM","TXN","UNH","UNP","UPS","V","VZ","WMT","XOM","ZM","ZS"
]
my_stocks = []

# === File for alert memory ===
ALERTS_FILE = "alerted_signals.pkl"
if os.path.exists(ALERTS_FILE):
    with open(ALERTS_FILE, "rb") as f:
        alerted_signals = pickle.load(f)
else:
    alerted_signals = set()

# === NLTK ===
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

# === Core signal check ===
def check_signals():
    global alerted_signals
    print("🔍 Checking for MACD crosses + RSI + trend filtering...")

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
            df = data_dict[ticker].dropna()
            macd_indicator = ta.trend.MACD(df["Close"])
            df["MACD"] = macd_indicator.macd()
            df["Signal"] = macd_indicator.macd_signal()
            df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
            df["MA50"] = df["Close"].rolling(window=50).mean()
            df["MA200"] = df["Close"].rolling(window=200).mean()

            macd_prev, macd_now = df["MACD"].iloc[-2], df["MACD"].iloc[-1]
            signal_prev, signal_now = df["Signal"].iloc[-2], df["Signal"].iloc[-1]
            rsi_now = df["RSI"].iloc[-1]
            trend = "Uptrend" if df["MA50"].iloc[-1] > df["MA200"].iloc[-1] else "Downtrend"

            # Detect MACD cross
            cross_up = macd_prev < signal_prev and macd_now > signal_now
            cross_down = macd_prev > signal_prev and macd_now < signal_now

            # Only alert:
            # - cross DOWN if you OWN it
            # - cross UP if you DON'T own it
            if (ticker in my_stocks and cross_down) or (ticker not in my_stocks and cross_up):
                cross_type = "🔴 MACD CROSS DOWN" if cross_down else "🔵 MACD CROSS UP"
                signal_id = f"{ticker}_{cross_type}"

                if signal_id not in alerted_signals:
                    alerted_signals.add(signal_id)
                    msg = f"{cross_type}: {ticker} RSI = {rsi_now:.2f} Trend: {trend}"
                    send_telegram_message(msg)

                    signal_log.append({
                        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "ticker": ticker,
                        "type": cross_type,
                        "rsi": f"{rsi_now:.2f}",
                        "trend": trend
                    })

        except Exception as e:
            print(f"⚠️ Error on {ticker}: {e}")
            continue

    # Save alert memory
    with open(ALERTS_FILE, "wb") as f:
        pickle.dump(alerted_signals, f)

    print("✅ check_signals() complete.")

# === News check ===
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
                send_telegram_message(
                    f"📈 {ticker}: strong positive sentiment!\n{num_positive}/{total_articles} bullish.\nAvg sentiment: {avg_sentiment:.2f}"
                )

            if ticker in my_stocks and avg_sentiment <= 0.3 and num_negative >= 3:
                send_telegram_message(
                    f"⚠️ {ticker}: trending negative.\n{num_negative}/{total_articles} bearish.\nAvg sentiment: {avg_sentiment:.2f}"
                )

        except Exception as e:
            print(f"Error fetching news for {ticker}: {e}")

# === Scheduler ===
def get_next_run_time():
    pst = ZoneInfo("America/Los_Angeles")
    now = datetime.datetime.now(pst)
    schedule_times = ["06:45", "10:00", "13:05"]
    for t in schedule_times:
        run_time = datetime.datetime.strptime(t, "%H:%M").replace(
            year=now.year, month=now.month, day=now.day, tzinfo=pst
        )
        if run_time > now:
            return run_time
    tomorrow = now + datetime.timedelta(days=1)
    return datetime.datetime.strptime(schedule_times[0], "%H:%M").replace(
        year=tomorrow.year, month=tomorrow.month, day=tomorrow.day, tzinfo=pst
    )

def scheduler_loop():
    print("⏳ Scheduler started. Will run at PST/PDT times: ['06:45', '10:00', '13:05']")
    while True:
        next_run = get_next_run_time()
        print(f"🕒 Next run scheduled at: {next_run.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        sleep_seconds = (next_run - datetime.datetime.now(ZoneInfo('America/Los_Angeles'))).total_seconds()
        time.sleep(max(0, sleep_seconds))
        check_signals()
        check_news_alerts()

# === Flask thread + startup ===
def run_flask():
    app.run(host="0.0.0.0", port=10000)

threading.Thread(target=run_flask, daemon=True).start()

# === Flask thread + startup ===
def run_flask():
    app.run(host="0.0.0.0", port=10000)

threading.Thread(target=run_flask, daemon=True).start()

# === STARTUP CONTROL ===
LAST_START_FILE = "last_start.txt"

def already_started_today():
    """Check if bot has already announced start today."""
    today = datetime.date.today().isoformat()
    if os.path.exists(LAST_START_FILE):
        with open(LAST_START_FILE, "r") as f:
            last = f.read().strip()
        if last == today:
            return True
    with open(LAST_START_FILE, "w") as f:
        f.write(today)
    return False

# === MAIN LOOP ===
if not already_started_today():
    send_telegram_message("✅ Bot started successfully — running first test.")

check_signals()
check_news_alerts()
print("✅ Initial test complete. Now waiting for schedule...")

def scheduler_loop():
    print("⏳ Scheduler started. Will run at PST/PDT times: ['06:45', '10:00', '13:05']")
    while True:
        next_run = get_next_run_time()
        next_str = next_run.strftime("%Y-%m-%d %H:%M:%S %Z")
        print(f"🕒 Next run scheduled at: {next_str}")

        sleep_seconds = (next_run - datetime.datetime.now(ZoneInfo('America/Los_Angeles'))).total_seconds()
        time.sleep(max(0, sleep_seconds))

        # --- Run the checks ---
        check_signals()
        check_news_alerts()

        # --- End of run message ---
        send_telegram_message(f"✅ Daily checks complete — next run at {next_str}")

scheduler_loop()




