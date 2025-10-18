import yfinance as yf
import ta
import pickle
import os
import asyncio
import datetime
from zoneinfo import ZoneInfo
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
signal_log = []

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

# === NLTK setup ===
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

def clear_old_alerts():
    """Keep only today's alerts so we don't resend or spam."""
    global alerted_signals
    today = datetime.date.today().isoformat()
    alerted_signals = {a for a in alerted_signals if a.startswith(today)}
    with open(ALERTS_FILE, "wb") as f:
        pickle.dump(alerted_signals, f)

# === Core signal check ===
async def check_signals():
    global alerted_signals
    clear_old_alerts()
    print("🔍 Checking for MACD crosses + RSI + trend filtering...")

    # download stock data
    try:
        data_dict = yf.download(
            tickers,
            period="6mo",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False
        )
    except Exception as e:
        print(f"⚠️ Error downloading data: {e}")
        return

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

            cross_up = macd_prev < signal_prev and macd_now > signal_now
            cross_down = macd_prev > signal_prev and macd_now < signal_now

            if (ticker in my_stocks and cross_down) or (ticker not in my_stocks and cross_up):
                cross_type = "CROSS_DOWN" if cross_down else "CROSS_UP"
                emoji = "🔴" if cross_down else "🔵"
                today = datetime.date.today().isoformat()
                signal_id = f"{today}_{ticker}_{cross_type}"

                if signal_id not in alerted_signals:
                    alerted_signals.add(signal_id)
                    msg = f"{emoji} MACD {cross_type.replace('_', ' ')}: {ticker} RSI = {rsi_now:.2f} Trend: {trend}"
                    await send_async_message(msg)  # <-- await here
                    print(f"📈 Alert sent: {msg}")

                    signal_log.append({
                        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "ticker": ticker,
                        "type": f"{emoji} MACD {cross_type.replace('_', ' ')}",
                        "rsi": f"{rsi_now:.2f}",
                        "trend": trend
                    })

                    with open(ALERTS_FILE, "wb") as f:
                        pickle.dump(alerted_signals, f)

        except Exception as e:
            print(f"⚠️ Error processing {ticker}: {e}")

# === Async Scheduler ===
async def schedule_bot():
    vancouver_tz = ZoneInfo("America/Vancouver")
    last_run_hour = None

    def log_run(message):
        timestamp = datetime.datetime.now(vancouver_tz).strftime("%Y-%m-%d %I:%M:%S %p")
        with open("run_log.txt", "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")

    # --- Startup message ---
    startup_msg = "✅ Bot started - running ..."
    print(startup_msg)
    await send_async_message(startup_msg)
    log_run(startup_msg)

    # --- First scan immediately ---
    await check_signals()
    log_run("✅ First scan complete.")

    while True:
        now = datetime.datetime.now(vancouver_tz)
        current_hour = now.hour

        if current_hour in [6, 12, 18] and current_hour != last_run_hour:
            run_msg = f"🕕 Running scheduled scan at {now.strftime('%I:%M %p')}..."
            print(run_msg)
            await send_async_message(run_msg)
            log_run(run_msg)

            await check_signals()
            last_run_hour = current_hour

            next_run_hour = {6: 12, 12: 18, 18: 6}[current_hour]
            hours_until_next = (next_run_hour - current_hour) % 24
            next_str = (now + datetime.timedelta(hours=hours_until_next)).strftime("%I:%M %p")
            complete_msg = f"✅ Run complete — next run in {hours_until_next} h at {next_str}"
            print(complete_msg)
            await send_async_message(complete_msg)
            log_run(complete_msg)

        await asyncio.sleep(900)

# === Flask keepalive thread ===
def run_flask():
    app.run(host="0.0.0.0", port=5000)


if __name__ == "__main__" and not os.environ.get("WERKZEUG_RUN_MAIN"):
    # Start Flask in the background
    threading.Thread(target=run_flask, daemon=True).start()
    
    # Run the async scheduler (starts scan immediately and then loops forever)
    asyncio.run(schedule_bot())





