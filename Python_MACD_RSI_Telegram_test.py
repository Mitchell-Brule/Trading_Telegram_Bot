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
STARTUP_FILE = "startup_sent.txt"
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

    # Download all tickers at once
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

            # Extract latest values
            macd_prev, macd_now = df["MACD"].iloc[-2], df["MACD"].iloc[-1]
            signal_prev, signal_now = df["Signal"].iloc[-2], df["Signal"].iloc[-1]
            rsi_now = df["RSI"].iloc[-1]
            trend = "Uptrend" if df["MA50"].iloc[-1] > df["MA200"].iloc[-1] else "Downtrend"

            # Detect MACD signals
            cross_up = macd_prev < signal_prev and macd_now > signal_now
            cross_down = macd_prev > signal_prev and macd_now < signal_now

            # Determine signal direction based on holdings preference
            if (ticker in my_stocks and cross_down) or (ticker not in my_stocks and cross_up):
                cross_type = "CROSS_DOWN" if cross_down else "CROSS_UP"
                emoji = "🔴" if cross_down else "🔵"

                # Create unique ID per candle
                bar_date = df.index[-1].date()
                signal_id = f"{bar_date}_{ticker}_{cross_type}"

                # ✅ Stop duplicates: Only alert once per day per ticker
                if signal_id in alerted_signals:
                    continue  # Skip if alerted already today

                # ✅ Only alert if new signal is from TODAY
                if bar_date != datetime.date.today():
                    continue

                # ✅ Store signal so it won't repeat
                alerted_signals.add(signal_id)

                # Build alert message
                msg = f"{emoji} MACD {cross_type.replace('_', ' ')}: {ticker} RSI = {rsi_now:.2f} Trend: {trend}"
                await send_async_message(msg)
                print(f"📈 Alert sent: {msg}")

                # Log alert for dashboard
                signal_log.append({
                    "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "ticker": ticker,
                    "type": f"{emoji} MACD {cross_type.replace('_', ' ')}",
                    "rsi": f"{rsi_now:.2f}",
                    "trend": trend
                })

                # Persist alerted signals to disk
                with open(ALERTS_FILE, "wb") as f:
                    pickle.dump(alerted_signals, f)

        except Exception as e:
            print(f"⚠️ Error processing {ticker}: {e}")


# === Async Scheduler ===
async def schedule_bot():
    """
    Runs 3 scans per day (06:00, 12:00, 18:00 America/Vancouver).
    On startup it runs one immediate scan.
    """
    vancouver_tz = ZoneInfo("America/Vancouver")
    last_run_date = None   # date of the last completed run
    last_run_hour = None   # hour of the last completed run

    # --- Startup check ---
    try:
        today_str = str(datetime.date.today())
        last_startup = ""
        if os.path.exists(STARTUP_FILE):
            with open(STARTUP_FILE) as f:
                last_startup = f.read().strip()

        if last_startup != today_str:
            startup_msg = "✅ Bot started - Running 24/7 with 3 scans per day!"
            print(startup_msg)
            await send_async_message(startup_msg)

            # Run one immediate scan at startup
            print("🕒 Running initial startup scan...")
            await check_signals()
            print("✅ Initial startup scan complete.")

            # Mark startup done
            with open(STARTUP_FILE, "w") as f:
                f.write(today_str)
        else:
            print("✅ Bot already started today, skipping Telegram startup message and initial scan.")
    except Exception as e:
        print(f"⚠️ Error during startup: {e}")

    scheduled_hours = [6, 12, 18]  # Vancouver local time hours to run

    while True:
        try:
            now = datetime.datetime.now(vancouver_tz)
            current_hour = now.hour
            current_date = now.date()

            if current_hour in scheduled_hours:
                # Run only once per scheduled hour per day
                if last_run_hour != current_hour or last_run_date != current_date:
                    # If the date changed, clear old alerts (keeps memory small)
                    if last_run_date != current_date:
                        clear_old_alerts()

                    run_msg = f"🕕 Scan started at {now.strftime('%Y-%m-%d %I:%M %p %Z')}..."
                    print(run_msg)
                    try:
                        await send_async_message(run_msg)
                    except Exception as e:
                        print(f"⚠️ Failed to send run-start message: {e}")

                    # Run the main scan
                    try:
                        await check_signals()
                    except Exception as e:
                        print(f"⚠️ Error in check_signals(): {e}")

                    complete_msg = f"✅ Scan done — next scheduled run pending."
                    print(complete_msg)
                    try:
                        await send_async_message(complete_msg)
                    except Exception as e:
                        print(f"⚠️ Failed to send run-complete message: {e}")

                    # Persist last-run markers
                    last_run_hour = current_hour
                    last_run_date = current_date

            # Wake up frequently to stay responsive; Render + UptimeRobot will keep process alive
            await asyncio.sleep(60)

        except Exception as loop_exc:
            # Catch-all to ensure scheduler loop doesn't die
            print(f"🔥 Scheduler loop error, continuing: {loop_exc}")
            await asyncio.sleep(60)


# === Flask keepalive thread ===
def run_flask():
    app.run(host="0.0.0.0", port=5000)


if __name__ == "__main__" and not os.environ.get("WERKZEUG_RUN_MAIN"):
    # Start Flask in the background
    threading.Thread(target=run_flask, daemon=True).start()
    
    # Run the async scheduler (starts scan immediately and then loops forever)
    asyncio.run(schedule_bot())



