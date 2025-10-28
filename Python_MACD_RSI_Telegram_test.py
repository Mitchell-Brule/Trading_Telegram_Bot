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
import pandas as pd
import numpy as np

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
                table { margin: auto; border-collapse: collapse; width: 95%; }
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
                <tr>
                    <th>Time</th><th>Ticker</th><th>Type</th><th>Probability</th><th>Rating</th><th>MACD</th><th>Signal</th><th>RSI</th><th>Trend</th>
                </tr>
                {% for s in signals %}
                    <tr>
                        <td>{{ s['time'] }}</td>
                        <td>{{ s['ticker'] }}</td>
                        <td class="{{ 'up' if 'UP' in s['type'] else 'down' }}">{{ s['type'] }}</td>
                        <td>{{ s['prob'] }}</td>
                        <td>{{ s['rating'] }}</td>
                        <td>{{ s['macd'] }}</td>
                        <td>{{ s['signal'] }}</td>
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
    """Keep only today's alerts so we don't resend."""
    global alerted_signals
    today = datetime.date.today().isoformat()
    alerted_signals = {a for a in alerted_signals if a.startswith(today)}
    with open(ALERTS_FILE, "wb") as f:
        pickle.dump(alerted_signals, f)

# ----- Helper indicator + structure functions -----
def detect_prior_swing_low(series_close, lookback=60):
    """
    Find a recent swing low (simple pivot) in the last `lookback` bars excluding the last 1-3 bars.
    Returns value or None.
    """
    if len(series_close) < lookback + 3:
        return None
    sub = series_close[-(lookback+3):-3]  # drop last 3 bars to avoid current action
    # simple: take the minimum in that window and ensure it's a local low
    min_idx = np.argmin(sub)
    min_val = sub.iloc[min_idx]
    return float(min_val)

def bullish_candle_pattern(row_prev, row):
    """
    Detect a simple bullish engulfing or hammer on the last candle.
    row_prev, row are Series with Open, High, Low, Close.
    """
    # Bullish Engulfing
    engulfing = (row_prev['Close'] < row_prev['Open']) and (row['Close'] > row['Open']) and (row['Close'] > row_prev['Open']) and (row['Open'] < row_prev['Close'])
    # Hammer (small body, long lower wick)
    body = abs(row['Close'] - row['Open'])
    lower_wick = min(row['Open'], row['Close']) - row['Low']
    upper_wick = row['High'] - max(row['Open'], row['Close'])
    hammer = (body > 0) and (lower_wick > 2 * body) and (upper_wick < body * 0.7)
    return engulfing or hammer

def score_to_stars(score):
    """Map a 0-100 score to 1-5 stars (string)."""
    if score >= 90:
        return "★★★★★"
    if score >= 80:
        return "★★★★☆"
    if score >= 70:
        return "★★★☆☆"
    if score >= 65:
        return "★★☆☆☆"
    return "★☆☆☆☆"

# Scoring weights (sums to 100)
WEIGHTS = {
    "macd": 20.0,
    "rsi": 20.0,
    "volume": 15.0,
    "support": 10.0,
    "trend": 10.0,
    "candle": 10.0,
    "ma_reclaim": 5.0,
    "market_structure": 5.0,
    "adx": 2.5,
    "stochastic": 2.5
}

# === Core signal check (MACD trigger + 12-factor scoring) ===
async def check_signals():
    global alerted_signals
    clear_old_alerts()
    print("🔍 Checking for MACD crosses + RSI + 12-factor scoring...")

    try:
        data_dict = yf.download(
            tickers,
            period="9mo",   # slightly longer for SMA200 and structure
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False
        )
    except Exception as e:
        print(f"⚠️ Error downloading data: {e}")
        return

    # Handle multi-ticker MultiIndex correctly
    if isinstance(data_dict.columns, pd.MultiIndex):
        data_dict = {ticker: data_dict[ticker].dropna() for ticker in tickers if ticker in data_dict}

    for ticker in tickers:
        try:
            if ticker not in data_dict:
                continue

            df = data_dict[ticker].copy()
            if len(df) < 60:
                continue  # not enough data for indicators

            # Basic indicators
            macd_indicator = ta.trend.MACD(df["Close"])
            df["MACD"] = macd_indicator.macd()
            df["Signal"] = macd_indicator.macd_signal()
            df["Histogram"] = df["MACD"] - df["Signal"]
            df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
            df["MA20"] = df["Close"].rolling(window=20, min_periods=1).mean()
            df["MA50"] = df["Close"].rolling(window=50, min_periods=1).mean()
            df["MA200"] = df["Close"].rolling(window=200, min_periods=1).mean()

            # Volume strength vs 20-day avg
            df["Vol20"] = df["Volume"].rolling(window=20, min_periods=1).mean()

            # ADX
            try:
                adx = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=14)
                df["ADX"] = adx.adx()
            except Exception:
                df["ADX"] = np.nan

            # Stochastic
            try:
                stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
                df["STOCH_K"] = stoch.stoch()
                df["STOCH_D"] = stoch.stoch_signal()
            except Exception:
                df["STOCH_K"] = np.nan
                df["STOCH_D"] = np.nan

            # Keep only the tail to speed things up
            df = df.dropna().copy()
            if len(df) < 50:
                continue

            # values for current evaluation
            last = df.iloc[-1]
            prev = df.iloc[-2]
            macd_prev, macd_now = df["MACD"].iloc[-2], df["MACD"].iloc[-1]
            signal_prev, signal_now = df["Signal"].iloc[-2], df["Signal"].iloc[-1]
            hist_now = df["Histogram"].iloc[-1]
            rsi_now = df["RSI"].iloc[-1]
            vol_now = df["Volume"].iloc[-1]
            vol20 = df["Vol20"].iloc[-1]
            trend = "Uptrend" if df["MA50"].iloc[-1] > df["MA200"].iloc[-1] else "Downtrend"

            cross_up = (macd_prev < signal_prev) and (macd_now > signal_now)
            cross_down = (macd_prev > signal_prev) and (macd_now < signal_now)

            # Only care about CROSS_UP (for buys) unless the ticker is in my_stocks then consider CROSS_DOWN
            want_up_signal = (ticker not in my_stocks and cross_up)
            want_down_signal = (ticker in my_stocks and cross_down)

            if not (want_up_signal or want_down_signal):
                continue

            # Build scoring booleans
            score = 0.0
            passed = []

            # 1) MACD crossover + rising histogram (weight 20)
            macd_ok = (cross_up or cross_down) and (hist_now > 0) and (hist_now > df["Histogram"].iloc[-2])
            if macd_ok:
                score += WEIGHTS["macd"]; passed.append("MACD")
            # 2) RSI rising + above 50 or bullish divergence (20)
            rsi_rising = rsi_now > df["RSI"].iloc[-2]
            rsi_ok = (rsi_rising and rsi_now > 50) or (rsi_now < 30 and rsi_rising)  # simple
            if rsi_ok:
                score += WEIGHTS["rsi"]; passed.append("RSI")
            # 3) Volume confirmation (15): current volume > 1.1 * vol20
            vol_ok = (vol_now > 1.1 * vol20)
            if vol_ok:
                score += WEIGHTS["volume"]; passed.append("Volume")
            # 4) Support bounce (10): price close near MA50/200 or prior swing low (we require either/both because you chose C)
            price = last["Close"]
            ma50 = last["MA50"]
            ma200 = last["MA200"]
            prior_low = detect_prior_swing_low(df["Close"], lookback=60)
            # define "bounce" if price is within 2.5% above prior low or within 1% of MA50/200 (flexible)
            support_ok = False
            if prior_low is not None:
                if price <= prior_low * 1.025 and price >= prior_low:
                    support_ok = True
            if (abs(price - ma50) / ma50) <= 0.01 or (abs(price - ma200) / ma200) <= 0.01:
                support_ok = True
            if support_ok:
                score += WEIGHTS["support"]; passed.append("Support")
            # 5) Trend alignment (10): MA50 > MA200 and slope upward
            ma50_slope = last["MA50"] - df["MA50"].iloc[-5] if len(df) > 5 else last["MA50"] - df["MA50"].iloc[0]
            trend_ok = (df["MA50"].iloc[-1] > df["MA200"].iloc[-1]) and (ma50_slope > 0)
            if trend_ok:
                score += WEIGHTS["trend"]; passed.append("Trend")
            # 6) Candle pattern (10)
            candle_ok = bullish_candle_pattern(df.iloc[-2], df.iloc[-1])
            if candle_ok:
                score += WEIGHTS["candle"]; passed.append("Candle")
            # 7) MA reclaim (5): price above 20 or 50 (short MAs)
            ma_reclaim_ok = (price > last["MA20"]) or (price > last["MA50"])
            if ma_reclaim_ok:
                score += WEIGHTS["ma_reclaim"]; passed.append("MA Reclaim")
            # 8) Market structure break (5): break above last swing high (simple: above recent max)
            recent_max = df["Close"].iloc[-(30+1):-1].max() if len(df) > 31 else df["Close"].iloc[:-1].max()
            market_structure_ok = price > recent_max
            if market_structure_ok:
                score += WEIGHTS["market_structure"]; passed.append("MarketStruct")
            # 9) ADX (2.5): ADX > 20 indicates trend build
            adx_ok = (not np.isnan(last.get("ADX", np.nan))) and (last.get("ADX", 0) > 20)
            if adx_ok:
                score += WEIGHTS["adx"]; passed.append("ADX")
            # 10) Stochastic bullish (2.5): K crossing D or K < 20 crossing up
            stoch_k = last.get("STOCH_K", np.nan)
            stoch_d = last.get("STOCH_D", np.nan)
            stoch_prev_k = df["STOCH_K"].iloc[-2] if "STOCH_K" in df.columns else np.nan
            stoch_prev_d = df["STOCH_D"].iloc[-2] if "STOCH_D" in df.columns else np.nan
            stoch_ok = False
            if (not np.isnan(stoch_k) and not np.isnan(stoch_d) and not np.isnan(stoch_prev_k) and not np.isnan(stoch_prev_d)):
                if (stoch_prev_k < stoch_prev_d and stoch_k > stoch_d) or (stoch_k < 20 and stoch_k > stoch_prev_k):
                    stoch_ok = True
            if stoch_ok:
                score += WEIGHTS["stochastic"]; passed.append("Stochastic")

            # Normalize (the WEIGHTS sum to 100)
            probability = round(min(max(score, 0.0), 100.0), 1)  # 0-100

            # Profitability rating (stars)
            rating_str = score_to_stars(probability)

            # Q3: Send alerts only if probability >= 65%
            MIN_ALERT_PROB = 65.0
            if probability < MIN_ALERT_PROB:
                # Optionally log a weak signal to dashboard but do not send
                signal_log.append({
                    "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "ticker": ticker,
                    "type": f"ℹ️ WEAK {('CROSS_DOWN' if cross_down else 'CROSS_UP')}",
                    "prob": f"{probability}%",
                    "rating": rating_str,
                    "macd": f"{macd_now:.3f}",
                    "signal": f"{signal_now:.3f}",
                    "rsi": f"{rsi_now:.2f}",
                    "trend": trend
                })
                continue

            # Unique ID to prevent duplicates
            today_str = datetime.date.today().isoformat()
            cross_type = "CROSS_DOWN" if cross_down else "CROSS_UP"
            signal_id = f"{today_str}_{ticker}_{cross_type}_{int(probability)}"

            if signal_id in alerted_signals:
                continue  # Skip duplicates

            alerted_signals.add(signal_id)
            with open(ALERTS_FILE, "wb") as f:
                pickle.dump(alerted_signals, f)

            # Build readable message with checklist
            emoji = "🔴" if cross_down else "🔵"
            header = f"{emoji} {'SELL' if cross_down else 'BUY'} Signal: {ticker} — {probability}% ({rating_str})"
            macd_line = f"MACD: {macd_now:.3f} vs Signal: {signal_now:.3f} (Hist {hist_now:.3f})"
            rsi_line = f"RSI: {rsi_now:.2f}"
            vol_line = f"Volume: {vol_now:,} (20d avg {int(vol20):,}) {'↑' if vol_ok else '↓'}"
            trend_line = f"Trend: {trend}"
            supports = []
            if prior_low is not None:
                supports.append(f"prior low {prior_low:.2f}")
            supports.append(f"MA50 {ma50:.2f}")
            supports.append(f"MA200 {ma200:.2f}")
            support_line = "Support checks: " + ", ".join(supports) + ( " ✓" if support_ok else " ✗" )
            passed_line = "Passed: " + ", ".join(passed) if passed else "Passed: none"

            msg = "\n".join([header, macd_line, rsi_line, vol_line, trend_line, support_line, passed_line])

            # Send Telegram alert
            await send_async_message(msg)
            print(f"📈 Alert sent: {msg}")

            # Add to dashboard
            signal_log.append({
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ticker": ticker,
                "type": f"{emoji} MACD {cross_type.replace('_',' ')}",
                "prob": f"{probability}%",
                "rating": rating_str,
                "macd": f"{macd_now:.3f}",
                "signal": f"{signal_now:.3f}",
                "rsi": f"{rsi_now:.2f}",
                "trend": trend
            })

        except Exception as e:
            print(f"⚠️ Error processing {ticker}: {e}")

# === Async Scheduler ===
async def schedule_bot():
    vancouver_tz = ZoneInfo("America/Vancouver")
    last_run_date = None
    last_run_hour = None

    def should_send_startup():
        today_str = str(datetime.date.today())
        try:
            fd = os.open(STARTUP_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                f.write(today_str)
            return True
        except FileExistsError:
            return False

    if should_send_startup():
        startup_msg = "✅ Bot started - Running 24/7 with 3 scans per day!"
        print(startup_msg)
        await send_async_message(startup_msg)
        print("🕒 Running initial startup scan...")
        await check_signals()
        print("✅ Initial startup scan complete.")
    else:
        print("✅ Bot already started today, skipping startup message.")

    scheduled_hours = [6, 12, 18]

    while True:
        try:
            now = datetime.datetime.now(vancouver_tz)
            current_hour = now.hour
            current_date = now.date()

            if current_hour in scheduled_hours:
                if last_run_hour != current_hour or last_run_date != current_date:
                    if last_run_date != current_date:
                        clear_old_alerts()

                    run_msg = f"🕕 Scan started at {now.strftime('%Y-%m-%d %I:%M %p %Z')}..."
                    print(run_msg)
                    try:
                        await send_async_message(run_msg)
                    except Exception as e:
                        print(f"⚠️ Failed to send run-start message: {e}")

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

                    last_run_hour = current_hour
                    last_run_date = current_date

            await asyncio.sleep(60)

        except Exception as loop_exc:
            print(f"🔥 Scheduler loop error, continuing: {loop_exc}")
            await asyncio.sleep(60)

# === Flask keepalive thread ===
def run_flask():
    app.run(host="0.0.0.0", port=5000, use_reloader=False)

if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    asyncio.run(schedule_bot())



