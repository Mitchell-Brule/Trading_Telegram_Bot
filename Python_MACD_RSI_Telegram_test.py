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
import sys
import json
import gspread
from google.oauth2.service_account import Credentials

nest_asyncio.apply()  # allows nested event loops

STARTUP_FILE = "startup_sent.txt"
ALERTS_FILE = "alerted_signals.pkl"
LEADER_LOCK = "bot_leader.lock"   # prevents duplicate startup messages from concurrent processes

# === Telegram setup ===
bot_token = "7481105387:AAHsNaOFEuMuWan2E1Y44VMrWeiZcxBjCAw"
chat_id = 7602575312

def update_google_sheet(data_row):
    try:
        import base64

        encoded = os.environ.get("GOOGLE_CREDENTIALS_BASE64")
        if not encoded:
            raise ValueError("GOOGLE_CREDENTIALS_BASE64 not set")

        decoded = base64.b64decode(encoded)
        creds_dict = json.loads(decoded)

        scope = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]

        creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
        client = gspread.authorize(creds)

        sheet_url = "https://docs.google.com/spreadsheets/d/1BEwe7YaudsSCNxaYsd12ZrM2eTbLqCJdsJFvAKw43yU/edit#gid=0"
        sheet = client.open_by_url(sheet_url).sheet1

        row = [
            data_row.get('Date', ''),
            data_row.get('Ticker', ''),
            data_row.get('Buy_Price', ''),
            data_row.get('Trailing_Exit', ''), # Updated from Target_Price
            data_row.get('Horizon', ''),
            data_row.get('Prob', '')
        ]

        sheet.append_row(row)
        print(f"SUCCESS: Wrote {data_row['Ticker']} to Google Sheets.")

    except Exception as e:
        print(f"Sheets Error: {e}")

bot = Bot(
    token=bot_token,
    request=HTTPXRequest(connect_timeout=10, read_timeout=20, connection_pool_size=10)
)

# === Helper: leader election so only one process announces startup ===
def claim_leadership():
    try:
        fd = os.open(LEADER_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            f.write(f"{os.getpid()}\n{datetime.datetime.utcnow().isoformat()}\n")
        return True
    except FileExistsError:
        return False

def release_leadership():
    try:
        if os.path.exists(LEADER_LOCK):
            os.remove(LEADER_LOCK)
    except Exception:
        pass

# === Async Telegram sender ===
async def send_async_message(text):
    for attempt in range(3):
        try:
            # ADDED HTML PARSE MODE FOR CLEAN FORMATTING
            await bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML")
            print(f"📩 Telegram alert sent...")
            return
        except NetworkError as e:
            print(f"🌐 Telegram network error (try {attempt+1}/3): {e}")
            await asyncio.sleep(2)
        except Exception as e:
            print(f"⚠️ Telegram send error: {e}")
            return

def send_telegram_message(text):
    try:
        loop = asyncio.get_running_loop()
        asyncio.create_task(send_async_message(text))
    except RuntimeError:
        asyncio.run(send_async_message(text))

# === Flask dashboard ===
app = Flask(__name__)
signal_log = []

@app.route("/")
def index():
    html = """
    <html>
        <head>
            <title>Trading Bot Signals</title>
            <meta http-equiv="refresh" content="30">
            <style>
                body { font-family: Arial; background-color: #111; color: #eee; text-align: center; }
                h1 { color: #4CAF50; }
                table { margin: auto; border-collapse: collapse; width: 95%; }
                td, th { border: 1px solid #444; padding: 6px; font-size: 13px; }
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
                    <th>Time</th><th>Ticker</th><th>Signal</th><th>Probability</th><th>Rating</th><th>MACD</th><th>SignalLine</th><th>RSI</th><th>Horizon</th><th>Trend</th>
                </tr>
                {% for s in signals %}
                    <tr>
                        <td>{{ s['time'] }}</td>
                        <td>{{ s['ticker'] }}</td>
                        <td class="{{ 'up' if 'BUY' in s['signal'] else 'down' }}">{{ s['signal'] }}</td>
                        <td>{{ s['prob'] }}</td>
                        <td>{{ s['rating'] }}</td>
                        <td>{{ s['macd'] }}</td>
                        <td>{{ s['signal_line'] }}</td>
                        <td>{{ s['rsi'] }}</td>
                        <td>{{ s['horizon'] }}</td>
                        <td>{{ s['trend'] }}</td>
                    </tr>
                {% endfor %}
            </table>
        </body>
    </html>
    """
    return render_template_string(
        html,
        signals=signal_log[-200:],
        last_update=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

# === Stock list (Added Crypto for weekend/high-beta momentum) ===
tickers = [
    "ARB-USD", "BTC-USD", "ETH-USD", "PLTR", # High volatility additions
    "ABBV","ABNB","ABT","ADBE","ADI","ADP","ADSK","ADM","AAPL","ALGN","AMAT","AMGN","AMD","AMT","AMZN","ANET","APD","ASML","AVB","AVGO","AXP",
    "BA","BAC","BDX","BIIB","BKNG","BLK","BMY","BRK-B","BX","CAT","CDNS","CDW","CHRW","CHTR","CI","CL","CLX","CMCSA","CMG","COF","COP","COST","CPB",
    "CRM","CRWD","CSCO","CTAS","CVX","DE","DELL","DG","DHR","DIS","DOCU","DUK","DVN","DDOG","EMR","ENPH","EOG","EQIX","ETN","EW","EXPE",
    "FAST","FDX","FIS","FTNT","GE","GILD","GIS","GOOG","GOOGL","GS","HCA","HD","HON","HPE","IBM","ILMN","INTC","INTU","ISRG","JCI","JPM",
    "KHC","KIM","KKR","KLAC","KMB","KMI","KO","LAMR","LIN","LLY","LMT","LOW","LRCX","MA","MAR","MCD","MCHP","MDB","MDT","META","MMC","MNST",
    "MO","MRK","MS","MSFT","MSI","MU","NFLX","NKE","NOC","NOW","NTES","NVDA","NXPI","OKTA","ORCL","ORLY","PANW","PAYC","PEP","PFE","PG","PGR",
    "PLD","PM","PSX","PYPL","QCOM","REGN","RMD","ROST","RTX","SBUX","SHOP","SLB","SNOW","SNPS","SO","SPGI","SPOT","SYK","TEAM","T","TMO","TJX",
    "TMUS","TSLA","TSM","TXN","UNH","UNP","UPS","V","VZ","WMT","XOM","ZM","ZS"
]

if os.path.exists(ALERTS_FILE):
    with open(ALERTS_FILE, "rb") as f:
        alerted_signals = pickle.load(f)
else:
    alerted_signals = set()

nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

def clear_old_alerts():
    global alerted_signals
    today = datetime.date.today().isoformat()
    alerted_signals = {a for a in alerted_signals if a.startswith(today)}
    with open(ALERTS_FILE, "wb") as f:
        pickle.dump(alerted_signals, f)

def detect_prior_swing_low(series_close, lookback=60):
    if len(series_close) < lookback + 3:
        return None
    sub = series_close[-(lookback+3):-3]
    min_idx = np.argmin(sub)
    min_val = sub.iloc[min_idx]
    return float(min_val)

def bullish_candle_pattern(row_prev, row):
    engulfing = (row_prev['Close'] < row_prev['Open']) and (row['Close'] > row['Open']) and (row['Close'] > row_prev['Open']) and (row['Open'] < row_prev['Close'])
    body = abs(row['Close'] - row['Open'])
    lower_wick = min(row['Open'], row['Close']) - row['Low']
    upper_wick = row['High'] - max(row['Open'], row['Close'])
    hammer = (body > 0) and (lower_wick > 2 * body) and (upper_wick < body * 0.7)
    return engulfing or hammer

def score_to_stars(score):
    if score >= 90: return "★★★★★"
    if score >= 80: return "★★★★☆"
    if score >= 70: return "★★★☆☆"
    if score >= 65: return "★★☆☆☆"
    return "★☆☆☆☆"

# Scoring weights
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

# === Core engine: MACD + SQUEEZE TRIGGER ===
async def check_signals():
    global alerted_signals
    clear_old_alerts()
    print("🔍 Checking 4H Pump setups...")

    try:
        # CHANGED INTERVAL TO 4H FOR FASTER BUT RELIABLE MOMENTUM
        data_dict = yf.download(
            tickers,
            period="1mo",
            interval="4h", 
            group_by="ticker",
            auto_adjust=False,
            progress=False
        )
    except Exception as e:
        print(f"⚠️ Error downloading data: {e}")
        return

    if isinstance(data_dict.columns, pd.MultiIndex):
        data_dict = {ticker: data_dict[ticker].dropna() for ticker in tickers if ticker in data_dict}

    for ticker in tickers:
        try:
            if ticker not in data_dict:
                continue

            df = data_dict[ticker].copy()
            if len(df) < 60:
                continue

            macd_indicator = ta.trend.MACD(df["Close"])
            df["MACD"] = macd_indicator.macd()
            df["Signal"] = macd_indicator.macd_signal()
            df["Histogram"] = df["MACD"] - df["Signal"]
            df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
            df["MA20"] = df["Close"].rolling(window=20, min_periods=1).mean()
            df["MA50"] = df["Close"].rolling(window=50, min_periods=1).mean()
            df["MA200"] = df["Close"].rolling(window=200, min_periods=1).mean()
            df["Vol20"] = df["Volume"].rolling(window=20, min_periods=1).mean()

            try:
                adx = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=14)
                df["ADX"] = adx.adx()
            except Exception:
                df["ADX"] = np.nan

            try:
                stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
                df["STOCH_K"] = stoch.stoch()
                df["STOCH_D"] = stoch.stoch_signal()
            except Exception:
                df["STOCH_K"] = np.nan
                df["STOCH_D"] = np.nan

            df = df.dropna().copy()
            if len(df) < 50:
                continue

            last = df.iloc[-1]
            prev = df.iloc[-2]
            macd_prev, macd_now = df["MACD"].iloc[-2], df["MACD"].iloc[-1]
            signal_prev, signal_now = df["Signal"].iloc[-2], df["Signal"].iloc[-1]
            hist_now = df["Histogram"].iloc[-1]
            rsi_now = df["RSI"].iloc[-1]
            vol_now = df["Volume"].iloc[-1]
            vol20 = df["Vol20"].iloc[-1]
            adx_now = last.get("ADX", np.nan)
            trend = "Uptrend" if df["MA50"].iloc[-1] > df["MA200"].iloc[-1] else "Downtrend"
            price = last["Close"]
            ma50 = last["MA50"]
            ma200 = last["MA200"]
            prior_low = detect_prior_swing_low(df["Close"], lookback=60)

            # === PUMP CATCHER LOGIC ===
            cross_up = (macd_prev < signal_prev) and (macd_now > signal_now)
            
            # 1. MA Squeeze (Are the MA20 and MA50 within 2% of each other?)
            ma_spread = abs(last["MA20"] - last["MA50"]) / last["MA50"]
            is_squeezed = ma_spread <= 0.02
            
            # 2. RSI Momentum Breakout (Instead of oversold, we want institutional buying)
            rsi_breakout = (rsi_now >= 60) and (rsi_now > df["RSI"].iloc[-2])
            
            # 3. Price confirmation (Did price just cross above the squeeze?)
            price_breakout = price > last["MA20"] and price > last["MA50"]

            # We ONLY want alerts if there's a squeeze AND a breakout, or standard MACD cross
            if not cross_up:
                continue

            if not (is_squeezed and rsi_breakout and price_breakout):
               continue # Skip standard boring crosses, only alert the explosive ones
            
            squeeze_text = "🔥 MA Squeeze Detected (High Vol. Expected)" if is_squeezed else "Standard Trend"

            # Scoring
            score = 30.0 # Huge base score because it passed the strict squeeze filter
            passed = ["Squeeze Breakout"]

            macd_ok = cross_up and (hist_now > 0) and (hist_now > df["Histogram"].iloc[-2])
            if macd_ok: score += WEIGHTS["macd"]; passed.append("MACD")

            rsi_rising = rsi_now > df["RSI"].iloc[-2]
            if rsi_breakout: score += WEIGHTS["rsi"]; passed.append("RSI Momentum")

            vol_ok = (vol_now > 1.1 * vol20)
            if vol_ok: score += WEIGHTS["volume"]; passed.append("Volume")

            support_ok = False
            if prior_low is not None:
                if price <= prior_low * 1.025 and price >= prior_low: support_ok = True
            if (abs(price - ma50) / ma50) <= 0.01 or (abs(price - ma200) / ma200) <= 0.01: support_ok = True
            if support_ok: score += WEIGHTS["support"]; passed.append("Support")

            ma50_slope = last["MA50"] - df["MA50"].iloc[-5] if len(df) > 5 else last["MA50"] - df["MA50"].iloc[0]
            trend_ok = (df["MA50"].iloc[-1] > df["MA200"].iloc[-1]) and (ma50_slope > 0)
            if trend_ok: score += WEIGHTS["trend"]; passed.append("Trend")

            candle_ok = bullish_candle_pattern(df.iloc[-2], df.iloc[-1])
            if candle_ok: score += WEIGHTS["candle"]; passed.append("Candle")

            ma_reclaim_ok = (price > last["MA20"]) or (price > last["MA50"])
            if ma_reclaim_ok: score += WEIGHTS["ma_reclaim"]; passed.append("MA Reclaim")

            recent_max = df["Close"].iloc[-(30+1):-1].max() if len(df) > 31 else df["Close"].iloc[:-1].max()
            market_structure_ok = price > recent_max
            if market_structure_ok: score += WEIGHTS["market_structure"]; passed.append("MarketStruct")

            adx_ok = (not np.isnan(last.get("ADX", np.nan))) and (last.get("ADX", 0) > 20)
            if adx_ok: score += WEIGHTS["adx"]; passed.append("ADX")

            stoch_k, stoch_d = last.get("STOCH_K", np.nan), last.get("STOCH_D", np.nan)
            stoch_prev_k = df["STOCH_K"].iloc[-2] if "STOCH_K" in df.columns else np.nan
            stoch_prev_d = df["STOCH_D"].iloc[-2] if "STOCH_D" in df.columns else np.nan
            stoch_ok = False
            if (not np.isnan(stoch_k) and not np.isnan(stoch_d) and not np.isnan(stoch_prev_k) and not np.isnan(stoch_prev_d)):
                if (stoch_prev_k < stoch_prev_d and stoch_k > stoch_d) or (stoch_k < 20 and stoch_k > stoch_prev_k):
                    stoch_ok = True
            if stoch_ok: score += WEIGHTS["stochastic"]; passed.append("Stochastic")

            probability = round(min(max(score, 0.0), 100.0), 1)
            rating_str = score_to_stars(probability)

            MIN_ALERT_PROB = 65.0
            if probability < MIN_ALERT_PROB:
                continue

            cross_date = df.index[-1].strftime("%Y-%m-%d-%H")
            signal_id = f"{cross_date}_{ticker}_PUMP_{int(probability)}"

            if signal_id in alerted_signals:
                continue

            alerted_signals.add(signal_id)
            with open(ALERTS_FILE, "wb") as f:
                pickle.dump(alerted_signals, f)

            # === CLEAN HTML TELEGRAM MESSAGE ===
            msg = f"""
<b>🚀 PUMP CATCHER: {ticker}</b>
━━━━━━━━━━━━━━━━━━━━
<b>Setup:</b> {squeeze_text}
<b>Prob:</b> {probability}% {rating_str}

<b><u>Key Triggers</u></b>
• <b>MACD:</b> {macd_prev:.2f} ➔ {macd_now:.2f}
• <b>RSI Breakout:</b> {rsi_now:.1f} 
• <b>Trend:</b> {trend}

<b><u>Execution Data</u></b>
• <b>Price:</b> <code>${price:.2f}</code>
• <b>Trailing Exit:</b> <code>${last["MA20"]:.2f}</code> <i>(Sell if close below MA20)</i>
• <b>Volume:</b> {vol_now:,} <i>(Avg: {int(vol20):,})</i>
"""
            await send_async_message(msg)
            print(f"📈 Alert sent: {ticker} | {probability}%")

            signal_log.append({
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ticker": ticker,
                "signal": "BUY",
                "prob": f"{probability}%",
                "rating": rating_str,
                "macd": f"{macd_now:.3f}",
                "signal_line": f"{signal_now:.3f}",
                "rsi": f"{rsi_now:.2f}",
                "horizon": "Trailing MA20",
                "trend": trend
            })

            # THE FIX: Dynamic Trailing Exit for Sheets
            log_payload = {
                'Date': datetime.datetime.now().strftime("%Y-%m-%d"),
                'Ticker': ticker,
                'Buy_Price': round(float(price), 2),
                'Trailing_Exit': f"Sell if close < ${round(last['MA20'], 2)}",
                'Horizon': "Hold until MA20 breaks",
                'Prob': f"{probability}%"
            }
            update_google_sheet(log_payload)

        except Exception as e:
            print(f"⚠️ Error processing {ticker}: {e}")

# === Scheduler with single-startup announcement ===
async def schedule_bot():
    vancouver_tz = ZoneInfo("America/Vancouver")
    last_run_date = None
    last_run_hour = None

    leader = claim_leadership()

    def should_send_startup():
        today_str = str(datetime.date.today())
        if not leader: return False
        try:
            fd = os.open(STARTUP_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                f.write(today_str)
            return True
        except FileExistsError:
            return False

    if should_send_startup():
        startup_msg = "✅ <b>Bot started</b> - Hunting 4H Pump Setups!"
        print(startup_msg)
        await send_async_message(startup_msg)
        print("🕒 Running initial startup scan...")
        await check_signals()
        print("✅ Initial startup scan complete.")

    # Modified to run every 4 hours to match the new interval
    scheduled_hours = [2, 6, 10, 14, 18, 22] 

    try:
        while True:
            try:
                now = datetime.datetime.now(vancouver_tz)
                current_hour = now.hour
                current_date = now.date()

                if current_hour in scheduled_hours:
                    if last_run_hour != current_hour or last_run_date != current_date:
                        if last_run_date != current_date:
                            clear_old_alerts()

                        print(f"🕕 4H Scan started at {now.strftime('%Y-%m-%d %I:%M %p %Z')}...")
                        try:
                            await check_signals()
                        except Exception as e:
                            print(f"⚠️ Error in check_signals(): {e}")

                        last_run_hour = current_hour
                        last_run_date = current_date

                await asyncio.sleep(60)

            except Exception as loop_exc:
                print(f"🔥 Scheduler loop error, continuing: {loop_exc}")
                await asyncio.sleep(60)
    finally:
        if leader:
            release_leadership()

# === Flask keepalive thread ===
def run_flask():
    app.run(host="0.0.0.0", port=5000, use_reloader=False)

if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    try:
        asyncio.run(schedule_bot())
    except KeyboardInterrupt:
        print("Exiting...")
        try:
            release_leadership()
        except: pass
        sys.exit(0)

        
























