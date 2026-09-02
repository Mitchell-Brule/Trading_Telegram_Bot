import yfinance as yf
import ta
import pickle
import os
import asyncio
import time
import datetime
from zoneinfo import ZoneInfo
from telegram import Bot
from telegram.request import HTTPXRequest
from telegram.error import NetworkError
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

# Windows consoles/redirected files default stdout to cp1252, which can't encode
# the emoji used in print()/log messages below and crashes the whole process
# before it even sends the startup Telegram message. Force UTF-8. Also force line
# buffering - Python fully block-buffers stdout when it isn't a terminal (e.g. a
# hosting platform's log capture), which can delay prints by minutes.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

STARTUP_FILE = "startup_sent.txt"
ALERTS_FILE = "alerted_signals.pkl"
POSITIONS_FILE = "open_positions.pkl"   # tickers we've told the user to buy, tracked until sold
LEADER_LOCK = "bot_leader.lock"   # prevents duplicate startup messages from concurrent processes
MAX_OPEN_POSITIONS = int(os.environ.get("MAX_OPEN_POSITIONS", "15"))  # cap new buy alerts once this many are open
MIN_HOLD_DAYS = 5  # ignore MACD bearish-cross sells before this - see stop-loss note in check_signals
# backtested 2026-08-31 on 20mo/544-ticker universe: 5-day min hold nearly doubles avg
# per-trade return (+1.96%->+3.82%) vs the old 2-day setting, same win rate, positive
# core+satellite alpha over SPY in both halves of the test window (+13.5pt / +11.8pt)
HORIZON_DAYS = 10  # soft "review by" window - backtest showed avg hold ~3.9d and horizon>=7d makes no
                   # difference to outcomes (trades resolve via MACD-cross/stop-loss first), so this is
                   # just a sane buffer past the typical hold, not a tuned exit parameter (2026-08-31)

# === Telegram setup ===
_missing_env = [v for v in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID") if not os.environ.get(v)]
if _missing_env:
    sys.exit(f"❌ Missing required environment variable(s): {', '.join(_missing_env)}. Check your .env file.")

bot_token = os.environ["TELEGRAM_BOT_TOKEN"]
# Buy and sell alerts go to separate chats so they don't get mixed together.
# Set TELEGRAM_CHAT_ID_SELL to a second chat/group's id to split them; until then,
# sell alerts fall back to the same chat as buy alerts.
BUY_CHAT_ID = int(os.environ["TELEGRAM_CHAT_ID"])
SELL_CHAT_ID = int(os.environ.get("TELEGRAM_CHAT_ID_SELL", os.environ["TELEGRAM_CHAT_ID"]))

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
STALE_LOCK_MINUTES = 30  # a crashed process can never clean up its own lock

def claim_leadership():
    if os.path.exists(LEADER_LOCK):
        age_minutes = (time.time() - os.path.getmtime(LEADER_LOCK)) / 60
        if age_minutes > STALE_LOCK_MINUTES:
            print(f"⚠️ Removing stale leader lock ({age_minutes:.0f} min old - previous run likely crashed)")
            try:
                os.remove(LEADER_LOCK)
            except Exception:
                pass
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
async def send_async_message(text, chat_id=None):
    target_chat_id = chat_id if chat_id is not None else BUY_CHAT_ID
    for attempt in range(3):
        try:
            # ADDED HTML PARSE MODE FOR CLEAN FORMATTING
            await bot.send_message(chat_id=target_chat_id, text=text, parse_mode="HTML")
            print(f"📩 Telegram alert sent...")
            return
        except NetworkError as e:
            print(f"🌐 Telegram network error (try {attempt+1}/3): {e}")
            await asyncio.sleep(2)
        except Exception as e:
            print(f"⚠️ Telegram send error: {e}")
            return

# === Flask dashboard ===
app = Flask(__name__)
SIGNAL_LOG_FILE = "signal_log.pkl"
if os.path.exists(SIGNAL_LOG_FILE):
    with open(SIGNAL_LOG_FILE, "rb") as f:
        signal_log = pickle.load(f)
else:
    signal_log = []

BOT_START_TIME = datetime.datetime.now()
last_scan_time = None
last_scan_error = None

def save_signal_log():
    with open(SIGNAL_LOG_FILE, "wb") as f:
        pickle.dump(signal_log[-500:], f)

@app.route("/")
def index():
    html = """
    <html>
        <head>
            <title>Trading Bot Signals</title>
            <meta http-equiv="refresh" content="30">
            <style>
                body { font-family: Arial; background-color: #111; color: #eee; text-align: center; }
                h1, h2 { color: #4CAF50; }
                table { margin: auto; border-collapse: collapse; width: 95%; margin-bottom: 30px; }
                td, th { border: 1px solid #444; padding: 6px; font-size: 13px; }
                tr:nth-child(even) { background-color: #222; }
                .up { color: #4CAF50; }
                .down { color: #F44336; }
            </style>
        </head>
        <body>
            <h1>📈 Trading Bot Signals</h1>
            <p>Last updated: {{ last_update }} | Last scan: {{ last_scan }}</p>

            <h2>Open Positions ({{ positions|length }})</h2>
            <table>
                <tr>
                    <th>Ticker</th><th>Buy Price</th><th>Buy Date</th><th>Stop Loss</th><th>Review By</th><th>Setup</th>
                </tr>
                {% for ticker, pos in positions.items() %}
                    <tr>
                        <td>{{ ticker }}</td>
                        <td>${{ pos['buy_price'] }}</td>
                        <td>{{ pos['buy_date'] }}</td>
                        <td>${{ pos['stop_loss'] }}</td>
                        <td>{{ pos['review_by'] }}</td>
                        <td>{{ pos['setup'] }}</td>
                    </tr>
                {% endfor %}
            </table>

            <h2>Signal History</h2>
            <table>
                <tr>
                    <th>Time</th><th>Ticker</th><th>Signal</th><th>Score</th><th>Rating</th><th>MACD</th><th>SignalLine</th><th>RSI</th><th>Horizon</th><th>Trend</th>
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
        signals=list(reversed(signal_log[-200:])),
        positions=open_positions,
        last_update=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        last_scan=last_scan_time.strftime("%Y-%m-%d %H:%M:%S") if last_scan_time else "never yet"
    )

@app.route("/health")
def health():
    return {
        "status": "ok" if last_scan_error is None else "degraded",
        "started_at": BOT_START_TIME.isoformat(),
        "last_scan_at": last_scan_time.isoformat() if last_scan_time else None,
        "last_scan_error": last_scan_error,
        "open_positions": len(open_positions),
        "tickers_watched": len(tickers),
    }

# === Stock list: full S&P 500 (2026-08-31 build-out) ===
# Crypto was tested and removed the same day - every threshold combo came out
# net-negative for crypto under this strategy (see LOCK-tier comment below).
tickers = [
    "A", "AA", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACN", "ADBE", "ADI",
    "ADM", "ADP", "ADSK", "AEE", "AEP", "AES", "AFL", "AFRM", "AIG", "AIZ",
    "AJG", "AKAM", "ALB", "ALGN", "ALL", "ALLE", "AMAT", "AMCR", "AMD", "AME",
    "AMGN", "AMP", "AMT", "AMZN", "ANET", "AON", "AOS", "APA", "APD", "APH",
    "APO", "APP", "APTV", "ARE", "ARES", "ARM", "ASML", "ATO", "AVGO", "AVY",
    "AWK", "AXON", "AXP", "AZO", "BA", "BABA", "BAC", "BALL", "BAX", "BBY",
    "BDX", "BEN", "BF-B", "BG", "BIIB", "BKNG", "BKR", "BLDR", "BLK", "BMY",
    "BNTX", "BNY", "BR", "BRK-B", "BRO", "BSX", "BX", "BXP", "C", "CAH",
    "CARR", "CASY", "CAT", "CB", "CBOE", "CBRE", "CCI", "CCL", "CDNS", "CDW",
    "CEG", "CF", "CFG", "CHD", "CHRW", "CHTR", "CI", "CIEN", "CINF", "CL",
    "CLX", "CMCSA", "CME", "CMG", "CMI", "CMS", "CNC", "CNP", "COF", "COHR",
    "COIN", "COO", "COP", "COR", "COST", "CPAY", "CPB", "CPRT", "CPT", "CRH",
    "CRL", "CRM", "CRWD", "CSCO", "CSGP", "CSX", "CTAS", "CTSH", "CTVA", "CVNA",
    "CVS", "CVX", "D", "DAL", "DASH", "DD", "DDOG", "DE", "DECK", "DELL",
    "DG", "DGX", "DHI", "DHR", "DIS", "DKNG", "DLR", "DLTR", "DOC", "DOCU",
    "DOV", "DOW", "DPZ", "DRI", "DTE", "DUK", "DVA", "DVN", "DXCM", "EBAY",
    "ECHO", "ECL", "ED", "EFX", "EG", "EIX", "EL", "ELV", "EME", "EMR",
    "ENPH", "EOG", "EQIX", "EQT", "ERIE", "ES", "ESS", "ETN", "ETR", "ETSY",
    "EVRG", "EW", "EXC", "EXE", "EXPD", "EXPE", "EXR", "F", "FANG", "FAST",
    "FCX", "FDS", "FDX", "FDXF", "FE", "FERG", "FFIV", "FICO", "FIS", "FISV",
    "FITB", "FIX", "FLEX", "FOX", "FOXA", "FRT", "FSLR", "FTNT", "FTV", "GD",
    "GDDY", "GE", "GEHC", "GEN", "GEV", "GILD", "GIS", "GL", "GLW", "GM",
    "GNRC", "GOOG", "GOOGL", "GPC", "GPN", "GRMN", "GS", "GWW", "HAL", "HAS",
    "HBAN", "HCA", "HD", "HIG", "HII", "HLT", "HON", "HONA", "HOOD", "HPE",
    "HPQ", "HRL", "HSIC", "HST", "HSY", "HUBB", "HUM", "HWM", "IBKR", "IBM",
    "ICE", "IDXX", "IEX", "IFF", "ILMN", "INCY", "INTC", "INTU", "INVH", "IP",
    "IQV", "IR", "IRM", "ISRG", "IT", "ITW", "IVZ", "J", "JBHT", "JBL",
    "JCI", "JD", "JKHY", "JNJ", "JPM", "KDP", "KEY", "KEYS", "KHC", "KIM",
    "KKR", "KLAC", "KMB", "KMI", "KO", "KR", "KVUE", "L", "LAMR", "LDOS",
    "LEN", "LH", "LHX", "LII", "LIN", "LITE", "LLY", "LMT", "LNT", "LOW",
    "LRCX", "LULU", "LUV", "LVS", "LYB", "LYV", "MA", "MAA", "MAR", "MARA",
    "MAS", "MCD", "MCHP", "MCK", "MCO", "MDB", "MDLZ", "MDT", "MET", "META",
    "MGM", "MKC", "MLM", "MMM", "MNST", "MO", "MOS", "MPC", "MPWR", "MRK",
    "MRNA", "MRSH", "MRVL", "MS", "MSCI", "MSFT", "MSI", "MSTR", "MTB", "MTD",
    "MU", "NCLH", "NDAQ", "NDSN", "NEE", "NEM", "NET", "NFLX", "NI", "NIO",
    "NKE", "NOC", "NOW", "NRG", "NSC", "NTAP", "NTES", "NTRS", "NUE", "NVDA",
    "NVR", "NWS", "NWSA", "NXPI", "O", "ODFL", "OKE", "OKTA", "OMC", "ON",
    "ORCL", "ORLY", "OTIS", "OXY", "PANW", "PAYC", "PAYX", "PCAR", "PCG", "PDD",
    "PEG", "PEP", "PFE", "PFG", "PG", "PGR", "PH", "PHM", "PINS", "PKG",
    "PLD", "PLTR", "PM", "PNC", "PNR", "PNW", "PODD", "PPG", "PPL", "PRU",
    "PSA", "PSKY", "PSX", "PTC", "PWR", "PYPL", "Q", "QCOM", "RBLX", "RCL",
    "RDDT", "REG", "REGN", "RF", "RIOT", "RIVN", "RJF", "RL", "RMD", "ROK",
    "ROKU", "ROL", "ROP", "ROST", "RSG", "RTX", "RVTY", "SBAC", "SBUX", "SCHW",
    "SHOP", "SHW", "SJM", "SLB", "SMCI", "SNA", "SNAP", "SNDK", "SNOW", "SNPS",
    "SO", "SOFI", "SOLV", "SPG", "SPGI", "SPOT", "SRE", "STE", "STLD", "STT",
    "STX", "STZ", "SW", "SWK", "SWKS", "SYF", "SYK", "SYY", "T", "TAP",
    "TDG", "TDY", "TEAM", "TECH", "TEL", "TER", "TFC", "TGT", "TJX", "TKO",
    "TMO", "TMUS", "TPL", "TPR", "TRGP", "TRMB", "TROW", "TRV", "TSCO", "TSLA",
    "TSM", "TSN", "TT", "TTD", "TTWO", "TWLO", "TXN", "TXT", "TYL", "U",
    "UAL", "UBER", "UDR", "UHS", "ULTA", "UNH", "UNP", "UPS", "UPST", "URI",
    "USB", "V", "VEEV", "VICI", "VLO", "VLTO", "VMC", "VMRK", "VRSK", "VRSN",
    "VRT", "VRTX", "VST", "VTR", "VTRS", "VZ", "W", "WAB", "WAT", "WBD",
    "WDAY", "WDC", "WEC", "WELL", "WFC", "WM", "WMB", "WMT", "WRB", "WSM",
    "WST", "WTW", "WY", "WYNN", "XEL", "XOM", "XYL", "XYZ", "YUM", "ZBH",
    "ZBRA", "ZM", "ZS", "ZTS"
]

if os.path.exists(ALERTS_FILE):
    with open(ALERTS_FILE, "rb") as f:
        alerted_signals = pickle.load(f)
else:
    alerted_signals = set()

if os.path.exists(POSITIONS_FILE):
    with open(POSITIONS_FILE, "rb") as f:
        open_positions = pickle.load(f)
else:
    open_positions = {}

def save_positions():
    with open(POSITIONS_FILE, "wb") as f:
        pickle.dump(open_positions, f)

# === Multi-timeframe confirmation: does the daily chart agree with the 4h signal? ===
# A 4h MACD/RSI reversal that fires while the stock is still below its daily
# MA50/MA200 is much more likely to be noise than one backed by an actual daily
# uptrend. One bulk daily-bar download per scan (chunked, like the main fetch),
# not one call per ticker - this is the main new "complementary strategy" added
# 2026-08-30 to make the signal rare and high-conviction instead of just rare.
def _daily_trend_chunk_sync(chunk):
    for attempt in range(2):
        try:
            return yf.download(chunk, period="1y", interval="1d", group_by="ticker", auto_adjust=False, progress=False)
        except Exception as e:
            print(f"⚠️ Daily-trend chunk download failed (attempt {attempt+1}/2): {e}")
            if attempt == 0:
                time.sleep(5)
    return None

async def fetch_daily_trend(all_tickers):
    daily_bullish = {}
    chunk_size = 60
    chunks = [all_tickers[i:i + chunk_size] for i in range(0, len(all_tickers), chunk_size)]
    semaphore = asyncio.Semaphore(3)

    async def _fetch(chunk):
        async with semaphore:
            return chunk, await asyncio.to_thread(_daily_trend_chunk_sync, chunk)

    chunk_results = await asyncio.gather(*[_fetch(c) for c in chunks])

    for chunk, d in chunk_results:
        if d is None or d.empty:
            continue
        sub_frames = {}
        if isinstance(d.columns, pd.MultiIndex):
            sub_frames = {t: d[t] for t in chunk if t in d}
        elif len(chunk) == 1:
            sub_frames = {chunk[0]: d}
        for t, sub in sub_frames.items():
            close = sub["Close"].dropna()
            if len(close) < 60:
                continue
            ma20 = close.rolling(20).mean().iloc[-1]
            ma50 = close.rolling(50).mean().iloc[-1]
            ma200 = close.rolling(200, min_periods=50).mean().iloc[-1]
            # Full aligned-uptrend requirement (price>MA20>MA50>MA200) added 2026-08-31:
            # backtest showed this beats the looser price>MA50>MA200 check on every
            # metric (66% win/+1.96% avg vs 62%/+1.78%), split-half robust (60%/72%).
            daily_bullish[t] = bool(close.iloc[-1] > ma20) and bool(ma20 > ma50) and bool(ma50 > ma200)
    return daily_bullish

DAILY_TREND_CACHE_FILE = "daily_trend_cache.pkl"

async def fetch_daily_trend_cached(all_tickers):
    """Daily bars can't have changed since the last scan today - the live bot
    scans every 4h (6x/day) but only actually needs this refreshed once a day.
    Cuts 5 of every 6 daily-bar downloads for free."""
    today = datetime.date.today().isoformat()
    if os.path.exists(DAILY_TREND_CACHE_FILE):
        try:
            with open(DAILY_TREND_CACHE_FILE, "rb") as f:
                cached_date, cached_trend = pickle.load(f)
            if cached_date == today:
                return cached_trend
        except Exception:
            pass
    trend = await fetch_daily_trend(all_tickers)
    with open(DAILY_TREND_CACHE_FILE, "wb") as f:
        pickle.dump((today, trend), f)
    return trend

# === Earnings-date risk check ===
# A swing trade held into an earnings print is a much bigger gamble than the
# same technical setup with no earnings in the way - excluded outright below,
# not just scored down, now that alerts are a rare pass/fail gate.
def check_earnings_risk(ticker, horizon_days):
    """Returns (next_earnings_date or None, is_within_horizon)."""
    if ticker.endswith("-USD"):  # crypto has no earnings
        return None, False
    try:
        cal = yf.Ticker(ticker).calendar or {}
        dates = cal.get("Earnings Date") or []
        if not dates:
            return None, False
        next_date = min(dates)  # calendar can list a range; take the earliest
        days_until = (next_date - datetime.date.today()).days
        return next_date, (0 <= days_until <= horizon_days)
    except Exception:
        return None, False

# === Market-cap gate ===
# Added 2026-08-31 night: backtested company size against trade outcome on the
# bollinger-only strategy (112 signals, 20mo) and found a large, robust effect -
# mega/large-caps (>=$50B) averaged +7.47%/trade at 76% win vs <$50B's +0.26%/53%.
# Bootstrap 95% CI on the gap [+3.28,+11.62]pts, 100% of 5000 resamples positive -
# the strongest, cleanest signal found in two nights of testing. Split-half stable
# (+7.82% vs +7.16% in each half) and diversified (40 unique tickers). Costs about
# half the signal frequency (1/10.9 days vs 1/5.4) in exchange for roughly doubling
# average win size - a direct hit on Mitchell's explicit "larger wins" ask.
LOCK_MIN_MARKET_CAP_B = 50

def check_market_cap(ticker):
    """Returns market cap in $ (or None if unavailable/crypto)."""
    if ticker.endswith("-USD"):
        return None
    try:
        return yf.Ticker(ticker).fast_info["marketCap"]
    except Exception:
        return None

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

# === LOCK-tier thresholds ===
# Rewritten 2026-08-30 per request: no more weighted score where enough small
# bonuses could add up to an alert - every condition below must hold at once.
#
# 2026-08-30: classic-crossover-only couldn't hit "~1 every 3-4 days" without
# the edge going negative. 2026-08-31: adding the Bollinger squeeze-breakout
# as a second, independent entry (see setup_style below) solved it - the two
# are complementary, not redundant. Then, after expanding the watchlist and
# re-testing at scale, found crypto was dragging the whole strategy down hard
# (every vol/ADX combo tested came out net negative for crypto, e.g. at these
# exact thresholds: 36% win, -2.67% avg - stricter filtering made it WORSE,
# not better, so it's not a tuning gap, it's a real mismatch between this
# equity-shaped strategy and crypto's behavior) - all crypto tickers were
# removed from the watchlist entirely as a result (also caught a genuinely
# corrupted Yahoo data feed for TON-USD along the way: price ranged $0.004-
# $4.11 with 401 implausible >30%-in-4h moves - excluded regardless of the
# performance finding). Tightened again same day: the daily-trend check below
# now requires the fully aligned price>MA20>MA50>MA200 (was just >MA50>MA200) -
# beat the looser version on every metric. FINAL validated numbers, equity-
# only (20mo, ~540/544 tickers loaded): ~129 signals = ~1 every 4.7 days, 66%
# win rate, +1.96% avg / +1.48% median 14d return. Split-half checked (60%/72%
# win, both strong) and not ticker-concentrated (max 4 of 129 from any one
# name, 97 unique tickers contributing). Full sweep table in
# backtest_results.csv - lower MIN_VOL/MIN_ADX for more frequent/weaker, raise
# for rarer/stronger.
LOCK_MIN_VOL_RATIO = 1.2   # volume vs its 20-bar average
LOCK_MIN_ADX = 25          # trend strength (ADXIndicator, 14-period)

# Shown in every buy message as "probability of success" - this is the strategy's
# backtested win rate (20mo, 129 trades, equity-only S&P 500 - see backtest_results.csv),
# NOT a per-signal confidence score. Update this string if a fresh backtest.py run
# changes the validated numbers.
BACKTESTED_WIN_RATE = "66%"

# === Core engine: MACD/RSI crossover + Bollinger breakout, lock-tier only ===
async def check_signals():
    global alerted_signals, last_scan_time, last_scan_error
    clear_old_alerts()
    print("🔍 Checking 4H Pump setups...")

    # Download in chunks so one bad ticker or a transient rate-limit doesn't take
    # the whole scan down with it - each chunk gets one retry before being skipped.
    # Chunks run concurrently (bounded, so we don't hammer Yahoo) instead of one
    # at a time - meaningfully cuts scan wall-clock time as the watchlist grows.
    CHUNK_SIZE = 60
    MAX_CONCURRENT_CHUNKS = 3
    ticker_chunks = [tickers[i:i + CHUNK_SIZE] for i in range(0, len(tickers), CHUNK_SIZE)]
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_CHUNKS)

    def _download_chunk_sync(chunk):
        for attempt in range(2):
            try:
                return yf.download(
                    chunk,
                    period="1mo",
                    interval="4h",
                    group_by="ticker",
                    auto_adjust=False,
                    progress=False
                )
            except Exception as e:
                print(f"⚠️ Error downloading chunk (attempt {attempt+1}/2): {e}")
                if attempt == 0:
                    time.sleep(5)
        return None

    async def _download_chunk(chunk):
        async with semaphore:
            return chunk, await asyncio.to_thread(_download_chunk_sync, chunk)

    chunk_results = await asyncio.gather(*[_download_chunk(c) for c in ticker_chunks])

    data_dict = {}
    chunk_failures = 0
    for chunk, chunk_data in chunk_results:
        if chunk_data is None or chunk_data.empty:
            chunk_failures += 1
            continue
        if isinstance(chunk_data.columns, pd.MultiIndex):
            data_dict.update({t: chunk_data[t].dropna() for t in chunk if t in chunk_data})
        elif len(chunk) == 1:
            data_dict[chunk[0]] = chunk_data.dropna()

    last_scan_time = datetime.datetime.now()
    if not data_dict:
        last_scan_error = "All ticker download chunks failed"
        print(f"⚠️ {last_scan_error}")
        return
    last_scan_error = f"{chunk_failures}/{len(ticker_chunks)} chunks failed" if chunk_failures else None

    # Broad-market (SPY) regime was tested as a hard gate alongside the daily
    # per-ticker trend check below and dropped 2026-08-30: the backtest sweep
    # showed it added no measurable edge once daily_ok was already required,
    # while cutting real signals - a stock in a confirmed daily uptrend doesn't
    # need the whole index to also be up. See backtest_results.csv.
    daily_trend = await fetch_daily_trend_cached(tickers)

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
                df["ATR"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=14).average_true_range()
            except Exception:
                df["ATR"] = np.nan

            try:
                bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
                df["BB_HIGH"] = bb.bollinger_hband()
                df["BB_WIDTH"] = (df["BB_HIGH"] - bb.bollinger_lband()) / df["Close"]
            except Exception:
                df["BB_HIGH"] = np.nan
                df["BB_WIDTH"] = np.nan

            df = df.dropna().copy()
            if len(df) < 50:
                continue

            last = df.iloc[-1]
            macd_prev, macd_now = df["MACD"].iloc[-2], df["MACD"].iloc[-1]
            signal_prev, signal_now = df["Signal"].iloc[-2], df["Signal"].iloc[-1]
            rsi_now = df["RSI"].iloc[-1]
            rsi_prev = df["RSI"].iloc[-2]
            vol_now = df["Volume"].iloc[-1]
            vol20 = df["Vol20"].iloc[-1]
            adx_now = last.get("ADX", np.nan)
            price = last["Close"]
            prior_low = detect_prior_swing_low(df["Close"], lookback=60)

            # === Core trigger: Bollinger squeeze breakout only ===
            # Classic MACD/RSI crossover was removed 2026-08-31 after backtesting
            # showed it was dragging down average win size: over 20mo/544 tickers,
            # bollinger-only signals averaged +4.47%/trade (67% win, bootstrap 95%
            # CI [+2.21%,+7.02%], split-half consistent) vs classic's -0.33%/trade
            # (56% win, n=16, several -8% to -13% losers). Dropping classic barely
            # costs frequency (1/4.7d combined -> 1/5.3d bollinger-only).
            bb_width = df["BB_WIDTH"]
            bb_width_low = bb_width.rolling(100, min_periods=30).min()
            was_squeezed = bb_width.shift(1).iloc[-1] <= bb_width_low.shift(1).iloc[-1] * 1.1
            breakout_now = price > last["BB_HIGH"]
            breakout_prev = df["Close"].iloc[-2] > df["BB_HIGH"].iloc[-2]
            bollinger_setup = bool(was_squeezed and breakout_now and not breakout_prev)

            if not bollinger_setup:
                continue
            setup_style = "bollinger"

            # Already holding this one - don't re-buy and clobber its tracked entry
            # price/trailing stop just because a new cross happened mid-position.
            if ticker in open_positions:
                continue

            # Don't keep piling on new positions past a realistic capital limit.
            if len(open_positions) >= MAX_OPEN_POSITIONS:
                continue

            # === LOCK gate: every complementary condition must agree, no partial credit ===
            vol_ratio = (vol_now / vol20) if vol20 else 0
            volume_ok = vol_ratio >= LOCK_MIN_VOL_RATIO
            trend_strong = (not np.isnan(adx_now)) and (adx_now >= LOCK_MIN_ADX)
            daily_ok = daily_trend.get(ticker, False)
            earnings_date, earnings_in_horizon = check_earnings_risk(ticker, HORIZON_DAYS)
            market_cap = check_market_cap(ticker)
            cap_ok = market_cap is not None and market_cap >= LOCK_MIN_MARKET_CAP_B * 1e9

            if not (volume_ok and trend_strong and daily_ok and not earnings_in_horizon and cap_ok):
                continue

            cross_date = df.index[-1].strftime("%Y-%m-%d-%H")
            signal_id = f"{cross_date}_{ticker}_LOCK"

            if signal_id in alerted_signals:
                continue

            alerted_signals.add(signal_id)
            with open(ALERTS_FILE, "wb") as f:
                pickle.dump(alerted_signals, f)

            # === Stop loss: wider of an ATR-based stop or the last swing low, floored ===
            # backtest.py showed a tighter "closer of the two" stop cut trades short in
            # under 2 days on average. A multi-day swing needs room for normal daily
            # noise, not a 4h-bar-sized leash, so this takes the WIDER candidate,
            # floored at a 12% max risk. See backtest_results.csv.
            atr_now = last.get("ATR", np.nan)
            atr_stop = price - (3.0 * atr_now) if (not np.isnan(atr_now) and atr_now > 0) else price * 0.90
            structure_stop = (prior_low * 0.97) if prior_low is not None else atr_stop
            stop_loss_price = round(max(min(atr_stop, structure_stop), price * 0.88), 2)
            stop_loss_price = min(stop_loss_price, price * 0.99)
            stop_loss_price = max(stop_loss_price, 0.01)
            risk_pct = round((price - stop_loss_price) / price * 100, 1)
            review_by_date = (datetime.date.today() + datetime.timedelta(days=HORIZON_DAYS)).strftime("%Y-%m-%d")
            initial_risk_pct = risk_pct  # locked in at buy time, used to ratchet the trailing stop later

            # Point-form grade per measurement, per request 2026-08-31. Grades are
            # informational (how far past the LOCK threshold, A=strongest) - backtest.py
            # found no correlation between margin-past-threshold and outcome (~0.00),
            # so a higher grade here does NOT mean a better bet, just a stronger reading.
            # Win rate is the strategy's backtested average (20mo/129 trades), not a
            # per-signal confidence score - every LOCK signal shows the same number.
            vol_grade = "A" if vol_ratio >= 2.0 else ("B" if vol_ratio >= 1.5 else "C")
            adx_grade = "A" if adx_now >= 35 else ("B" if adx_now >= 30 else "C")
            msg = (
                f"<b>Trading Bot - BUY {ticker}</b> ${price:.2f}\n"
                f"- Setup: {setup_style}\n"
                f"- Volume: {vol_grade} ({vol_ratio:.1f}x avg)\n"
                f"- Trend (ADX): {adx_grade} ({adx_now:.0f})\n"
                f"- Daily Trend: Aligned\n"
                f"Stop ${stop_loss_price:.2f} - Sell by {review_by_date}\n"
                f"{BACKTESTED_WIN_RATE} historical win rate"
            )
            await send_async_message(msg, chat_id=BUY_CHAT_ID)
            print(f"📈 Lock alert sent: {ticker} @ ${price:.2f}")

            signal_log.append({
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ticker": ticker,
                "signal": "BUY",
                "prob": "LOCK",
                "rating": f"{setup_style}, Vol {vol_ratio:.1f}x, ADX {adx_now:.0f}",
                "macd": f"{macd_now:.3f}",
                "signal_line": f"{signal_now:.3f}",
                "rsi": f"{rsi_now:.2f}",
                "horizon": f"Stop ${stop_loss_price} / Review {review_by_date}",
                "trend": "Uptrend"
            })
            save_signal_log()

            open_positions[ticker] = {
                "buy_price": float(price),
                "buy_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "stop_loss": float(stop_loss_price),
                "high_water_mark": float(price),
                "initial_risk_pct": float(initial_risk_pct),
                "review_by": review_by_date,
                "review_sent": False,
                "setup": setup_style,
            }
            save_positions()

            log_payload = {
                'Date': datetime.datetime.now().strftime("%Y-%m-%d"),
                'Ticker': ticker,
                'Buy_Price': round(float(price), 2),
                'Trailing_Exit': f"Stop-loss ${stop_loss_price} or MACD bearish cross",
                'Horizon': f"Review by {review_by_date}",
                'Prob': "LOCK"
            }
            update_google_sheet(log_payload)

        except Exception as e:
            print(f"⚠️ Error processing {ticker}: {e}")

    await check_sell_signals(data_dict)

# === Exit logic for anything we've told the user to buy ===
# Hard sell (position closed): MACD crosses back below Signal ("blue below orange"), or stop-loss hit.
# Soft sell (position stays open): review-by date reached with no exit signal yet - one reminder only.
async def check_sell_signals(data_dict):
    global open_positions
    if not open_positions:
        return

    for ticker in list(open_positions.keys()):
        pos = open_positions[ticker]
        try:
            df = data_dict.get(ticker)
            if df is None or len(df) < 3:
                fetched = yf.download(ticker, period="1mo", interval="4h", auto_adjust=False, progress=False)
                if isinstance(fetched.columns, pd.MultiIndex):
                    fetched.columns = fetched.columns.get_level_values(0)
                df = fetched.dropna() if not fetched.empty else None
            if df is None or len(df) < 30:
                continue

            df = df.copy()
            macd_indicator = ta.trend.MACD(df["Close"])
            df["MACD"] = macd_indicator.macd()
            df["Signal"] = macd_indicator.macd_signal()
            df = df.dropna()
            if len(df) < 2:
                continue

            price = float(df["Close"].iloc[-1])
            macd_prev, macd_now = df["MACD"].iloc[-2], df["MACD"].iloc[-1]
            signal_prev, signal_now = df["Signal"].iloc[-2], df["Signal"].iloc[-1]
            pnl_pct = round((price - pos["buy_price"]) / pos["buy_price"] * 100, 1)

            # Trailing stop: as the price makes new highs since buy, drag the stop up
            # behind it (same distance as the original risk %) so gains get locked in.
            # Never loosens - only ratchets toward the current price.
            hwm = pos.get("high_water_mark", pos["buy_price"])
            initial_risk_pct = pos.get("initial_risk_pct", round((pos["buy_price"] - pos["stop_loss"]) / pos["buy_price"] * 100, 1))
            if price > hwm:
                pos["high_water_mark"] = price
                pos["initial_risk_pct"] = initial_risk_pct
                trailing_stop = round(price * (1 - initial_risk_pct / 100), 2)
                if trailing_stop > pos["stop_loss"]:
                    pos["stop_loss"] = trailing_stop
                save_positions()

            # A MACD cross right after entry is often just noise on 4h bars - require a
            # minimum hold before honoring it as a real exit signal. Stop-loss is risk
            # protection, so it stays immediate regardless of how long we've held.
            buy_dt = datetime.datetime.strptime(pos["buy_date"], "%Y-%m-%d %H:%M")
            held_days = (datetime.datetime.now() - buy_dt).total_seconds() / 86400
            cross_down = (macd_prev > signal_prev) and (macd_now < signal_now) and (held_days >= MIN_HOLD_DAYS)
            stop_hit = price <= pos["stop_loss"]

            if cross_down or stop_hit:
                reason = "MACD cross" if cross_down else "Stop loss"
                msg = f"<b>Trading Bot - SELL {ticker}</b> now - {reason}\n${price:.2f} (P/L {pnl_pct:+.1f}%)"
                await send_async_message(msg, chat_id=SELL_CHAT_ID)
                print(f"📉 Sell alert sent: {ticker} | {reason} | P/L {pnl_pct:+.1f}%")
                signal_log.append({
                    "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "ticker": ticker,
                    "signal": "SELL",
                    "prob": f"{pnl_pct:+.1f}% P/L",
                    "rating": reason,
                    "macd": f"{macd_now:.3f}",
                    "signal_line": f"{signal_now:.3f}",
                    "rsi": "-",
                    "horizon": "-",
                    "trend": "-"
                })
                save_signal_log()
                del open_positions[ticker]
                save_positions()
                continue

            review_date = datetime.date.fromisoformat(pos["review_by"])
            if (not pos.get("review_sent")) and datetime.date.today() >= review_date:
                msg = f"<b>Trading Bot - {ticker}</b> hit its review date, still holding - ${price:.2f} (P/L {pnl_pct:+.1f}%), stop ${pos['stop_loss']:.2f}. Your call."
                await send_async_message(msg, chat_id=SELL_CHAT_ID)
                pos["review_sent"] = True
                save_positions()

        except Exception as e:
            print(f"⚠️ Error checking sell signal for {ticker}: {e}")

# === Once-a-day status check-in, even when nothing fired ===
async def send_daily_digest():
    if not open_positions:
        return
    try:
        prices = yf.download(list(open_positions.keys()), period="1d", interval="1d", auto_adjust=False, progress=False, group_by="ticker")
    except Exception as e:
        print(f"⚠️ Could not fetch prices for daily digest: {e}")
        return

    lines = [f"<b>Trading Bot - Daily Position Check-In</b> ({len(open_positions)} open)", "----------------------"]
    for ticker, pos in open_positions.items():
        try:
            last_price = float(prices[ticker]["Close"].dropna().iloc[-1]) if len(open_positions) > 1 else float(prices["Close"].dropna().iloc[-1])
        except Exception:
            last_price = None
        pnl = f"{(last_price - pos['buy_price']) / pos['buy_price'] * 100:+.1f}%" if last_price else "n/a"
        price_str = f"${last_price:.2f}" if last_price else "n/a"
        lines.append(f"- <b>{ticker}</b>: {price_str} (P/L {pnl}) | Stop ${pos['stop_loss']:.2f} | Review {pos['review_by']}")

    await send_async_message("\n".join(lines), chat_id=SELL_CHAT_ID)
    print("📋 Daily digest sent.")

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
        startup_msg = "<b>Trading Bot started</b>"
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
                            try:
                                await send_daily_digest()
                            except Exception as e:
                                print(f"⚠️ Error sending daily digest: {e}")

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
    except Exception as e:
        # A silent crash on a hosted bot can go unnoticed for days - say something.
        print(f"🔥 FATAL: bot crashed: {e}")
        try:
            asyncio.run(send_async_message(f"<b>Trading Bot crashed and stopped:</b>\n<code>{e}</code>\nIt needs a manual restart."))
        except Exception:
            pass
        try:
            release_leadership()
        except Exception:
            pass
        raise

        
























