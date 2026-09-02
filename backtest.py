"""
Research/backtest tool for Python_MACD_RSI_Telegram_test.py's LOCK strategy.

Caches raw OHLCV+indicator data per ticker to disk (data_cache/) so any new
idea - a different entry pattern, a different holding horizon, a different
stop-loss formula - can be tested in seconds without re-downloading. The
network fetch is the expensive part; everything after it is cheap.

Run standalone (no Telegram/Flask/env vars needed), safe to run anytime
without touching the live bot's state.

Known limitation: earnings-date exclusion (which the live bot applies) is
NOT replayed here - yfinance's calendar API only returns the *next* earnings
date relative to today, not what was "next" at each historical point in
time, so it can't be honestly backtested with the same free data source.

Usage:
    python backtest.py                              # fetch (or reuse cache), full report + sweep
    python backtest.py --refresh                    # force re-download everything
    python backtest.py --months 20                  # lookback window
    python backtest.py --strategy bollinger          # test a different entry pattern
    python backtest.py --horizon-sweep               # compare holding horizons
    python backtest.py --min-vol 1.3 --min-adx 20 --horizon 10
"""
import sys
import os
import time
import argparse
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import yfinance as yf
import ta

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

DEFAULT_TICKERS = [
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

BARS_PER_DAY = 6  # 4h bars
CACHE_DIR = "data_cache"
CACHE_MAX_AGE_HOURS = 20  # reuse same-day cache across many experiment runs


# ============================== Data layer ===============================

def _cache_path(ticker, months):
    safe = ticker.replace("/", "_")
    return os.path.join(CACHE_DIR, f"{safe}_{months}mo.pkl")


def get_ticker_df(ticker, months, refresh=False):
    """Raw OHLCV, cached to disk. Indicators are NOT baked in here, so
    changing an indicator's parameters never requires a re-download."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(ticker, months)
    if not refresh and os.path.exists(path):
        age_hours = (time.time() - os.path.getmtime(path)) / 3600
        if age_hours < CACHE_MAX_AGE_HOURS:
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
    try:
        df = yf.download(ticker, period=f"{months}mo", interval="4h", auto_adjust=False, progress=False)
    except Exception as e:
        print(f"  [!] {ticker}: download failed ({e})")
        return None
    if df.empty or len(df) < 250:
        print(f"  [!] {ticker}: not enough history ({len(df)} bars), skipping")
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    with open(path, "wb") as f:
        pickle.dump(df, f)
    return df


def get_spy_regime(refresh=False):
    path = os.path.join(CACHE_DIR, "SPY_regime.pkl")
    os.makedirs(CACHE_DIR, exist_ok=True)
    if not refresh and os.path.exists(path):
        age_hours = (time.time() - os.path.getmtime(path)) / 3600
        if age_hours < CACHE_MAX_AGE_HOURS:
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
    try:
        spy = yf.download("SPY", period="2y", interval="1d", auto_adjust=False, progress=False)
    except Exception as e:
        print(f"[!] Could not fetch SPY regime series: {e}")
        return pd.Series(dtype=bool)
    if spy.empty:
        return pd.Series(dtype=bool)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    close = spy["Close"].dropna()
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200, min_periods=50).mean()
    result = ((ma50 > ma200)).shift(1)
    with open(path, "wb") as f:
        pickle.dump(result, f)
    return result


def build_indicators(raw_df):
    """Adds every indicator any strategy variant below might need. Cheap -
    recomputed fresh each run from the cached raw OHLCV."""
    df = raw_df.copy()
    close, high, low = df["Close"], df["High"], df["Low"]
    macd_ind = ta.trend.MACD(close)
    df["MACD"] = macd_ind.macd()
    df["Signal"] = macd_ind.macd_signal()
    df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    df["MA20"] = close.rolling(20, min_periods=1).mean()
    df["MA50"] = close.rolling(50, min_periods=1).mean()
    df["MA200"] = close.rolling(200, min_periods=1).mean()
    df["Vol20"] = df["Volume"].rolling(20, min_periods=1).mean()
    try:
        df["ATR"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    except Exception:
        df["ATR"] = np.nan
    try:
        df["ADX"] = ta.trend.ADXIndicator(high, low, close, window=14).adx()
    except Exception:
        df["ADX"] = np.nan
    try:
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        df["BB_HIGH"] = bb.bollinger_hband()
        df["BB_LOW"] = bb.bollinger_lband()
        df["BB_WIDTH"] = (df["BB_HIGH"] - df["BB_LOW"]) / close
    except Exception:
        df["BB_HIGH"] = df["BB_LOW"] = df["BB_WIDTH"] = np.nan
    df["PriorLow"] = close.rolling(60).min().shift(3)

    daily = close.resample("1D").last().dropna()
    d_ma20 = daily.rolling(20).mean()
    d_ma50 = daily.rolling(50).mean()
    d_ma200 = daily.rolling(200, min_periods=50).mean()
    # Full aligned-uptrend requirement (price>MA20>MA50>MA200), tightened from
    # price>MA50>MA200 on 2026-08-31 - see LOCK-tier comment in the live bot.
    daily_bullish = ((daily > d_ma20) & (d_ma20 > d_ma50) & (d_ma50 > d_ma200)).shift(1)
    df["DailyTrendOK"] = daily_bullish.reindex(df.index, method="ffill")

    return df


def attach_spy(df, spy_bullish):
    if spy_bullish.empty:
        df["SpyOk"] = False
    else:
        idx_naive = df.index.tz_localize(None) if df.index.tz is not None else df.index
        df["SpyOk"] = spy_bullish.reindex(idx_naive, method="ffill").values
    return df


def load_universe(tickers, months, refresh=False, max_workers=6):
    """Returns {ticker: fully-indicator-built df}. Network fetch (the
    expensive part) is parallelized across threads - cached tickers return
    near-instantly so this is cheap on repeat runs regardless."""
    spy_bullish = get_spy_regime(refresh=refresh)
    raw_by_ticker = {}
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(get_ticker_df, t, months, refresh): t for t in tickers}
        for future in as_completed(futures):
            t = futures[future]
            try:
                raw = future.result()
            except Exception as e:
                print(f"  [!] {t}: fetch error ({e})")
                raw = None
            if raw is not None:
                raw_by_ticker[t] = raw
            done += 1
            if done % 50 == 0:
                print(f"  ...{done}/{len(tickers)} tickers fetched")

    data = {}
    for t, raw in raw_by_ticker.items():
        df = build_indicators(raw)
        df = attach_spy(df, spy_bullish)
        data[t] = df
    return data


# ============================== Entry strategies ===============================
# Each returns a boolean Series aligned to df.index: True where that strategy's
# entry condition fires on that bar.

def entry_classic_macd_rsi(df):
    """The live bot's core trigger: MACD crosses up while RSI turns up out of
    neutral/oversold territory."""
    close, rsi = df["Close"], df["RSI"]
    macd, signal = df["MACD"], df["Signal"]
    cross_up = (macd.shift(1) < signal.shift(1)) & (macd > signal)
    rsi_rising = rsi > rsi.shift(1)
    rsi_prev = rsi.shift(1)
    return cross_up & rsi_rising & (rsi_prev < 55) & (rsi.between(35, 68))


def entry_classic_tight_rsi(df):
    """Variant: same MACD cross, but requires a deeper prior RSI dip (<45
    instead of <55) - a stronger "reversal from real weakness" requirement."""
    close, rsi = df["Close"], df["RSI"]
    macd, signal = df["MACD"], df["Signal"]
    cross_up = (macd.shift(1) < signal.shift(1)) & (macd > signal)
    rsi_rising = rsi > rsi.shift(1)
    rsi_prev = rsi.shift(1)
    return cross_up & rsi_rising & (rsi_prev < 45) & (rsi.between(30, 65))


def entry_bollinger_squeeze_breakout(df):
    """Different trading style: volatility contraction (Bollinger Band width
    at a 100-bar low) followed by a close breaking above the upper band -
    classic "coiled spring" breakout, independent of MACD/RSI."""
    width = df["BB_WIDTH"]
    width_low = width.rolling(100, min_periods=30).min()
    was_squeezed = width.shift(1) <= width_low.shift(1) * 1.1
    breakout = df["Close"] > df["BB_HIGH"]
    breakout_new = breakout & ~breakout.shift(1).fillna(False)
    return was_squeezed & breakout_new


def entry_rsi_oversold_bounce(df):
    """Different trading style: mean reversion. RSI dips under 30 (oversold)
    then closes back above it, while price stays above the daily uptrend -
    "buy the dip in an uptrend" rather than trend-following a crossover."""
    rsi = df["RSI"]
    was_oversold = rsi.shift(1) < 30
    bounced = rsi >= 30
    return was_oversold & bounced & df["DailyTrendOK"].fillna(False)


def entry_macd_zero_cross(df):
    """Variant: require MACD itself (not just the cross vs. signal) to be
    crossing above zero - a stronger, less frequent trend-confirmation signal
    than a signal-line cross alone."""
    macd = df["MACD"]
    return (macd.shift(1) < 0) & (macd > 0)


def entry_donchian_breakout(df):
    """Different trading style: price closes above its own 20-bar high (a
    Donchian-channel breakout) with volume confirmation - momentum/breakout
    style independent of both MACD/RSI and Bollinger Band width."""
    close = df["Close"]
    prior_high = close.rolling(20).max().shift(1)
    breakout = close > prior_high
    vol_ok = df["Volume"] > 1.3 * df["Vol20"]
    return breakout & vol_ok


def entry_classic_or_bollinger(df):
    """Combo: two independent edge sources (mean-reversion crossover +
    volatility breakout) firing on different market conditions - tests
    whether combining them adds frequency without diluting quality."""
    return entry_classic_macd_rsi(df) | entry_bollinger_squeeze_breakout(df)


def entry_triple_combo(df):
    """Three independent edge sources OR'd together - tests whether a third
    style (Donchian breakout) adds more without diluting the classic+bollinger
    combo's quality."""
    return entry_classic_macd_rsi(df) | entry_bollinger_squeeze_breakout(df) | entry_donchian_breakout(df)


STRATEGIES = {
    "classic": entry_classic_macd_rsi,
    "classic_tight": entry_classic_tight_rsi,
    "bollinger": entry_bollinger_squeeze_breakout,
    "rsi_bounce": entry_rsi_oversold_bounce,
    "macd_zero": entry_macd_zero_cross,
    "donchian": entry_donchian_breakout,
    "combo": entry_classic_or_bollinger,
    "triple_combo": entry_triple_combo,
}


# ============================== Exit simulation ===============================

def calc_stop_loss(price, atr, prior_low, atr_mult=3.0, floor_pct=0.88):
    atr_stop = price - (atr_mult * atr) if (atr is not None and not np.isnan(atr) and atr > 0) else price * (1 - (1 - floor_pct) / 2)
    structure_stop = (prior_low * 0.97) if (prior_low is not None and not np.isnan(prior_low)) else atr_stop
    stop = max(min(atr_stop, structure_stop), price * floor_pct)
    return max(min(stop, price * 0.99), 0.01)


def simulate_trade(df, entry_pos, entry_price, horizon_bars, min_hold_bars, atr_mult=3.0):
    atr = df["ATR"].iloc[entry_pos]
    prior_low = df["PriorLow"].iloc[entry_pos]
    stop_loss = calc_stop_loss(entry_price, atr, prior_low, atr_mult=atr_mult)
    initial_risk_pct = (entry_price - stop_loss) / entry_price * 100
    hwm = entry_price

    close, macd, signal = df["Close"], df["MACD"], df["Signal"]
    end_pos = min(entry_pos + horizon_bars, len(df) - 1)

    for t in range(entry_pos + 1, end_pos + 1):
        price_t = float(close.iloc[t])
        if price_t > hwm:
            hwm = price_t
            trail = price_t * (1 - initial_risk_pct / 100)
            if trail > stop_loss:
                stop_loss = trail

        bars_held = t - entry_pos
        cross_down = (macd.iloc[t - 1] > signal.iloc[t - 1]) and (macd.iloc[t] < signal.iloc[t]) and (bars_held >= min_hold_bars)
        if cross_down:
            return price_t, "macd_cross", bars_held
        if price_t <= stop_loss:
            return price_t, "stop_loss", bars_held

    return float(close.iloc[end_pos]), "horizon", end_pos - entry_pos


def run_strategy(data, entry_fn, horizon_days=14, min_hold_days=2, atr_mult=3.0):
    """Applies one entry strategy across the whole loaded universe and
    simulates every resulting trade. Pure in-memory - no I/O."""
    horizon_bars = horizon_days * BARS_PER_DAY
    min_hold_bars = min_hold_days * BARS_PER_DAY
    results = []
    for ticker, df in data.items():
        try:
            entries = entry_fn(df)
        except Exception as e:
            print(f"  [!] {ticker}: strategy error ({e})")
            continue
        vol_ratio_series = df["Volume"] / df["Vol20"]
        close = df["Close"]
        for ts in entries[entries.fillna(False)].index:
            pos = df.index.get_loc(ts)
            if pos + 1 >= len(df):
                continue
            row = df.iloc[pos]
            vol_ratio = float(vol_ratio_series.iloc[pos]) if not np.isnan(vol_ratio_series.iloc[pos]) else 0
            adx_val = row.get("ADX", np.nan)
            daily_ok = bool(row.get("DailyTrendOK", False)) if pd.notna(row.get("DailyTrendOK", np.nan)) else False
            spy_ok = bool(row.get("SpyOk", False)) if pd.notna(row.get("SpyOk", np.nan)) else False

            entry_price = float(close.iloc[pos])
            exit_price, reason, bars_held = simulate_trade(df, pos, entry_price, horizon_bars, min_hold_bars, atr_mult=atr_mult)
            realized_pct = (exit_price - entry_price) / entry_price * 100
            results.append({
                "ticker": ticker, "date": ts,
                "vol_ratio": round(vol_ratio, 2), "adx": round(float(adx_val), 1) if not np.isnan(adx_val) else np.nan,
                "daily_ok": daily_ok, "spy_ok": spy_ok,
                "entry": entry_price, "exit": exit_price, "exit_reason": reason,
                "days_held": round(bars_held / BARS_PER_DAY, 1), "realized_pct": realized_pct,
            })
    rdf = pd.DataFrame(results)
    if not rdf.empty:
        rdf["date"] = pd.to_datetime(rdf["date"], utc=True)  # mixed tz offsets (crypto UTC vs. equity ET) need normalizing
    return rdf


# ============================== Reporting ===============================

def summarize(rdf, label, total_days=None):
    if rdf.empty:
        print(f"--- {label}: 0 signals ---")
        return None
    if total_days is None:
        total_days = (rdf["date"].max() - rdf["date"].min()).days or 1
    n = len(rdf)
    win = (rdf["realized_pct"] > 0).mean() * 100
    avg = rdf["realized_pct"].mean()
    med = rdf["realized_pct"].median()
    days_per_sig = total_days / n
    print(f"--- {label} ---")
    print(f"  n={n}  |  1 signal every {days_per_sig:.1f} days  |  win {win:.0f}%  |  avg {avg:+.2f}%  |  median {med:+.2f}%  |  avg hold {rdf['days_held'].mean():.1f}d")
    return {"n": n, "days_per_sig": days_per_sig, "win": win, "avg": avg, "median": med}


def sweep_lock_thresholds(rdf, total_days, grid):
    print(f"\n{'=' * 100}\nTHRESHOLD SWEEP ({total_days} days of history)\n{'=' * 100}")
    print(f"{'min_vol':>8} {'min_adx':>8} {'daily?':>7} {'spy?':>6} | {'n':>5} {'days/sig':>9} {'win%':>6} {'avg%':>7} {'median%':>8}")
    best = None
    for min_vol, min_adx, req_daily, req_spy in grid:
        mask = (rdf["vol_ratio"] >= min_vol) & (rdf["adx"] >= min_adx)
        if req_daily:
            mask &= rdf["daily_ok"]
        if req_spy:
            mask &= rdf["spy_ok"]
        sub = rdf[mask]
        n = len(sub)
        if n == 0:
            print(f"{min_vol:>8} {min_adx:>8} {str(req_daily):>7} {str(req_spy):>6} | {0:>5} {'--':>9} {'--':>6} {'--':>7} {'--':>8}")
            continue
        days_per_sig = total_days / n
        win = (sub["realized_pct"] > 0).mean() * 100
        avg = sub["realized_pct"].mean()
        med = sub["realized_pct"].median()
        print(f"{min_vol:>8} {min_adx:>8} {str(req_daily):>7} {str(req_spy):>6} | {n:>5} {days_per_sig:>9.1f} {win:>6.0f} {avg:>+7.2f} {med:>+8.2f}")
        if 2.5 <= days_per_sig <= 12:
            score = win + avg * 3  # weight avg return more; win rate alone can be misleading with skewed payoffs
            if best is None or score > best[0]:
                best = (score, min_vol, min_adx, req_daily, req_spy, n, days_per_sig, win, avg)
    if best:
        print(f"\nBest in the 2.5-12 day/signal range: min_vol={best[1]}, min_adx={best[2]}, daily={best[3]}, spy={best[4]} "
              f"-> n={best[5]}, {best[6]:.1f} days/signal, {best[7]:.0f}% win, {best[8]:+.2f}% avg")
    return best


def main():
    parser = argparse.ArgumentParser(description="Research/backtest tool for the LOCK strategy")
    parser.add_argument("tickers", nargs="*", default=DEFAULT_TICKERS)
    parser.add_argument("--months", type=int, default=20, help="Lookback window in months (yfinance 4h data maxes out around 2 years)")
    parser.add_argument("--refresh", action="store_true", help="Force re-download instead of using the disk cache")
    parser.add_argument("--strategy", choices=list(STRATEGIES.keys()), default="bollinger")
    parser.add_argument("--compare-strategies", action="store_true", help="Run every strategy in STRATEGIES and compare")
    parser.add_argument("--horizon", type=int, default=14)
    parser.add_argument("--horizon-sweep", action="store_true", help="Compare several holding horizons")
    parser.add_argument("--min-vol", type=float, default=1.2)
    parser.add_argument("--min-adx", type=float, default=25.0)
    parser.add_argument("--no-sweep", action="store_true")
    args = parser.parse_args()

    print(f"Loading {len(args.tickers)} tickers ({args.months}mo, {'refresh' if args.refresh else 'cache-if-fresh'})...")
    data = load_universe(args.tickers, args.months, refresh=args.refresh)
    print(f"Loaded {len(data)}/{len(args.tickers)} tickers.\n")
    if not data:
        print("No data loaded - aborting.")
        return

    total_days = max((df.index.max() - df.index.min()).days for df in data.values())

    if args.compare_strategies:
        print(f"{'=' * 100}\nSTRATEGY COMPARISON: raw entry quality vs. the same LOCK gate applied\n(vol>={args.min_vol}, adx>={args.min_adx}, daily trend required)\n{'=' * 100}")
        for name, fn in STRATEGIES.items():
            rdf = run_strategy(data, fn, horizon_days=args.horizon)
            summarize(rdf, f"{name} (ungated)", total_days)
            gated = rdf[(rdf["vol_ratio"] >= args.min_vol) & (rdf["adx"] >= args.min_adx) & rdf["daily_ok"]] if not rdf.empty else rdf
            summarize(gated, f"{name} (LOCK-gated)", total_days)
            print()
        return

    entry_fn = STRATEGIES[args.strategy]
    print(f"Running strategy '{args.strategy}' with {args.horizon}d horizon...")
    rdf = run_strategy(data, entry_fn, horizon_days=args.horizon)
    if rdf.empty:
        print("No signals produced.")
        return

    print(f"\n{'=' * 78}\nAll '{args.strategy}' entries: {len(rdf)}\n{'=' * 78}")
    summarize(rdf, f"ALL '{args.strategy}' entries (ungated)", total_days)

    lock_mask = (rdf["vol_ratio"] >= args.min_vol) & (rdf["adx"] >= args.min_adx) & rdf["daily_ok"]
    summarize(rdf[lock_mask], "LOCK-gated", total_days)

    if args.horizon_sweep:
        print(f"\n{'=' * 100}\nHORIZON SWEEP (strategy={args.strategy}, min_vol={args.min_vol}, min_adx={args.min_adx})\n{'=' * 100}")
        for h in (5, 7, 10, 14, 18, 21):
            rdf_h = run_strategy(data, entry_fn, horizon_days=h)
            sub = rdf_h[(rdf_h["vol_ratio"] >= args.min_vol) & (rdf_h["adx"] >= args.min_adx) & rdf_h["daily_ok"]]
            summarize(sub, f"horizon={h}d", total_days)

    if not args.no_sweep:
        grid = [
            (mv, ma, daily, spy)
            for mv in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5)
            for ma in (15, 18, 20, 22, 25)
            for daily in (True, False)
            for spy in (True, False)
        ]
        sweep_lock_thresholds(rdf, total_days, grid)

    rdf.to_csv("backtest_results.csv", index=False)
    print(f"\nFull raw results saved to backtest_results.csv")


if __name__ == "__main__":
    main()
