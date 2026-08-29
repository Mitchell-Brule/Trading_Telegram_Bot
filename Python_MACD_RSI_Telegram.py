import yfinance as yf
import ta
from telegram import Bot
import time
import pickle
import os

# === Telegram setup ===
bot_token = "YOUR_BOT_TOKEN"  # Replace with your bot token
chat_id = YOUR_CHAT_ID        # Replace with your chat ID
bot = Bot(token=bot_token)

# === List of stocks to monitor (start small for testing) ===
tickers = ["AAPL", "MSFT", "TSLA", "AMZN", "GOOGL"]

# === File to store previous alerts ===
alert_file = "alerts.pkl"

if os.path.exists(alert_file):
    with open(alert_file, "rb") as f:
        alerted = pickle.load(f)
else:
    alerted = set()

# === Function to check signals ===
def check_signals():
    global alerted
    try:
        data_dict = yf.download(tickers, period="3mo", interval="1h", group_by='ticker', auto_adjust=True)
        for ticker in tickers:
            try:
                data = data_dict[ticker].copy()
                close_prices = data['Close'].squeeze()

                # EMA trend filter
                data['ema50'] = ta.trend.ema_indicator(close_prices, window=50)
                trend_ok = close_prices > data['ema50']

                # MACD & RSI
                data['macd'] = ta.trend.macd(close_prices)
                data['macd_signal'] = ta.trend.macd_signal(close_prices)
                data['rsi'] = ta.momentum.rsi(close_prices, window=14)

                # Signals
                data['macd_cross'] = (data['macd'] > data['macd_signal']) & (data['macd'].shift(1) <= data['macd_signal'].shift(1))
                data['rsi_signal'] = data['rsi'] < 50
                data['buy_signal'] = data['macd_cross'] & data['rsi_signal'] & trend_ok

                if data['buy_signal'].iloc[-1] and ticker not in alerted:
                    current_rsi = data['rsi'].iloc[-1]
                    message = f"{ticker} MACD crossed above signal, RSI={current_rsi:.2f} < 50, above EMA50"
                    print(message)
                    bot.send_message(chat_id=chat_id, text=message)
                    alerted.add(ticker)
                elif not data['buy_signal'].iloc[-1] and ticker in alerted:
                    alerted.remove(ticker)

            except Exception as e:
                print(f"Error processing {ticker}: {e}")

        with open(alert_file, "wb") as f:
            pickle.dump(alerted, f)

    except Exception as e:
        print(f"Error fetching data: {e}")

# === Main loop: every 10 minutes ===
while True:
    print("Checking signals...")
    check_signals()
    print("Waiting 10 minutes before next check...")
    time.sleep(10 * 60)
