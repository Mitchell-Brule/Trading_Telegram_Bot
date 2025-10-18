# === Use Python base image ===
FROM python:3.10-slim

# === Set working directory ===
WORKDIR /app

# === Copy project files ===
COPY . /app

# === Install dependencies ===
RUN pip install --no-cache-dir -r requirements.txt

# === Run your bot ===
CMD ["python", "Python_MACD_RSI_Telegram_test.py"]

