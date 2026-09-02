# === Use Python base image ===
FROM python:3.10-slim

# === Set working directory ===
WORKDIR /app

# === Copy project files ===
COPY . /app

# === Install dependencies ===
RUN pip install --no-cache-dir -r requirements.txt

# === Let the host restart the container if the bot goes unresponsive ===
HEALTHCHECK --interval=5m --timeout=10s --start-period=1m --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:5000/health', timeout=8).status == 200 else 1)"

# === Run your bot ===
CMD ["python", "Python_MACD_RSI_Telegram_test.py"]

