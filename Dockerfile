# Use Python 3.11 base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and code
COPY requirements.txt .
COPY Python_MACD_RSI_Telegram_test.py .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Start the bot
CMD ["python", "Python_MACD_RSI_Telegram_test.py"]
