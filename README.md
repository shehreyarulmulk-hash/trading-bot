# Trading Signal Bot

A Python bot that checks technical signals (RSI, SMA crossovers, MACD, Bollinger Bands, squeezes, 52-week levels)
and sends alerts to Telegram.

## Run locally
1. Install dependencies:
   `pip install -r requirements.txt`

2. Export your credentials:
   `export TELEGRAM_BOT_TOKEN=your_token`
   `export TELEGRAM_CHAT_ID=your_chat_id`

3. Run the bot:
   `python signal_bot.py`
