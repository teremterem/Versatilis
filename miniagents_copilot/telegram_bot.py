"""
This module is responsible for creating the telegram bot instance.
"""

from miniagents.messages import Message
from telegram.ext import ApplicationBuilder

from versatilis_config import TELEGRAM_TOKEN


class TelegramUpdateMessage(Message):
    """
    Telegram update MiniAgent message.
    """


telegram_app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
