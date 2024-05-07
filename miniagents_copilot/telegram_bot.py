"""
This module is responsible for creating the telegram bot instance.
"""

from telegram.ext import ApplicationBuilder

from versatilis_config import TELEGRAM_TOKEN

telegram_app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
