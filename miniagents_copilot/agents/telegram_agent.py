"""
A miniagent that is connected to a Telegram bot.
"""

from pprint import pprint

from miniagents.messages import Message
from miniagents.miniagents import miniagent, InteractionContext
from telegram import Update
from telegram.ext import ApplicationBuilder

from versatilis_config import TELEGRAM_TOKEN

telegram_app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()


@miniagent
async def telegram_agent(ctx: InteractionContext) -> None:
    """
    Telegram agent.
    """
    async for message_promise in ctx.messages:
        message = await message_promise.acollect()
        update = Update.de_json(message.model_dump(), telegram_app.bot)
        print()
        pprint(update.to_dict())
        print()


class TelegramUpdateMessage(Message):
    """
    Telegram update MiniAgent message.
    """
