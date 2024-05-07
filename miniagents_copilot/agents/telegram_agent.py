"""
A miniagent that is connected to a Telegram bot.
"""

from pprint import pprint

from miniagents.miniagents import miniagent, InteractionContext
from telegram import Update

from miniagents_copilot.telegram_bot import telegram_app


@miniagent
async def telegram_agent(ctx: InteractionContext) -> None:
    """
    Telegram agent.
    """
    async for message_promise in ctx.messages:
        message = await message_promise.acollect()
        update = Update.de_json(message.model_dump(), telegram_app.bot)
        print()
        pprint(update.to_json())
        print()
