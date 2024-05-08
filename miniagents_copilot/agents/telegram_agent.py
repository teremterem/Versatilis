"""
A miniagent that is connected to a Telegram bot.
"""

from miniagents.messages import Message
from miniagents.miniagents import miniagent, InteractionContext
from telegram import Update
from telegram.ext import ApplicationBuilder

from versatilis_config import TELEGRAM_TOKEN, anthropic_agent

telegram_app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()


@miniagent
async def telegram_update_agent(ctx: InteractionContext) -> None:
    """
    MiniAgent that receives Telegram updates from the webhook.
    """
    async for message_promise in ctx.messages:
        message = await message_promise.acollect()
        update: Update = Update.de_json(message.model_dump(), telegram_app.bot)
        await process_telegram_update(update)


async def process_telegram_update(update: Update) -> None:
    """
    Process a Telegram update.
    """
    if not update.effective_message or not update.effective_message.text:
        return

    if update.edited_message:
        # TODO Oleksandr: update the history when messages are edited
        return

    if update.effective_message.text == "/start":
        # TODO Oleksandr: `delegate` instead of `inquire`
        reverse_agent.inquire(telegram_chat_id=update.effective_chat.id)
        return

    # TODO TODO TODO
    reply_sequence = anthropic_agent.inquire(
        update.effective_message.text,
        model="claude-3-haiku-20240307",  # "claude-3-opus-20240229",
        max_tokens=1000,
        temperature=0.0,
    )
    async for reply_promise in reply_sequence:
        reply = await reply_promise.acollect()
        await update.effective_chat.send_message(str(reply))


@miniagent
async def reverse_agent(_: InteractionContext, telegram_chat_id: int) -> None:
    """
    MiniAgent that starts the conversation first.
    """
    await telegram_app.bot.send_message(telegram_chat_id, "Hello, I am a miniagent. How can I help you?")


class TelegramUpdateMessage(Message):
    """
    Telegram update MiniAgent message.
    """
