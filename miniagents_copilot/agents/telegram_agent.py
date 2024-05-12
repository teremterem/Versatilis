"""
A MiniAgent that is connected to a Telegram bot.
"""

import asyncio
import logging

import telegram.error
from miniagents.messages import Message, MessageType
from miniagents.miniagents import miniagent, InteractionContext, MessageSequence
from miniagents.utils import split_messages
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder

from miniagents_copilot.agents.versatilis_agent import versatilis_agent
from versatilis_config import TELEGRAM_TOKEN

logger = logging.getLogger(__name__)

telegram_app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

active_chats: dict[int, asyncio.Queue] = {}
chat_histories: dict[int, list[MessageType]] = {}


@miniagent
async def telegram_update_agent(ctx: InteractionContext) -> None:
    """
    MiniAgent that receives Telegram updates from the webhook.
    """
    # noinspection PyBroadException
    try:
        async for message_promise in ctx.messages:
            message = await message_promise.acollect()
            update: Update = Update.de_json(message.model_dump(), telegram_app.bot)
            await process_telegram_update(update)
    except Exception:  # pylint: disable=broad-except
        logger.exception("ERROR PROCESSING A TELEGRAM UPDATE")


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
        if update.effective_chat.id not in active_chats:
            # Start a conversation if it is not already started.
            # The following function will not return until the conversation is over (and it is never over :D)
            active_chats[update.effective_chat.id] = asyncio.Queue()
            await loop_chain(telegram_chat_id=update.effective_chat.id)
        return

    if update.effective_chat.id not in active_chats:
        # conversation is not started yet
        return

    queue = active_chats[update.effective_chat.id]
    await queue.put(update.effective_message.text)


async def loop_chain(telegram_chat_id: int, messages: MessageType = None) -> None:
    """
    TODO Oleksandr: move this function to `miniagents.utils` and add a docstring
    """
    while True:
        messages = versatilis_agent.inquire(messages)
        messages = user_agent.inquire(messages, telegram_chat_id=telegram_chat_id)
        await messages.acollect_messages()  # let's wait for user messages to avoid instant looping


@miniagent
async def user_agent(ctx: InteractionContext, telegram_chat_id: int) -> None:
    """
    This is a proxy agent that represents the user in the conversation loop. It is also responsible for maintaining
    the chat history.
    """
    history = chat_histories.setdefault(telegram_chat_id, [])
    cur_interaction_seq = MessageSequence()

    history.append(cur_interaction_seq.sequence_promise)
    # TODO Oleksandr: implement a utility in MiniAgents that deep-copies/freezes mutable data containers
    #  while keeping objects of other types intact and use it in AppendProducer to freeze the state of those
    #  objects upon their submission (this way the user will not have to worry about things like `history[:]`
    #  in the code below)
    ctx.reply(history[:])

    with cur_interaction_seq.append_producer as interaction_appender:
        async for message_promise in split_messages(ctx.messages):
            await telegram_app.bot.send_chat_action(telegram_chat_id, "typing")

            # it's ok to sleep asynchronously, because the message tokens will be collected in the background anyway,
            # thanks to the way `MiniAgents` (or, more specifically, `promising`) framework is designed
            await asyncio.sleep(1)

            message = await message_promise.acollect()
            try:
                await telegram_app.bot.send_message(
                    chat_id=telegram_chat_id, text=str(message), parse_mode=ParseMode.MARKDOWN
                )
            except telegram.error.BadRequest:
                await telegram_app.bot.send_message(chat_id=telegram_chat_id, text=str(message))

            interaction_appender.append(message)

        chat_queue = active_chats[telegram_chat_id]
        interaction_appender.append(await chat_queue.get())
        try:
            # let's give the user a chance to send a follow-up if they forgot something
            interaction_appender.append(await asyncio.wait_for(chat_queue.get(), timeout=3))
            while True:
                # if they did actually send a follow-up, then let's wait for a bit longer
                interaction_appender.append(await asyncio.wait_for(chat_queue.get(), timeout=15))
        except asyncio.TimeoutError:
            # if timeout happens we just finish the function - the user is done sending messages and is waiting for a
            # response from the Versatilis agent
            pass
    # TODO Oleksandr: send "Sorry, something went wrong 🤖" followed by `str(error)` if something goes wrong


class TelegramUpdateMessage(Message):
    """
    Telegram update MiniAgent message.
    """
