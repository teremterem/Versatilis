"""
A miniagent that is connected to a Telegram bot.
"""

import asyncio
import logging

import telegram.error
from miniagents.messages import Message
from miniagents.miniagents import miniagent, InteractionContext
from miniagents.utils import split_messages
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder

from versatilis_config import TELEGRAM_TOKEN, anthropic_agent

logger = logging.getLogger(__name__)

telegram_app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

active_chats: dict[int, asyncio.Queue] = {}


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
            await conversation_loop(telegram_chat_id=update.effective_chat.id)
        return

    if update.effective_chat.id not in active_chats:
        # conversation is not started yet
        return

    queue = active_chats[update.effective_chat.id]
    await queue.put(update.effective_message.text)


async def conversation_loop(telegram_chat_id: int) -> None:
    """
    Conversation loop between the user and the Versatilis agent.
    """
    # TODO Oleksandr: introduce a concept of conversation managers into the MiniAgent framework
    history = []

    error = None
    while True:
        # noinspection PyBroadException
        try:
            # TODO Oleksandr: implement a utility in MiniAgents that deep-copies/freezes mutable data containers
            #  while keeping objects of other types intact and use it in AppendProducer to freeze the state of those
            #  objects upon their submission (this way the user will not have to worry about things like `list(history)`
            #  in the code below)
            if error:
                # if there was an error then we just wait for the user input and don't ask Versatilis again
                versatilis_reply_sequence = ["Sorry, something went wrong 🤖", str(error)]
            else:
                versatilis_reply_sequence = versatilis_agent.inquire(list(history))
                # we are putting the whole sequence as one element (the framework supports this)
                # TODO Oleksandr: don't just blindly put the whole Versatilis response into the history, better make
                #  the history a responsibility of the user agent, so it only puts into the history the messages that
                #  it successfully delivered to the user
                history.append(versatilis_reply_sequence)

            user_replies = await user_agent.inquire(
                versatilis_reply_sequence, telegram_chat_id=telegram_chat_id
            ).acollect_messages()  # let's wait for user messages to avoid instant looping

            history.extend(user_replies)
            error = None
        except Exception as exc:  # pylint: disable=broad-except
            if error:
                # this is the second error in a row - let's break the loop
                raise exc from error
            logger.exception("ERROR IN THE CONVERSATION LOOP")
            error = exc


@miniagent
async def user_agent(ctx: InteractionContext, telegram_chat_id: int) -> None:
    """
    This is a proxy agent that represents the user in the conversation loop.
    """
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

    chat_queue = active_chats[telegram_chat_id]
    ctx.reply(await chat_queue.get())
    try:
        # let's give the user a chance to send a follow-up if they forgot something
        ctx.reply(await asyncio.wait_for(chat_queue.get(), timeout=3))
        while True:
            # if they did actually send a follow-up, then let's wait for a bit longer
            ctx.reply(await asyncio.wait_for(chat_queue.get(), timeout=15))
    except asyncio.TimeoutError:
        # if timeout happens we just finish the function - the user is done sending messages and is waiting for a
        # response
        pass


@miniagent
async def versatilis_agent(ctx: InteractionContext) -> None:
    """
    The main agent.
    """
    messages = await ctx.messages.acollect_messages()
    if messages:
        ctx.reply(
            anthropic_agent.inquire(
                messages,
                model="claude-3-haiku-20240307",
                # model="claude-3-opus-20240229",
                max_tokens=1000,
                temperature=0.0,
            )
        )
    else:
        ctx.reply(Message(text="Hello, I am Versatilis. How can I help you?", role="assistant"))


class TelegramUpdateMessage(Message):
    """
    Telegram update MiniAgent message.
    """
