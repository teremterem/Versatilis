"""
A MiniAgent that is connected to a Telegram bot.
"""

import asyncio
import logging
from typing import AsyncIterable

import telegram.error
from miniagents.messages import Message
from miniagents.miniagents import miniagent, InteractionContext
from miniagents.promising.sentinels import AWAIT, CLEAR
from miniagents.utils import achain_loop, split_messages
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder

from miniagents_copilot.agents.history_agents import append_history_agent
from miniagents_copilot.agents.versatilis_agents import versatilis_agent
from versatilis_config import TELEGRAM_TOKEN

logger = logging.getLogger(__name__)

telegram_app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

telegram_input_queue = asyncio.Queue()
LAST_TELEGRAM_CHAT_ID = None


@miniagent
async def telegram_update_agent(ctx: InteractionContext) -> None:
    """
    MiniAgent that receives Telegram updates from the webhook.
    """
    # noinspection PyBroadException
    try:
        async for message_promise in ctx.messages:
            message = await message_promise
            update: Update = Update.de_json(message.model_dump(), telegram_app.bot)
            await process_telegram_update(update)
    except Exception:  # pylint: disable=broad-except
        logger.exception("ERROR PROCESSING A TELEGRAM UPDATE")


async def process_telegram_update(update: Update) -> None:
    """
    Process a Telegram update.
    """
    global LAST_TELEGRAM_CHAT_ID  # pylint: disable=global-statement

    if not update.effective_message or not update.effective_message.text or update.edited_message:
        return

    LAST_TELEGRAM_CHAT_ID = update.effective_chat.id
    await telegram_input_queue.put(update.effective_message.text)


async def telegram_chain_loop() -> None:
    """
    The main Telegram agent loop.
    """
    try:
        # The following function will not return until the conversation is over (and it is never over :D)
        await achain_loop(
            agents=[
                user_agent,  # the following agent spews out only user input and nothing else
                AWAIT,
                CLEAR,  # whole dialog (including current exchange) will be read from history file in next step
                versatilis_agent,  # this agent spews out only its own response
                echo_to_console,
            ],
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("ERROR IN THE CONVERSATION LOOP")
        await telegram_app.bot.send_message(chat_id=LAST_TELEGRAM_CHAT_ID, text="Sorry, something went wrong 🤖")
        await telegram_app.bot.send_message(chat_id=LAST_TELEGRAM_CHAT_ID, text=str(exc))


@miniagent
async def echo_to_console(ctx: InteractionContext) -> None:
    """
    MiniAgent that echoes messages to the console token by token.
    """
    ctx.reply(ctx.messages)  # return the messages as they are
    async for message_promise in ctx.messages:
        async for token in message_promise:
            print(f"\033[92;1m{token}\033[0m", end="", flush=True)


@miniagent
async def user_agent(ctx: InteractionContext) -> None:
    """
    This is a proxy agent that represents the user in the conversation loop. It is also responsible for maintaining
    the chat history.
    """
    versatilis_output = split_messages(ctx.messages, role="assistant")
    telegram_input = telegram_user_agent.inquire(versatilis_output)

    ctx.reply(telegram_input)

    # append the user input to the chat history and wait until the append operation is done
    await append_history_agent.inquire(
        [
            versatilis_output,
            telegram_input,
        ]
    ).acollect_messages()


@miniagent
async def telegram_user_agent(ctx: InteractionContext) -> None:
    """
    Integration of Telegram as an input channel.
    """
    # send Versatilis messages to Telegram

    async for message_promise in ctx.messages:
        if LAST_TELEGRAM_CHAT_ID is not None:
            await telegram_app.bot.send_chat_action(LAST_TELEGRAM_CHAT_ID, "typing")

        # it's ok to sleep asynchronously, because the message tokens will be collected in the background
        # anyway, thanks to the way `MiniAgents` (or, more specifically, `promising`) framework is designed
        await asyncio.sleep(1)

        message = await message_promise
        if str(message).strip():
            try:
                await telegram_app.bot.send_message(
                    chat_id=LAST_TELEGRAM_CHAT_ID, text=str(message), parse_mode=ParseMode.MARKDOWN
                )
            except telegram.error.BadRequest:
                await telegram_app.bot.send_message(chat_id=LAST_TELEGRAM_CHAT_ID, text=str(message))

    # receive user responses to Versatilis messages (aka user inputs)

    async for user_input in get_user_inputs():
        if user_input == "/start":
            # /start command means that we want to force a response from the agent - so we break the user input
            # loop and let the agent respond
            break
        ctx.reply(user_input)


async def get_user_inputs() -> AsyncIterable[str]:
    """
    Get user inputs from the Telegram input queue. Do "smart waiting" if the user sends quick follow-ups.
    """
    yield await telegram_input_queue.get()
    try:
        # let's give the user a chance to send a follow-up if they forgot something
        yield await asyncio.wait_for(telegram_input_queue.get(), timeout=3)
        while True:
            # if they did actually send a follow-up, then let's wait for a bit longer
            yield await asyncio.wait_for(telegram_input_queue.get(), timeout=15)
    except asyncio.TimeoutError:
        # if timeout happens we just finish the function - the user is done sending messages and is waiting
        # for a response from the Versatilis agent
        pass


class TelegramUpdateMessage(Message):
    """
    Telegram update MiniAgent message.
    """
