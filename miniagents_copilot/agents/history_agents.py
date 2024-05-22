"""
This module contains agents that are responsible for fetching and appending chat history to `CHAT.md` file.
"""

from miniagents.messages import Message
from miniagents.miniagents import miniagent, InteractionContext

from miniagents_copilot.agents.versatilis_agents import BASE_SETUP_FOLDER, MODEL


@miniagent
async def fetch_history_agent(ctx: InteractionContext) -> None:
    """
    TODO Oleksandr: docstring
    """
    chat_history_file = BASE_SETUP_FOLDER / "CHAT.md"
    history_file_not_empty = chat_history_file.exists() and chat_history_file.stat().st_size > 0

    history_messages = []
    if history_file_not_empty:
        history_md = chat_history_file.read_text(encoding="utf-8")

        portions = history_md.split("\n-------------------------------\n")
        last_role = None
        for idx, portion in enumerate(portions):
            if idx == 0:
                last_role = portion.rsplit("\n", maxsplit=1)[-1]
                continue

            if idx == len(portions) - 1:
                last_message = portion  # the whole "text portion" is a message
                next_role = last_role  # there is no next role, so we just keep the last one
            else:
                last_message, next_role = portion.rsplit("\n", maxsplit=1)

            # if it's a model name, we just turn it into "assistant"
            last_generic_role = last_role if last_role in ["user", "system"] else "assistant"
            # remove trailing spaces and newlines from the message
            last_message = last_message.rstrip()

            history_messages.append(Message(text=last_message, role=last_generic_role))
            last_role = next_role

    ctx.reply(history_messages)
    ctx.reply(ctx.messages)


@miniagent(
    current_model=MODEL,  # TODO Oleksandr: fix `split_messages()` so `model` could be read from resulting messages ?
)
async def append_history_agent(ctx: InteractionContext, current_model: str):
    """
    TODO Oleksandr: docstring
    """
    ctx.reply(ctx.messages)  # just pass the same input messages forward (before saving them to the history file)

    chat_history_file = BASE_SETUP_FOLDER / "CHAT.md"
    history_file_not_empty = chat_history_file.exists() and chat_history_file.stat().st_size > 0

    last_role = None
    with chat_history_file.open("a", encoding="utf-8") as chat_history:
        async for message_promise in ctx.messages:
            message = await message_promise
            if not str(message).strip():
                continue

            role = getattr(message, "role", None) or "user"
            if role == "assistant":
                role = current_model

            if role != last_role:
                if history_file_not_empty:
                    chat_history.write("\n")
                else:
                    history_file_not_empty = True

                chat_history.write(f"{role}\n-------------------------------")
                last_role = role

            chat_history.write(f"\n{message}\n")
            chat_history.flush()
