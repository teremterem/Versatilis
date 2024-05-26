"""
This module contains agents that are responsible for fetching and appending chat history to `CHAT.md` file.
"""

from pathlib import Path

from miniagents.messages import Message
from miniagents.miniagents import miniagent, InteractionContext


async def fetch_history(history_file: str | Path) -> list[Message]:
    """
    TODO Oleksandr: docstring
    """
    history_file = Path(history_file)

    if not history_file.exists():
        return []

    history_md = history_file.read_text(encoding="utf-8")
    if not history_md.strip():
        return []

    history_messages = []
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

    return history_messages


@miniagent
async def append_history_agent(ctx: InteractionContext, history_file: str | Path, model: str):
    """
    TODO Oleksandr: docstring
    """
    ctx.reply(ctx.messages)  # just pass the same input messages forward (before saving them to the history file)

    history_file_not_empty = history_file.exists() and history_file.stat().st_size > 0

    last_role = None
    with history_file.open("a", encoding="utf-8") as chat_history:
        async for message_promise in ctx.messages:
            message = await message_promise
            if not str(message).strip():
                continue

            role = getattr(message, "role", None) or "user"
            if role == "assistant":
                role = model  # TODO Oleksandr: fix `split_messages()` so `model` could be read from messages ?

            if role != last_role:
                if history_file_not_empty:
                    chat_history.write("\n")
                else:
                    history_file_not_empty = True

                chat_history.write(f"{role}\n-------------------------------")
                last_role = role

            chat_history.write(f"\n{message}\n")
            chat_history.flush()
