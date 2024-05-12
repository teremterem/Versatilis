"""
The main MiniAgent.
"""

from pathlib import Path

from miniagents.messages import Message
from miniagents.miniagents import miniagent, InteractionContext

from versatilis_config import anthropic_agent


@miniagent
async def versatilis_agent(ctx: InteractionContext) -> None:
    """
    The main MiniAgent.
    """
    messages = await ctx.messages.acollect_messages()
    if messages:
        ctx.reply(
            anthropic_agent.inquire(
                messages,
                # model="claude-3-haiku-20240307",
                # model="claude-3-sonnet-20240229",
                model="claude-3-opus-20240229",
                max_tokens=1000,
                temperature=0.0,
            )
        )
    else:
        miniagents_dir = Path("../MiniAgents")
        miniagent_files = [(f.relative_to(miniagents_dir).as_posix(), f) for f in miniagents_dir.rglob("*")]
        miniagent_files = [
            f_posix
            for f_posix, f in miniagent_files
            if f.is_file()
            if (
                not any(f_posix.startswith(prefix) for prefix in [".", "venv/", "dist/", "htmlcov/"])
                and not any(f_posix.endswith(suffix) for suffix in [".pyc"])
                and not any(f_posix in full_path for full_path in ["poetry.lock"])
                and f.stat().st_size > 0
            )
        ]
        miniagent_files.sort()
        miniagent_files_str = "\n".join(miniagent_files)
        ctx.reply(Message(text=f"```\n{miniagent_files_str}\n```", role="assistant"))
        ctx.reply(Message(text="Hello, I am Versatilis. How can I help you?", role="assistant"))
