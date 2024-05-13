"""
TODO Oleksandr: figure out the role of this module
"""

from pathlib import Path

from miniagents.miniagents import miniagent, InteractionContext

from versatilis_config import anthropic_agent


@miniagent
async def versatilis_agent(ctx: InteractionContext) -> None:
    """
    The main MiniAgent.
    """
    ctx.reply(
        anthropic_agent.inquire(
            [_INITIAL_PROMPT, ctx.messages],
            model="claude-3-haiku-20240307",
            # model="claude-3-sonnet-20240229",
            # model="claude-3-opus-20240229",
            max_tokens=1000,
            temperature=0.0,
        )
    )


def _prepare_initial_prompt() -> str:
    miniagents_dir = Path("../MiniAgents")
    miniagent_files = [(f.relative_to(miniagents_dir).as_posix(), f) for f in miniagents_dir.rglob("*")]
    miniagent_files = [
        (f_posix, f)
        for f_posix, f in miniagent_files
        if f.is_file()
        if (
            not any(f_posix.startswith(prefix) for prefix in [".", "venv/", "dist/", "htmlcov/"])
            and not any(f_posix.endswith(suffix) for suffix in [".pyc"])
            and not any(f_posix in full_path for full_path in ["poetry.lock"])
            and f.stat().st_size > 0
        )
    ]
    miniagent_files.sort(key=lambda entry: entry[0])
    miniagent_files_str = "\n".join([f_posix for f_posix, _ in miniagent_files])

    return "\n\n\n\n".join(
        [
            f"```\n{miniagent_files_str}\n```",
            *(f"{f_posix}\n```\n{f.read_text(encoding="utf-8")}\n```" for f_posix, f in miniagent_files),
        ]
    )


_INITIAL_PROMPT = _prepare_initial_prompt()
