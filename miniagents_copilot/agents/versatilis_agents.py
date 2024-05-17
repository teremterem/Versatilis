"""
TODO Oleksandr: figure out the role of this module
"""

from functools import partial, wraps
from pathlib import Path

from miniagents.messages import Message
from miniagents.miniagents import miniagent, InteractionContext

from versatilis_config import anthropic_agent


BASE_SETUP_FOLDER = Path("../talk-about-miniagents")


async def full_repo_agent(ctx: InteractionContext, setup_folder: str) -> None:
    """
    MiniAgent that receives the complete content of the MiniAgents project in its prompt.
    """
    system_header = (BASE_SETUP_FOLDER / setup_folder / "setup/system-header.md").read_text(encoding="utf-8")
    system_footer = (BASE_SETUP_FOLDER / setup_folder / "setup/system-footer.md").read_text(encoding="utf-8")

    ctx.reply(
        anthropic_agent.inquire(
            [
                Message(text=system_header, role="system"),
                _INITIAL_PROMPT,
                Message(text=system_footer, role="system"),
                ctx.messages,
            ],
            model="claude-3-haiku-20240307",
            # model="claude-3-sonnet-20240229",
            # model="claude-3-opus-20240229",
            # model="gpt-4o-2024-05-13",
            max_tokens=1500,
            temperature=0.0,
        )
    )


# noinspection PyTypeChecker
soul_crusher = miniagent(wraps(full_repo_agent)(partial(full_repo_agent, setup_folder="soul-crusher")))


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
