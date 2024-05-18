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
                FullRepoMessage.create(),
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


class RepoFileMessage(Message):
    """
    A message that represents a file in the MiniAgents repository.
    """

    file_posix_path: str

    def _as_string(self) -> str:
        return f"{self.file_posix_path}\n```\n{self.text}\n```"


class FullRepoMessage(Message):
    """
    A message that represents the full content of the MiniAgents repository.
    """

    repo_files: tuple[RepoFileMessage, ...]

    @classmethod
    def create(cls) -> "FullRepoMessage":
        """
        Create a FullRepoMessage object that contains the full content of the MiniAgents repository. (Take a snapshot
        of the files as they currently are, in other words.)
        """
        miniagents_dir = Path("../MiniAgents")
        miniagent_files = [
            (file.relative_to(miniagents_dir).as_posix(), file)
            for file in miniagents_dir.rglob("*")
            if file.is_file() and file.stat().st_size > 0
        ]
        miniagent_files = [
            RepoFileMessage(file_posix_path=file_posix_path, text=file.read_text(encoding="utf-8"))
            for file_posix_path, file in miniagent_files
            if (
                not any(file_posix_path.startswith(prefix) for prefix in [".", "venv/", "dist/", "htmlcov/"])
                and not any(file_posix_path.endswith(suffix) for suffix in [".pyc"])
                and not any(file_posix_path in full_path for full_path in ["poetry.lock"])
            )
        ]
        miniagent_files.sort(key=lambda file_message: file_message.file_posix_path)
        return cls(repo_files=miniagent_files)

    def _as_string(self) -> str:
        miniagent_files_str = "\n".join([file_message.file_posix_path for file_message in self.repo_files])

        return "\n\n\n\n".join(
            [
                f"```\n{miniagent_files_str}\n```",
                *[str(file_message) for file_message in self.repo_files],
            ]
        )
