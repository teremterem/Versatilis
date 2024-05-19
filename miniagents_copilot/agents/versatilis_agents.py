"""
TODO Oleksandr: figure out the role of this module
"""

from pathlib import Path

from miniagents.messages import Message
from miniagents.miniagents import miniagent, InteractionContext

from versatilis_config import anthropic_agent

BASE_SETUP_FOLDER = Path("../talk-about-miniagents")
SOUL_CRUSHER_FOLDER = BASE_SETUP_FOLDER / "soul-crusher"

# MODEL = "gpt-4o-2024-05-13"
# MODEL = "claude-3-opus-20240229"
# MODEL = "claude-3-sonnet-20240229"
MODEL = "claude-3-haiku-20240307"


async def full_repo_agent(ctx: InteractionContext, agent_folder: Path, current_model: str) -> None:
    """
    MiniAgent that receives the complete content of the MiniAgents project in its prompt.
    """
    system_header = (agent_folder / "setup/system-header.md").read_text(encoding="utf-8")
    system_footer = (agent_folder / "setup/system-footer.md").read_text(encoding="utf-8")

    full_repo_message = FullRepoMessage.create()
    full_repo_md_file = agent_folder / "transient/full-repo.md"
    full_repo_md_file.parent.mkdir(parents=True, exist_ok=True)
    full_repo_md_file.write_text(str(full_repo_message), encoding="utf-8")

    ctx.reply(
        anthropic_agent.inquire(
            [
                Message(text=system_header, role="system"),
                full_repo_message,
                Message(text=system_footer, role="system"),
                ctx.messages,
            ],
            model=current_model,
            max_tokens=1500,
            temperature=0.0,
        )
    )


soul_crusher = miniagent(
    full_repo_agent,  # TODO Oleksandr: figure out why the type checker is not happy with this parameter
    agent_folder=SOUL_CRUSHER_FOLDER,
    current_model=MODEL,
)


@miniagent(
    agent_folder=SOUL_CRUSHER_FOLDER,
    current_model=MODEL,  # TODO Oleksandr: fix `split_messages()` so `model` could be read from resulting messages
)
async def history_agent(ctx: InteractionContext, agent_folder: Path, current_model: str) -> None:
    """
    TODO Oleksandr: docstring
    """
    chat_history_file = agent_folder / "CHAT.md"
    last_role = None
    with chat_history_file.open("w", encoding="utf-8") as chat_history:
        async for message_promise in ctx.messages:
            message = await message_promise

            role = getattr(message, "role", None) or "user"
            if role == "assistant":
                role = current_model

            if role != last_role:
                if last_role is not None:
                    chat_history.write("\n")
                chat_history.write(f"{role}\n-------------------------------")
                last_role = role

            chat_history.write(f"\n{message}\n")
            chat_history.flush()


class RepoFileMessage(Message):
    """
    A message that represents a file in the MiniAgents repository.
    """

    file_posix_path: str

    def _as_string(self) -> str:
        extra_newline = "" if self.text.endswith("\n") else "\n"
        return f"{self.file_posix_path}\n```\n{self.text}{extra_newline}```"


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
