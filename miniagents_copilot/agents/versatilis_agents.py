"""
TODO Oleksandr: figure out the role of this module
"""

from pathlib import Path

from miniagents.messages import Message
from miniagents.miniagents import miniagent, InteractionContext

from miniagents_copilot.agents.history_agents import fetch_history
from versatilis_config import openai_agent

BASE_SETUP_FOLDER = (Path(__file__).parent / "../../../talk-about-miniagents").resolve()

# MODEL = "claude-3-haiku-20240307"
# MODEL = "claude-3-sonnet-20240229"
# MODEL = "claude-3-opus-20240229"
MODEL = "gpt-4o-2024-05-13"


async def full_repo_agent(ctx: InteractionContext, agent_folder: Path, current_model: str) -> None:
    """
    MiniAgent that receives the complete content of the MiniAgents project in its prompt.
    """
    system_header = (agent_folder / "system-header.md").read_text(encoding="utf-8")
    system_footer = (agent_folder / "system-footer.md").read_text(encoding="utf-8")

    full_repo_message = FullRepoMessage.create()
    full_repo_md_file = BASE_SETUP_FOLDER / "transient/full-repo.md"
    full_repo_md_file.parent.mkdir(parents=True, exist_ok=True)
    full_repo_md_file.write_text(str(full_repo_message), encoding="utf-8")

    ctx.reply(
        openai_agent.inquire(
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
    agent_folder=BASE_SETUP_FOLDER / "soul-crusher",
    current_model=MODEL,
)
documenter = miniagent(
    full_repo_agent,  # TODO Oleksandr: figure out why the type checker is not happy with this parameter
    agent_folder=BASE_SETUP_FOLDER / "documenter",
    current_model=MODEL,
)
researcher = miniagent(
    full_repo_agent,  # TODO Oleksandr: figure out why the type checker is not happy with this parameter
    agent_folder=BASE_SETUP_FOLDER / "researcher",
    current_model=MODEL,
)
research_planner = miniagent(
    full_repo_agent,  # TODO Oleksandr: figure out why the type checker is not happy with this parameter
    agent_folder=BASE_SETUP_FOLDER / "research-planner",
    current_model=MODEL,
)
answerer = miniagent(
    full_repo_agent,  # TODO Oleksandr: figure out why the type checker is not happy with this parameter
    agent_folder=BASE_SETUP_FOLDER / "answerer",
    current_model=MODEL,
)


@miniagent
async def versatilis_agent(ctx: InteractionContext) -> None:
    """
    The main MiniAgent that orchestrates the conversation between the user and the Versatilis sub-agents.
    """
    ctx.reply(researcher.inquire(ctx.messages))


@miniagent
async def versatilis_answerer(ctx: InteractionContext) -> None:
    """
    While researcher agent asks questions, answerer agent tries to answer them instead of the user (it sees the
    roles in the message history as inverted).
    """
    chat_history = await fetch_history(file_name="CHAT.md")
    ctx.reply(answerer.inquire(role_inversion_agent.inquire(chat_history)))


@miniagent
async def role_inversion_agent(ctx: InteractionContext) -> None:
    """
    MiniAgent that inverts roles of incoming messages and replies with them.
    """
    async for message_promise in ctx.messages:
        # TODO Oleksandr: how to influence message attributes without breaking the token stream with this await ?
        #  (try to accomplish it right here with the tools you already have first, before introducing new ones)
        message = await message_promise
        if getattr(message, "role", None) in ["user", "assistant"]:
            message = Message(
                # TODO Oleksandr: this looks cumbersome
                **message.model_dump(exclude={"role"}),
                role="assistant" if message.role == "user" else "user",
            )
        ctx.reply(message)


class RepoFileMessage(Message):
    """
    A message that represents a file in the MiniAgents repository.
    """

    file_posix_path: str

    def _as_string(self) -> str:
        snippet_type = "python" if self.file_posix_path.endswith(".py") else ""
        extra_newline = "" if self.text.endswith("\n") else "\n"
        return f"{self.file_posix_path}\n```{snippet_type}\n{self.text}{extra_newline}```"


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
                f"File list:```\n{miniagent_files_str}\n```",
                *[str(file_message) for file_message in self.repo_files],
            ]
        )
