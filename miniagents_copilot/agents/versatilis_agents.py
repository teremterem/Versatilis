"""
TODO Oleksandr: figure out the role of this module
"""

from pathlib import Path

from miniagents.messages import Message
from miniagents.miniagents import miniagent, InteractionContext

from miniagents_copilot.agents.history_agents import fetch_history
from versatilis_config import openai_agent

BASE_SETUP_FOLDER = (Path(__file__).parent / "../../../talk-about-miniagents").resolve()

ANSWERS_FILE_NAME = "ANSWERS.md"

CHAT_FILE = BASE_SETUP_FOLDER / "CHAT.md"
ANSWERS_FILE = BASE_SETUP_FOLDER / ANSWERS_FILE_NAME

CLAUDE_HAIKU = "claude-3-haiku-20240307"
CLAUDE_SONNET = "claude-3-sonnet-20240229"
CLAUDE_OPUS = "claude-3-opus-20240229"
GPT_4O = "gpt-4o-2024-05-13"

RESEARCHER_MODEL = GPT_4O
ANSWERER_MODEL = CLAUDE_OPUS


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
    current_model=RESEARCHER_MODEL,
)
documenter = miniagent(
    full_repo_agent,  # TODO Oleksandr: figure out why the type checker is not happy with this parameter
    agent_folder=BASE_SETUP_FOLDER / "documenter",
    current_model=ANSWERER_MODEL,
)
researcher = miniagent(
    full_repo_agent,  # TODO Oleksandr: figure out why the type checker is not happy with this parameter
    agent_folder=BASE_SETUP_FOLDER / "researcher",
    current_model=RESEARCHER_MODEL,
)
research_planner = miniagent(
    full_repo_agent,  # TODO Oleksandr: figure out why the type checker is not happy with this parameter
    agent_folder=BASE_SETUP_FOLDER / "research-planner",
    current_model=RESEARCHER_MODEL,
)
answerer = miniagent(
    full_repo_agent,  # TODO Oleksandr: figure out why the type checker is not happy with this parameter
    agent_folder=BASE_SETUP_FOLDER / "answerer",
    current_model=ANSWERER_MODEL,
)


@miniagent
async def versatilis_agent(ctx: InteractionContext) -> None:
    """
    The main MiniAgent that orchestrates the conversation between the user and the Versatilis sub-agents.
    """
    chat_history = await fetch_history(history_file=CHAT_FILE)

    if ANSWERS_FILE.exists():
        # if answers file exists, then we are in "answerer" mode
        answers_history = await fetch_history(history_file=ANSWERS_FILE)
        ctx.reply(
            researcher.inquire(
                [
                    # in the chat history answerer sees assistant as user and user as assistant
                    role_inversion_agent.inquire(chat_history),
                    answers_history,  # in the answerer portion of the chat history roles are not flipped
                ]
            )
        )
    else:
        # otherwise, we are in "researcher" mode
        ctx.reply(researcher.inquire(chat_history))


@miniagent
async def versatilis_answerer(ctx: InteractionContext) -> None:
    """
    While researcher agent asks questions, answerer agent tries to answer them instead of the user (it sees the
    roles in the message history as inverted).
    """
    ctx.reply(answerer.inquire(role_inversion_agent.inquire(ctx.messages)))


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
                f"File list:\n```\n{miniagent_files_str}\n```",
                *[str(file_message) for file_message in self.repo_files],
            ]
        )
