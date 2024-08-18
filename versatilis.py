"""
Versatilis is a tool for multi-turn conversations with multiple Large Language Models simultaneously.
Optionally, multiple files can be provided as context for the conversation.
"""

import hashlib
from pathlib import Path
from typing import Optional, Sequence, Union

import click
from dotenv import load_dotenv
from miniagents import InteractionContext, Message, MiniAgents, miniagent
from miniagents.ext import MarkdownHistoryAgent, console_user_agent, dialog_loop, markdown_llm_logger_agent
from miniagents.ext.llms import AnthropicAgent, AssistantMessage, OpenAIAgent
from pypdf import PdfReader

load_dotenv()

VERSATILIS_FOLDER = Path.home() / ".versatilis"

CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20240620"
GPT_4O = "gpt-4o-2024-08-06"

FAVOURITE_MODEL = CLAUDE_3_5_SONNET

MAX_OUTPUT_TOKENS = 4096

MODEL_AGENT_FACTORIES = {
    CLAUDE_3_5_SONNET: AnthropicAgent.fork(max_tokens=MAX_OUTPUT_TOKENS, stop_sequences=["</model>"]),
    "claude-3-opus-20240229": AnthropicAgent.fork(max_tokens=MAX_OUTPUT_TOKENS, stop_sequences=["</model>"]),
    "claude-3-haiku-20240307": AnthropicAgent.fork(max_tokens=MAX_OUTPUT_TOKENS, stop_sequences=["</model>"]),
    GPT_4O: OpenAIAgent.fork(stop=["</model>"]),
    "gpt-4-turbo-2024-04-09": OpenAIAgent.fork(stop=["</model>"]),
    "gpt-4o-mini-2024-07-18": OpenAIAgent.fork(stop=["</model>"]),
}
MODEL_AGENTS = {
    model: MODEL_AGENT_FACTORIES[model].fork(model=model, temperature=0)
    for model in [
        CLAUDE_3_5_SONNET,
        GPT_4O,
    ]
}


class ModelAwareMessage(Message):
    """
    A message class that includes model information in its string representation.

    When converted to a string, this message will be wrapped in XML-like tags that include the model name.
    If the message has content, it will be formatted as:

        <model {model_name}>{message_content}</model>

    If no model is specified or there is no content, it behaves like a regular Message.
    """

    model: Optional[str] = None

    @property
    def is_wrapped_with_model_tag(self) -> bool:
        """
        Whether the message is (or should be) wrapped with a model tag.
        """
        return bool(self.model and self.content and self.content.strip())

    def _as_string(self) -> str:
        if self.is_wrapped_with_model_tag:
            return f"<model {self.model}>{self.content}</model>"
        return super()._as_string()


@miniagent
async def versatilis(ctx: InteractionContext) -> None:
    """
    The main agent that handles the conversation using multiple Large Language Models.
    """
    incoming_messages = await ctx.message_promises

    append_model_tag = False
    for incoming_message in incoming_messages:
        if getattr(incoming_message, "is_wrapped_with_model_tag", False):
            # the model will see some of the previous dialog turns wrapped with <model></model> se we need
            # to make sure it will not start the new response with another <model>
            append_model_tag = True
            break

    for idx, (model, model_agent) in enumerate(MODEL_AGENTS.items()):
        console_style = "36;1" if idx % 2 == 1 else None

        prompt_messages = incoming_messages
        if append_model_tag:
            # let's make our model think that it already generated the <model> tag
            # (so it doesn't actually generate it)
            # TODO Oleksandr: make it possible to read the model name directly from the MiniAgent
            prompt_messages = (*prompt_messages, AssistantMessage(f"<model {model}>"))

        ctx.reply(
            model_agent.inquire(
                prompt_messages,
                system="NEVER START YOUR RESPONSE WITH <model>",
                response_metadata={"console_style": console_style},
            )
        )


def adapt_file_for_prompt(file_path: Union[str, Path]) -> str:
    """
    Converts a file into a string that can be used in the prompt.
    """
    file_path = Path(file_path)

    if file_path.suffix.lower() == ".pdf":
        reader = PdfReader(file_path)
        file_content = "\n\n".join(page.extract_text() for page in reader.pages)
    else:
        file_content = file_path.read_text(encoding="utf-8")

    file_for_prompt = f"<file path={str(file_path)!r}>{file_content}</file>"
    return file_for_prompt


async def conversation_loop(
    file_paths: Sequence[Union[str, Path]], chat_md: Optional[Union[str, Path]] = None
) -> None:
    """
    The main conversation loop.
    """
    if chat_md:
        base_dir = Path(chat_md).parent
    elif len(file_paths) == 1:
        base_dir = Path(file_paths[0]).parent
    else:
        base_dir = Path.cwd()

    relative_file_paths = "\n".join(sorted(str(Path(file_path).relative_to(base_dir)) for file_path in file_paths))

    if chat_md:
        chat_md_path = Path(chat_md)
    else:
        if len(file_paths) == 1:
            chat_md_prefix = f"{file_paths[0]}."
        elif len(file_paths) > 1:
            chat_md_prefix = (
                f"MULTI_FILES_{hashlib.sha256(relative_file_paths.encode(encoding='utf-8')).hexdigest()[:8]}"
            )
        else:
            chat_md_prefix = ""

        chat_md_path = Path(f"{chat_md_prefix}CHAT.md")

    if relative_file_paths and (not chat_md_path.exists() or chat_md_path.stat().st_size == 0):
        chat_md_path.write_text(
            f"\ncontext\n========================================\n```\n{relative_file_paths}\n```\n", encoding="utf-8"
        )

    files_in_prompt = [adapt_file_for_prompt(file_path) for file_path in file_paths]
    for file_in_prompt in files_in_prompt:
        print()
        print(file_in_prompt)

    dialog_loop.kick_off(
        files_in_prompt,
        user_agent=console_user_agent.fork(
            # write chat history to a markdown file
            history_agent=MarkdownHistoryAgent.fork(
                history_md_file=str(chat_md_path),
                # The value of `history_message_factory` is "unfreezable", hence we need to pass it via `mutable_state`
                mutable_state={"history_message_factory": ModelAwareMessage},
            )
        ),
        assistant_agent=versatilis,
    )


@click.command(
    help=(
        "Have a multi-turn conversation with multiple Large Language Models simultaneously. "
        "Optionally, provide a list of file paths (FILE_PATHS) to include as context of the conversation "
        "(for LLMs the contents of those files will appear at the top, before all the conversation turns)."
    ),
)
@click.argument(
    "file_paths",
    nargs=-1,
    type=click.Path(exists=True),
)
@click.option(
    "-c",
    "--chat-md",
    type=click.Path(),
    help="Path to the chat history markdown file (if not provided, default file name will be used).",
)
def main(file_paths: Sequence[str], chat_md: Optional[str] = None) -> None:
    """
    Run the conversation loop between the user and multiple models.
    """
    MiniAgents(
        llm_logger_agent=markdown_llm_logger_agent.fork(log_folder=str(VERSATILIS_FOLDER / "llm_logs")),
        # log_reduced_tracebacks=False,
    ).run(conversation_loop(file_paths, chat_md))


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
