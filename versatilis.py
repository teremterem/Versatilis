"""
A conversation example between the user and multiple LLMs using the MiniAgents framework.
"""

import hashlib
from pathlib import Path
from typing import Optional, Sequence, Union

import click
from dotenv import load_dotenv
from miniagents import InteractionContext, Message, MiniAgent, MiniAgents, miniagent
from miniagents.ext import MarkdownHistoryAgent, console_user_agent, dialog_loop, markdown_llm_logger_agent
from miniagents.ext.llms import AnthropicAgent, AssistantMessage, OpenAIAgent
from pypdf import PdfReader

load_dotenv()

VERSATILIS_FOLDER = Path.home() / ".versatilis"

GPT_4O = "gpt-4o-2024-08-06"
CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20240620"

FAVOURITE_MODEL = CLAUDE_3_5_SONNET

MAX_OUTPUT_TOKENS = 4096

MODEL_AGENT_FACTORIES = {
    GPT_4O: OpenAIAgent.fork(stop=["</model>"]),
    "gpt-4-turbo-2024-04-09": OpenAIAgent.fork(stop=["</model>"]),
    "gpt-4o-mini-2024-07-18": OpenAIAgent.fork(stop=["</model>"]),
    CLAUDE_3_5_SONNET: AnthropicAgent.fork(max_tokens=MAX_OUTPUT_TOKENS, stop_sequences=["</model>"]),
    "claude-3-opus-20240229": AnthropicAgent.fork(max_tokens=MAX_OUTPUT_TOKENS, stop_sequences=["</model>"]),
    "claude-3-haiku-20240307": AnthropicAgent.fork(max_tokens=MAX_OUTPUT_TOKENS, stop_sequences=["</model>"]),
}
MODEL_AGENTS = {
    model: MODEL_AGENT_FACTORIES[model].fork(model=model, temperature=0)
    for model in [
        GPT_4O,
        CLAUDE_3_5_SONNET,
    ]
}
FAVOURITE_MODEL_AGENT = MODEL_AGENTS[FAVOURITE_MODEL]
ALT_MODEL_AGENTS = {model: MODEL_AGENTS[model] for model in MODEL_AGENTS if model != FAVOURITE_MODEL}


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
    This agent employs many models to answer to the user. The answers of the "favourite" model are considered part of
    the "official" chat history, while the answers of the other models are just written to separate markdown files.
    """
    incoming_messages = await ctx.message_promises

    append_model_tag = False
    for incoming_message in incoming_messages:
        if getattr(incoming_message, "is_wrapped_with_model_tag", False):
            # the model will see some of the previous dialog turns wrapped with <model></model> se we need
            # to make sure it will not start the new response with another <model>
            append_model_tag = True
            break

    def run_model(model: str, model_agent: MiniAgent, **kwargs) -> None:
        prompt_messages = incoming_messages
        if append_model_tag:
            # let's make our model think that it already generated the <model> tag
            # (so it doesn't actually generate it)
            # TODO Oleksandr: make it possible to read the model name directly from the MiniAgent
            prompt_messages = (*prompt_messages, AssistantMessage(f"<model {model}>"))

        ctx.reply(model_agent.inquire(prompt_messages, system="NEVER START YOUR RESPONSE WITH <model>", **kwargs))

    run_model(FAVOURITE_MODEL, FAVOURITE_MODEL_AGENT)

    for idx, (model, model_agent) in enumerate(ALT_MODEL_AGENTS.items()):
        console_style = "36;1" if idx % 2 == 0 else None

        run_model(model, model_agent, response_metadata={"console_style": console_style})


def adapt_file_for_prompt(file_path: Union[str, Path]) -> str:
    """
    Converts a file into a string that can be used as a prompt for a model.
    """
    file_path = Path(file_path)

    if file_path.suffix.lower() == ".pdf":
        reader = PdfReader(file_path)
        file_content = "\n\n".join(page.extract_text() for page in reader.pages)
    else:
        file_content = file_path.read_text(encoding="utf-8")

    file_for_prompt = f"<file path={str(file_path)!r}>{file_content}</file>"
    return file_for_prompt


async def amain(file_paths: Sequence[Union[str, Path]]) -> None:
    """
    The main conversation loop.
    """
    absolute_file_paths = "\n".join(sorted(str(Path(file_path).absolute()) for file_path in file_paths))

    if len(file_paths) == 1:
        file_path = file_paths[0]
        file_path_prefix = f"{file_path}."
    elif len(file_paths) > 1:
        file_paths_hash = hashlib.sha256(absolute_file_paths.encode(encoding="utf-8")).hexdigest()
        file_path_prefix = f"MULTI_FILES_{file_paths_hash[:8]}."
    else:
        file_path_prefix = ""
    history_md_file_path = Path(f"{file_path_prefix}CHAT.md")

    if file_paths and not history_md_file_path.exists():
        history_md_file_path.write_text(
            f"\ncontext\n========================================\n```\n{absolute_file_paths}\n```\n", encoding="utf-8"
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
                history_md_file=str(history_md_file_path),
                # The value of `history_message_factory` is "unfreezable", hence we need to pass it via `mutable_state`
                mutable_state={"history_message_factory": ModelAwareMessage},
            )
        ),
        assistant_agent=versatilis,
    )


@click.command()
@click.argument("file_paths", nargs=-1, type=click.Path(exists=True))
def main(file_paths: Sequence[str]) -> None:
    """
    The main conversation loop.

    FILE_PATHS: One or more file paths to process.
    """
    MiniAgents(
        llm_logger_agent=markdown_llm_logger_agent.fork(log_folder=str(VERSATILIS_FOLDER / "llm_logs")),
        # log_reduced_tracebacks=False,
    ).run(amain(file_paths))


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
