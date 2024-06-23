File list:
```
miniagents/ext/llm/anthropic.py
miniagents/ext/llm/llm_common.py
miniagents/ext/llm/openai.py
miniagents/messages.py
miniagents/miniagent_typing.py
miniagents/miniagents.py
miniagents/promising/errors.py
miniagents/promising/ext/frozen.py
miniagents/promising/promise_typing.py
miniagents/promising/promising.py
miniagents/promising/sentinels.py
miniagents/promising/sequence.py
miniagents/utils.py
pyproject.toml
tests/test_agents.py
tests/test_frozen.py
tests/test_llm.py
tests/test_message.py
tests/test_message_sequence.py
tests/test_promise.py
tests/test_sequence.py
```



miniagents/ext/llm/anthropic.py
```python
import logging
import typing
from pprint import pformat
from typing import AsyncIterator, Any, Optional

from anthropic import NOT_GIVEN

from miniagents.ext.llm.llm_common import message_to_llm_dict, AssistantMessage
from miniagents.miniagents import (
    miniagent,
    MiniAgent,
    MiniAgents,
    InteractionContext,
)

if typing.TYPE_CHECKING:
    import anthropic as anthropic_original

logger = logging.getLogger(__name__)


class AnthropicMessage(AssistantMessage):
    ...


def create_anthropic_agent(
    async_client: Optional["anthropic_original.AsyncAnthropic"] = None,
    reply_metadata: Optional[dict[str, Any]] = None,
    alias: str = "ANTHROPIC_AGENT",
    **mini_agent_kwargs,
) -> MiniAgent:
    if not async_client:
        import anthropic as anthropic_original

        async_client = anthropic_original.AsyncAnthropic()

    return miniagent(
        _anthropic_func,
        async_client=async_client,
        global_reply_metadata=reply_metadata,
        alias=alias,
        **mini_agent_kwargs,
    )


async def _anthropic_func(
    ctx: InteractionContext,
    async_client: "anthropic_original.AsyncAnthropic",
    global_reply_metadata: Optional[dict[str, Any]],
    reply_metadata: Optional[dict[str, Any]] = None,
    stream: Optional[bool] = None,
    system: Optional[str] = None,
    fake_first_user_message: str = "/start",
    message_delimiter_for_same_role: str = "\n\n",
    **kwargs,
) -> None:
    if stream is None:
        stream = MiniAgents.get_current().stream_llm_tokens_by_default

    async def message_token_streamer(metadata_so_far: dict[str, Any]) -> AsyncIterator[str]:
        resolved_messages = await ctx.messages.aresolve_messages()

        message_dicts = [message_to_llm_dict(msg) for msg in resolved_messages]
        message_dicts = _fix_message_dicts(
            message_dicts,
            fake_first_user_message=fake_first_user_message,
            message_delimiter_for_same_role=message_delimiter_for_same_role,
        )

        if message_dicts and message_dicts[-1]["role"] == "system":
            system_message_dict = message_dicts.pop()
            system_combined = (
                system_message_dict["content"]
                if system is None
                else f"{system}{message_delimiter_for_same_role}{system_message_dict['content']}"
            )
        else:
            system_combined = system

        if system_combined is None:
            system_combined = NOT_GIVEN

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "SENDING TO ANTHROPIC:\n\n%s\nSYSTEM:\n%s\n", pformat(message_dicts), pformat(system_combined)
            )

        if stream:
            async with async_client.messages.stream(
                messages=message_dicts, system=system_combined, **kwargs
            ) as response:
                async for token in response.text_stream:
                    yield token
                anthropic_final_message = await response.get_final_message()
        else:
            anthropic_final_message = await async_client.messages.create(
                messages=message_dicts, stream=False, system=system_combined, **kwargs
            )
            if len(anthropic_final_message.content) != 1:
                raise RuntimeError(
                    f"exactly one TextBlock was expected from Anthropic, "
                    f"but {len(anthropic_final_message.content)} were returned instead"
                )
            yield anthropic_final_message.content[0].text

        metadata_so_far.update(anthropic_final_message.model_dump(exclude={"content"}))

    ctx.reply(
        AnthropicMessage.promise(
            start_asap=True,
            message_token_streamer=message_token_streamer,
            agent_alias=ctx.this_agent.alias,
            **(global_reply_metadata or {}),
            **(reply_metadata or {}),
        )
    )


def _fix_message_dicts(
    message_dicts: list[dict[str, Any]], fake_first_user_message: str, message_delimiter_for_same_role: str
) -> list[dict[str, Any]]:
    if not message_dicts:
        return []

    non_system_message_dicts = [message_dict for message_dict in message_dicts if message_dict["role"] != "system"]
    system_message_dicts = [message_dict for message_dict in message_dicts if message_dict["role"] == "system"]
    message_dicts = non_system_message_dicts + system_message_dicts

    fixed_message_dicts = []
    if message_dicts[0]["role"] != "user":
        fixed_message_dicts.append({"role": "user", "content": fake_first_user_message})

    for message_dict in message_dicts:
        if fixed_message_dicts and message_dict["role"] == fixed_message_dicts[-1]["role"]:
            fixed_message_dicts[-1]["content"] += message_delimiter_for_same_role + message_dict["content"]
        else:
            fixed_message_dicts.append(message_dict)

    return fixed_message_dicts
```



miniagents/ext/llm/llm_common.py
```python
from typing import Any, Optional

from miniagents.messages import Message


class UserMessage(Message):
    role: str = "user"


class SystemMessage(Message):
    role: str = "system"


class AssistantMessage(Message):
    role: str = "assistant"
    model: Optional[str] = None


def message_to_llm_dict(message: Message) -> dict[str, Any]:
    try:
        role = message.role
    except AttributeError:
        role = "user"

    return {
        "role": role,
        "content": str(message),
    }
```



miniagents/ext/llm/openai.py
```python
import logging
import typing
from pprint import pformat
from typing import AsyncIterator, Any, Optional

from miniagents.ext.llm.llm_common import message_to_llm_dict, AssistantMessage
from miniagents.miniagents import (
    miniagent,
    MiniAgent,
    MiniAgents,
    InteractionContext,
)

if typing.TYPE_CHECKING:
    import openai as openai_original

logger = logging.getLogger(__name__)


class OpenAIMessage(AssistantMessage):
    ...


def create_openai_agent(
    async_client: Optional["openai_original.AsyncOpenAI"] = None,
    reply_metadata: Optional[dict[str, Any]] = None,
    alias: str = "OPENAI_AGENT",
    **mini_agent_kwargs,
) -> MiniAgent:
    if not async_client:
        import openai as openai_original

        async_client = openai_original.AsyncOpenAI()

    return miniagent(
        _openai_func,
        async_client=async_client,
        global_reply_metadata=reply_metadata,
        alias=alias,
        **mini_agent_kwargs,
    )


async def _openai_func(
    ctx: InteractionContext,
    async_client: "openai_original.AsyncOpenAI",
    global_reply_metadata: Optional[dict[str, Any]],
    reply_metadata: Optional[dict[str, Any]] = None,
    stream: Optional[bool] = None,
    system: Optional[str] = None,
    n: int = 1,
    **kwargs,
) -> None:
    if stream is None:
        stream = MiniAgents.get_current().stream_llm_tokens_by_default

    if n != 1:
        raise ValueError("Only n=1 is supported by MiniAgents for AsyncOpenAI().chat.completions.create()")

    async def message_token_streamer(metadata_so_far: dict[str, Any]) -> AsyncIterator[str]:
        resolved_messages = await ctx.messages.aresolve_messages()

        if system is None:
            message_dicts = []
        else:
            message_dicts = [
                {
                    "role": "system",
                    "content": system,
                },
            ]
        message_dicts.extend(message_to_llm_dict(msg) for msg in resolved_messages)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("SENDING TO OPENAI:\n\n%s\n", pformat(message_dicts))

        openai_response = await async_client.chat.completions.create(messages=message_dicts, stream=stream, **kwargs)
        if stream:
            metadata_so_far["openai"] = {}
            async for chunk in openai_response:
                if len(chunk.choices) != 1:
                    raise RuntimeError(
                        f"exactly one Choice was expected from OpenAI, "
                        f"but {len(openai_response.choices)} were returned instead"
                    )
                token = chunk.choices[0].delta.content
                if token:
                    yield token

                metadata_so_far["role"] = chunk.choices[0].delta.role or metadata_so_far["role"]
                _merge_openai_dicts(
                    metadata_so_far,
                    chunk.model_dump(exclude={"choices": {0: {"index": ..., "delta": {"content": ..., "role": ...}}}}),
                )
        else:
            if len(openai_response.choices) != 1:
                raise RuntimeError(
                    f"exactly one Choice was expected from OpenAI, "
                    f"but {len(openai_response.choices)} were returned instead"
                )
            yield openai_response.choices[0].message.content

            metadata_so_far["role"] = openai_response.choices[0].message.role
            metadata_so_far.update(
                openai_response.model_dump(
                    exclude={"choices": {0: {"index": ..., "message": {"content": ..., "role": ...}}}}
                )
            )

    ctx.reply(
        OpenAIMessage.promise(
            start_asap=True,
            message_token_streamer=message_token_streamer,
            agent_alias=ctx.this_agent.alias,
            **(global_reply_metadata or {}),
            **(reply_metadata or {}),
        )
    )


def _merge_openai_dicts(destination_dict: dict[str, Any], dict_to_merge: dict[str, Any]) -> None:
    for key, value in dict_to_merge.items():
        if value is not None:
            existing_value = destination_dict.get(key)
            if isinstance(existing_value, dict):
                _merge_openai_dicts(existing_value, value)
            elif isinstance(existing_value, list):
                if key == "choices":
                    if not existing_value:
                        destination_dict[key] = [{}]
                    _merge_openai_dicts(destination_dict[key][0], value[0])
                else:
                    destination_dict[key].extend(value)
            else:
                destination_dict[key] = value
```



miniagents/messages.py
```python
from functools import cached_property
from typing import AsyncIterator, Any, Union, Optional, Iterator

from miniagents.miniagent_typing import MessageTokenStreamer
from miniagents.promising.ext.frozen import Frozen
from miniagents.promising.promising import StreamedPromise
from miniagents.promising.sentinels import Sentinel, DEFAULT


class Message(Frozen):
    text: Optional[str] = None
    text_template: Optional[str] = None

    @cached_property
    def as_promise(self) -> "MessagePromise":
        return MessagePromise(prefill_message=self)

    @classmethod
    def promise(
        cls,
        start_asap: Union[bool, Sentinel] = DEFAULT,
        message_token_streamer: Optional[MessageTokenStreamer] = None,
        **preliminary_metadata,
    ) -> "MessagePromise":
        if message_token_streamer:
            return MessagePromise(
                start_asap=start_asap,
                message_token_streamer=message_token_streamer,
                message_class=cls,
                **preliminary_metadata,
            )
        return cls(**preliminary_metadata).as_promise

    def serialize(self) -> dict[str, Any]:
        include_into_serialization, sub_messages = self._serialization_metadata
        model_dump = self.model_dump(include=include_into_serialization)

        for path, message_or_messages in sub_messages.items():
            sub_dict = model_dump
            for path_part in path[:-1]:
                sub_dict = sub_dict[path_part]
            if isinstance(message_or_messages, Message):
                sub_dict[f"{path[-1]}__hash_key"] = message_or_messages.hash_key
            else:
                sub_dict[f"{path[-1]}__hash_keys"] = tuple(message.hash_key for message in message_or_messages)
        return model_dump

    def sub_messages(self) -> Iterator["Message"]:
        _, sub_messages = self._serialization_metadata
        for _, message_or_messages in sub_messages.items():
            if isinstance(message_or_messages, Message):
                yield from message_or_messages.sub_messages()
                yield message_or_messages
            else:
                for message in message_or_messages:
                    yield from message.sub_messages()
                    yield message

    @cached_property
    def _serialization_metadata(
        self,
    ) -> tuple[
        dict[Union[str, int], Any],
        dict[tuple[Union[str, int], ...], Union["Message", tuple["Message", ...]]],
    ]:
        include_into_serialization = {}
        sub_messages = {}

        def build_serialization_metadata(
            inclusion_dict: dict[Union[str, int], Any],
            node: Frozen,
            node_path: tuple[Union[str, int], ...],
        ) -> None:
            for field, value in node.frozen_fields_and_values():
                if isinstance(value, Message):
                    sub_messages[(*node_path, field)] = value

                elif isinstance(value, Frozen):
                    sub_dict = {}
                    build_serialization_metadata(sub_dict, value, (*node_path, field))
                    inclusion_dict[field] = sub_dict

                elif isinstance(value, tuple):
                    if value and isinstance(value[0], Message):
                        sub_messages[(*node_path, field)] = value

                    else:
                        sub_dict = {}
                        for idx, sub_value in enumerate(value):
                            if isinstance(sub_value, Frozen):
                                sub_sub_dict = {}
                                build_serialization_metadata(sub_sub_dict, sub_value, (*node_path, field, idx))
                                sub_dict[idx] = sub_sub_dict
                            else:
                                sub_dict[idx] = ...
                        inclusion_dict[field] = sub_dict

                else:
                    inclusion_dict[field] = ...

        build_serialization_metadata(include_into_serialization, self, ())
        return include_into_serialization, sub_messages

    def _as_string(self) -> str:
        if self.text is not None:
            return self.text
        if self.text_template is not None:
            return self.text_template.format(**self.model_dump())
        return super()._as_string()

    def __init__(self, text: Optional[str] = None, **metadata: Any) -> None:
        super().__init__(text=text, **metadata)
        self._persist_message_event_triggered = False


class MessagePromise(StreamedPromise[str, Message]):
    preliminary_metadata: Frozen

    def __init__(
        self,
        start_asap: Union[bool, Sentinel] = DEFAULT,
        message_token_streamer: Optional[MessageTokenStreamer] = None,
        prefill_message: Optional[Message] = None,
        message_class: type[Message] = Message,
        **preliminary_metadata,
    ) -> None:
        if prefill_message:
            self.preliminary_metadata = prefill_message

            super().__init__(
                start_asap=start_asap,
                prefill_pieces=[str(prefill_message)],
                prefill_result=prefill_message,
            )
        else:
            self.preliminary_metadata = Frozen(**preliminary_metadata)
            self._metadata_so_far = dict(self.preliminary_metadata.frozen_fields_and_values(exclude_class=True))

            self._message_token_streamer = message_token_streamer
            self._message_class = message_class
            super().__init__(start_asap=start_asap)

    def _streamer(self) -> AsyncIterator[str]:
        return self._message_token_streamer(self._metadata_so_far)

    async def _resolver(self) -> Message:
        return self._message_class(
            text="".join([token async for token in self]),
            **self._metadata_so_far,
        )


class MessageSequencePromise(StreamedPromise[MessagePromise, tuple[MessagePromise, ...]]):
    async def aresolve_messages(self) -> tuple[Message, ...]:
        return tuple([await message_promise async for message_promise in self])

    def as_single_promise(self, **kwargs) -> MessagePromise:
        from miniagents.utils import join_messages

        return join_messages(self, start_asap=False, **kwargs)
```



miniagents/miniagent_typing.py
```python
import typing
from typing import AsyncIterator, Protocol, Union, Any, Iterable, AsyncIterable

from pydantic import BaseModel

if typing.TYPE_CHECKING:
    from miniagents.messages import Message, MessagePromise
    from miniagents.miniagents import InteractionContext


class AgentFunction(Protocol):
    async def __call__(self, ctx: "InteractionContext", **kwargs) -> None: ...


class MessageTokenStreamer(Protocol):
    def __call__(self, metadata_so_far: dict[str, Any]) -> AsyncIterator[str]: ...


SingleMessageType = Union[str, dict[str, Any], BaseModel, "Message", "MessagePromise", BaseException]
MessageType = Union[SingleMessageType, Iterable["MessageType"], AsyncIterable["MessageType"]]
```



miniagents/miniagents.py
```python
import asyncio
import copy
import logging
from functools import partial
from typing import Protocol, AsyncIterator, Any, Union, Optional, Callable, Iterable, Awaitable

from pydantic import BaseModel

from miniagents.messages import MessagePromise, MessageSequencePromise, Message
from miniagents.miniagent_typing import MessageType, AgentFunction
from miniagents.promising.ext.frozen import freeze_dict_values
from miniagents.promising.promise_typing import PromiseStreamer, PromiseBound, PromiseResolvedEventHandler
from miniagents.promising.promising import StreamAppender, Promise, PromisingContext
from miniagents.promising.sentinels import Sentinel, DEFAULT
from miniagents.promising.sequence import FlatSequence

logger = logging.getLogger(__name__)


class PersistMessageEventHandler(Protocol):
    async def __call__(self, promise: PromiseBound, message: Message) -> None: ...


class MiniAgents(PromisingContext):
    def __init__(
        self,
        stream_llm_tokens_by_default: bool = True,
        on_promise_resolved: Union[PromiseResolvedEventHandler, Iterable[PromiseResolvedEventHandler]] = (),
        on_persist_message: Union[PersistMessageEventHandler, Iterable[PersistMessageEventHandler]] = (),
        **kwargs,
    ) -> None:
        on_promise_resolved = (
            [self._trigger_persist_message_event, on_promise_resolved]
            if callable(on_promise_resolved)
            else [self._trigger_persist_message_event, *on_promise_resolved]
        )
        super().__init__(on_promise_resolved=on_promise_resolved, **kwargs)
        self.stream_llm_tokens_by_default = stream_llm_tokens_by_default
        self.on_persist_message_handlers: list[PersistMessageEventHandler] = (
            [on_persist_message] if callable(on_persist_message) else list(on_persist_message)
        )

    def run(self, awaitable: Awaitable[Any]) -> Any:
        return asyncio.run(self.arun(awaitable))

    async def arun(self, awaitable: Awaitable[Any]) -> Any:
        async with self:
            return await awaitable

    @classmethod
    def get_current(cls) -> "MiniAgents":
        return super().get_current()

    def on_persist_message(self, handler: PersistMessageEventHandler) -> PersistMessageEventHandler:
        self.on_persist_message_handlers.append(handler)
        return handler

    async def _trigger_persist_message_event(self, _, obj: Any) -> None:
        if not isinstance(obj, Message):
            return

        log_level_for_errors = MiniAgents.get_current().log_level_for_errors

        for sub_message in obj.sub_messages():
            if sub_message._persist_message_event_triggered:
                continue

            for handler in self.on_persist_message_handlers:
                self.start_asap(
                    handler(_, sub_message), suppress_errors=True, log_level_for_errors=log_level_for_errors
                )
            sub_message._persist_message_event_triggered = True

        if obj._persist_message_event_triggered:
            return

        for handler in self.on_persist_message_handlers:
            self.start_asap(handler(_, obj), suppress_errors=True, log_level_for_errors=log_level_for_errors)
        obj._persist_message_event_triggered = True


def miniagent(
    func: Optional[AgentFunction] = None,
    alias: Optional[str] = None,
    description: Optional[str] = None,
    uppercase_func_name: bool = True,
    normalize_spaces_in_docstring: bool = True,
    interaction_metadata: Optional[dict[str, Any]] = None,
    **partial_kwargs,
) -> Union["MiniAgent", Callable[[AgentFunction], "MiniAgent"]]:
    if func is None:
        def _decorator(f: AgentFunction) -> "MiniAgent":
            return MiniAgent(
                f,
                alias=alias,
                description=description,
                uppercase_func_name=uppercase_func_name,
                normalize_spaces_in_docstring=normalize_spaces_in_docstring,
                interaction_metadata=interaction_metadata,
                **partial_kwargs,
            )

        return _decorator

    return MiniAgent(
        func,
        alias=alias,
        description=description,
        uppercase_func_name=uppercase_func_name,
        normalize_spaces_in_docstring=normalize_spaces_in_docstring,
        interaction_metadata=interaction_metadata,
        **partial_kwargs,
    )


class InteractionContext:
    def __init__(
        self, this_agent: "MiniAgent", messages: MessageSequencePromise, reply_streamer: StreamAppender[MessageType]
    ) -> None:
        self.this_agent = this_agent
        self.messages = messages
        self._reply_streamer = reply_streamer

    def reply(self, messages: MessageType) -> None:
        self._reply_streamer.append(messages)

    def finish_early(self) -> None:
        self._reply_streamer.close()


class AgentCall:
    def __init__(
        self,
        message_streamer: StreamAppender[MessageType],
        reply_sequence_promise: MessageSequencePromise,
    ) -> None:
        self._message_streamer = message_streamer
        self._reply_sequence_promise = reply_sequence_promise

        self._message_streamer.open()

    def send_message(self, message: MessageType) -> "AgentCall":
        self._message_streamer.append(message)
        return self

    def reply_sequence(self) -> MessageSequencePromise:
        self.finish()
        return self._reply_sequence_promise

    def finish(self) -> "AgentCall":
        self._message_streamer.close()
        return self


class MiniAgent:
    def __init__(
        self,
        func: AgentFunction,
        alias: Optional[str] = None,
        description: Optional[str] = None,
        uppercase_func_name: bool = True,
        normalize_spaces_in_docstring: bool = True,
        interaction_metadata: Optional[dict[str, Any]] = None,
        **partial_kwargs,
    ) -> None:
        self._func = func
        if partial_kwargs:
            self._func = partial(func, **partial_kwargs)
        self.frozen_interact_metadata = freeze_dict_values(interaction_metadata or {})

        self.alias = alias
        if self.alias is None:
            self.alias = func.__name__
            if uppercase_func_name:
                self.alias = self.alias.upper()

        self.description = description
        if self.description is None:
            self.description = func.__doc__
            if self.description and normalize_spaces_in_docstring:
                self.description = " ".join(self.description.split())
        if self.description:
            self.description = self.description.format(AGENT_ALIAS=self.alias)

        self.__name__ = self.alias
        self.__doc__ = self.description

    def inquire(
        self,
        messages: Optional[MessageType] = None,
        start_asap: Union[bool, Sentinel] = DEFAULT,
        **function_kwargs,
    ) -> MessageSequencePromise:
        agent_call = self.initiate_inquiry(start_asap=start_asap, **function_kwargs)
        if messages is not None:
            agent_call.send_message(messages)
        return agent_call.reply_sequence()

    def initiate_inquiry(
        self,
        start_asap: Union[bool, Sentinel] = DEFAULT,
        **function_kwargs,
    ) -> "AgentCall":
        input_sequence = MessageSequence(
            start_asap=False,
        )
        reply_sequence = AgentReplyMessageSequence(
            mini_agent=self,
            function_kwargs=function_kwargs,
            input_sequence_promise=input_sequence.sequence_promise,
            start_asap=start_asap,
        )

        agent_call = AgentCall(
            message_streamer=input_sequence.message_appender,
            reply_sequence_promise=reply_sequence.sequence_promise,
        )
        return agent_call


class AgentInteractionNode(Message):
    agent_alias: str


class AgentCallNode(AgentInteractionNode):
    messages: tuple[Message, ...]


class AgentReplyNode(AgentInteractionNode):
    agent_call: AgentCallNode
    replies: tuple[Message, ...]


class MessageSequence(FlatSequence[MessageType, MessagePromise]):
    message_appender: Optional[StreamAppender[MessageType]]
    sequence_promise: MessageSequencePromise

    def __init__(
        self,
        appender_capture_errors: Union[bool, Sentinel] = DEFAULT,
        start_asap: Union[bool, Sentinel] = DEFAULT,
        incoming_streamer: Optional[PromiseStreamer[MessageType]] = None,
    ) -> None:
        if incoming_streamer:
            self.message_appender = None
        else:
            self.message_appender = StreamAppender(capture_errors=appender_capture_errors)
            incoming_streamer = self.message_appender

        super().__init__(
            incoming_streamer=incoming_streamer,
            start_asap=start_asap,
            sequence_promise_class=MessageSequencePromise,
        )

    @classmethod
    def turn_into_sequence_promise(cls, messages: MessageType) -> MessageSequencePromise:
        message_sequence = cls(
            appender_capture_errors=True,
            start_asap=False,
        )
        with message_sequence.message_appender:
            message_sequence.message_appender.append(messages)
        return message_sequence.sequence_promise

    @classmethod
    async def aresolve_messages(cls, messages: MessageType) -> tuple[Message, ...]:
        return await cls.turn_into_sequence_promise(messages).aresolve_messages()

    async def _flattener(
        self, zero_or_more_items: MessageType
    ) -> AsyncIterator[MessagePromise]:
        if isinstance(zero_or_more_items, MessagePromise):
            yield zero_or_more_items
        elif isinstance(zero_or_more_items, Message):
            yield zero_or_more_items.as_promise
        elif isinstance(zero_or_more_items, BaseModel):
            yield Message(**zero_or_more_items.model_dump()).as_promise
        elif isinstance(zero_or_more_items, dict):
            yield Message(**zero_or_more_items).as_promise
        elif isinstance(zero_or_more_items, str):
            yield Message(text=zero_or_more_items).as_promise
        elif isinstance(zero_or_more_items, BaseException):
            raise zero_or_more_items
        elif hasattr(zero_or_more_items, "__iter__"):
            for item in zero_or_more_items:
                async for message_promise in self._flattener(item):
                    yield message_promise
        elif hasattr(zero_or_more_items, "__aiter__"):
            async for item in zero_or_more_items:
                async for message_promise in self._flattener(item):
                    yield message_promise
        else:
            raise TypeError(f"Unexpected message type: {type(zero_or_more_items)}")


class AgentReplyMessageSequence(MessageSequence):
    def __init__(
        self,
        mini_agent: MiniAgent,
        input_sequence_promise: MessageSequencePromise,
        function_kwargs: dict[str, Any],
        **kwargs,
    ) -> None:
        self._frozen_func_kwargs = freeze_dict_values(function_kwargs)
        self._function_kwargs = copy.deepcopy(function_kwargs)

        self._mini_agent = mini_agent
        self._input_sequence_promise = input_sequence_promise
        super().__init__(
            appender_capture_errors=True,
            **kwargs,
        )

    async def _streamer(self, _) -> AsyncIterator[MessagePromise]:
        async def run_the_agent(_) -> AgentCallNode:
            ctx = InteractionContext(
                this_agent=self._mini_agent,
                messages=self._input_sequence_promise,
                reply_streamer=self.message_appender,
            )
            with self.message_appender:
                await self._mini_agent._func(ctx, **self._function_kwargs)

            return AgentCallNode(
                messages=await self._input_sequence_promise.aresolve_messages(),
                agent_alias=self._mini_agent.alias,
                **self._mini_agent.frozen_interact_metadata,
                **self._frozen_func_kwargs,
            )

        agent_call_promise = Promise[AgentCallNode](
            start_asap=True,
            resolver=run_the_agent,
        )

        async for reply_promise in super()._streamer(_):
            yield reply_promise

        async def create_agent_reply_node(_) -> AgentReplyNode:
            return AgentReplyNode(
                replies=await self.sequence_promise.aresolve_messages(),
                agent_alias=self._mini_agent.alias,
                agent_call=await agent_call_promise,
                **self._mini_agent.frozen_interact_metadata,
            )

        Promise[AgentReplyNode](
            start_asap=True,
            resolver=create_agent_reply_node,
        )
```



miniagents/promising/errors.py
```python
class PromisingError(Exception):
    ...


class FunctionNotProvidedError(PromisingError):
    ...


class AppenderNotOpenError(PromisingError):
    ...


class AppenderClosedError(PromisingError):
    ...
```



miniagents/promising/ext/frozen.py
```python
import hashlib
import itertools
import json
from functools import cached_property
from typing import Any, Iterator, Optional, Union

from pydantic import BaseModel, ConfigDict, model_validator

FrozenType = Optional[Union[str, int, float, bool, tuple["FrozenType", ...], "Frozen"]]


def freeze_dict_values(d: dict[str, Any]) -> dict[str, FrozenType]:
    return dict(Frozen(**d).frozen_fields_and_values(exclude_class=True))


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    class_: str

    def __str__(self) -> str:
        return self.as_string

    @cached_property
    def as_string(self) -> str:
        return self._as_string()

    @cached_property
    def full_json(self) -> str:
        return self.model_dump_json()

    @cached_property
    def serialized(self) -> str:
        return json.dumps(self.serialize(), ensure_ascii=False, sort_keys=True)

    def serialize(self) -> dict[str, Any]:
        return self.model_dump()

    @cached_property
    def hash_key(self) -> str:
        from miniagents.promising.promising import PromisingContext

        hash_key = hashlib.sha256(self.serialized.encode("utf-8")).hexdigest()
        if not PromisingContext.get_current().longer_hash_keys:
            hash_key = hash_key[:40]
        return hash_key

    def frozen_fields(self, exclude_class: bool = False) -> Iterator[str]:
        if exclude_class:
            return itertools.chain(
                (field for field in self.model_fields if field != "class_"), self.__pydantic_extra__
            )
        return itertools.chain(self.model_fields, self.__pydantic_extra__)

    def frozen_fields_and_values(self, exclude_class: bool = False) -> Iterator[tuple[str, Any]]:
        if exclude_class:
            for field in self.model_fields:
                if field != "class_":
                    yield field, getattr(self, field)
        else:
            for field in self.model_fields:
                yield field, getattr(self, field)

        for field, value in self.__pydantic_extra__.items():
            yield field, value

    def _as_string(self) -> str:
        return self.full_json

    @classmethod
    def _preprocess_values(cls, values: dict[str, Any]) -> dict[str, Any]:
        if "class_" in values:
            if values["class_"] != cls.__name__:
                raise ValueError(
                    f"the `class_` field of a Frozen must be equal to its actual class name, got {values['class_']} "
                    f"instead of {cls.__name__}"
                )
        else:
            values = {"class_": cls.__name__, **values}
        return values

    @model_validator(mode="before")
    @classmethod
    def _validate_and_freeze_values(cls, values: dict[str, Any]) -> dict[str, FrozenType]:
        values = cls._preprocess_values(values)
        return {key: cls._validate_and_freeze_value(key, value) for key, value in values.items()}

    @classmethod
    def _validate_and_freeze_value(cls, key: str, value: Any) -> FrozenType:
        if isinstance(value, (tuple, list)):
            return tuple(cls._validate_and_freeze_value(key, sub_value) for sub_value in value)
        if isinstance(value, dict):
            return Frozen(**value)
        if not isinstance(value, cls._allowed_value_types()):
            raise ValueError(
                f"only {{{', '.join([t.__name__ for t in cls._allowed_value_types()])}}} "
                f"are allowed as field values in {cls.__name__}, got {type(value).__name__} in `{key}`"
            )
        return value

    @classmethod
    def _allowed_value_types(cls) -> tuple[type[Any], ...]:
        return type(None), str, int, float, bool, tuple, list, dict, Frozen
```



miniagents/promising/promise_typing.py
```python
from typing import TypeVar, AsyncIterator, Protocol, Union, Any

T = TypeVar("T")
PIECE = TypeVar("PIECE")
WHOLE = TypeVar("WHOLE")
IN = TypeVar("IN")
OUT = TypeVar("OUT")
PromiseBound = TypeVar("PromiseBound", bound="Promise")
StreamedPromiseBound = TypeVar("StreamedPromiseBound", bound="StreamedPromise")
FlatSequenceBound = TypeVar("FlatSequenceBound", bound="FlatSequence")


class PromiseResolver(Protocol[T]):
    async def __call__(self, promise: PromiseBound) -> T: ...


class PromiseStreamer(Protocol[PIECE]):
    def __call__(self, streamed_promise: StreamedPromiseBound) -> AsyncIterator[PIECE]: ...


class PromiseResolvedEventHandler(Protocol):
    async def __call__(self, promise: PromiseBound, result: Any) -> None: ...


class SequenceFlattener(Protocol[IN, OUT]):
    def __call__(
        self, flat_sequence: FlatSequenceBound, zero_or_more_items: Union[IN, BaseException]
    ) -> AsyncIterator[OUT]: ...
```



miniagents/promising/promising.py
```python
import asyncio
import contextvars
import logging
from asyncio import Task
from contextvars import ContextVar
from functools import partial
from types import TracebackType
from typing import Generic, AsyncIterator, Union, Optional, Iterable, Awaitable, Any

from miniagents.promising.errors import AppenderClosedError, AppenderNotOpenError, FunctionNotProvidedError
from miniagents.promising.promise_typing import (
    T,
    PIECE,
    WHOLE,
    PromiseStreamer,
    PromiseResolvedEventHandler,
    PromiseResolver,
)
from miniagents.promising.sentinels import Sentinel, NO_VALUE, FAILED, END_OF_QUEUE, DEFAULT

logger = logging.getLogger(__name__)


class PromisingContext:
    _current: ContextVar[Optional["PromisingContext"]] = ContextVar("PromisingContext._current", default=None)

    def __init__(
        self,
        start_everything_asap_by_default: bool = True,
        appenders_capture_errors_by_default: bool = False,
        longer_hash_keys: bool = False,
        log_level_for_errors: int = logging.ERROR,
        on_promise_resolved: Union[PromiseResolvedEventHandler, Iterable[PromiseResolvedEventHandler]] = (),
    ) -> None:
        self.parent = self._current.get()

        self.on_promise_resolved_handlers: list[PromiseResolvedEventHandler] = (
            [on_promise_resolved] if callable(on_promise_resolved) else [*on_promise_resolved]
        )
        self.child_tasks: set[Task] = set()

        self.start_everything_asap_by_default = start_everything_asap_by_default
        self.appenders_capture_errors_by_default = appenders_capture_errors_by_default
        self.longer_hash_keys = longer_hash_keys
        self.log_level_for_errors = log_level_for_errors

        self._previous_ctx_token: Optional[contextvars.Token] = None

    @classmethod
    def get_current(cls) -> "PromisingContext":
        current = cls._current.get()
        if not current:
            raise RuntimeError(
                f"No {cls.__name__} is currently active. Did you forget to do `async with {cls.__name__}():`?"
            )
        if not isinstance(current, cls):
            raise TypeError(
                f"You seem to have done `async with {type(current).__name__}():` (or similar), "
                f"but `async with {cls.__name__}():` is expected instead."
            )
        return current

    def on_promise_resolved(self, handler: PromiseResolvedEventHandler) -> PromiseResolvedEventHandler:
        self.on_promise_resolved_handlers.append(handler)
        return handler

    def start_asap(
        self,
        awaitable: Awaitable,
        suppress_errors: bool = False,
        log_level_for_errors: int = logging.DEBUG,
    ) -> Task:
        async def awaitable_wrapper() -> Any:
            try:
                return await awaitable
            except Exception:
                logger.log(
                    log_level_for_errors,
                    "AN ERROR OCCURRED IN AN ASYNC BACKGROUND TASK",
                    exc_info=True,
                )
                if not suppress_errors:
                    raise
            except BaseException:
                if not suppress_errors:
                    raise
            finally:
                self.child_tasks.remove(task)

        task = asyncio.create_task(awaitable_wrapper())
        self.child_tasks.add(task)
        return task

    def activate(self) -> "PromisingContext":
        if self._previous_ctx_token:
            raise RuntimeError("PromisingContext is not reentrant")
        self._previous_ctx_token = self._current.set(self)
        return self

    async def aflush_tasks(self) -> None:
        while self.child_tasks:
            await asyncio.gather(
                *self.child_tasks,
                return_exceptions=True,
            )

    async def afinalize(self) -> None:
        await self.aflush_tasks()
        self._current.reset(self._previous_ctx_token)
        self._previous_ctx_token = None

    async def __aenter__(self) -> "PromisingContext":
        return self.activate()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.afinalize()


class Promise(Generic[T]):
    def __init__(
        self,
        start_asap: Union[bool, Sentinel] = DEFAULT,
        resolver: Optional[PromiseResolver[T]] = None,
        prefill_result: Union[Optional[T], Sentinel] = NO_VALUE,
    ) -> None:
        promising_context = PromisingContext.get_current()

        if start_asap is DEFAULT:
            start_asap = promising_context.start_everything_asap_by_default

        if resolver:
            self._resolver = partial(resolver, self)

        if prefill_result is NO_VALUE:
            self._result: Union[T, Sentinel, BaseException] = NO_VALUE
        else:
            self._result = prefill_result
            self._trigger_promise_resolved_event()

        self._resolver_lock = asyncio.Lock()

        if start_asap and prefill_result is NO_VALUE:
            promising_context.start_asap(
                self, suppress_errors=True, log_level_for_errors=promising_context.log_level_for_errors
            )

    async def _resolver(self) -> T:
        raise FunctionNotProvidedError(
            "The `resolver` function should be provided either via the constructor "
            "or by subclassing the `Promise` class."
        )

    async def aresolve(self) -> T:
        if self._result is NO_VALUE:
            async with self._resolver_lock:
                if self._result is NO_VALUE:
                    try:
                        self._result = await self._resolver()
                    except BaseException as exc:
                        logger.debug("An error occurred while resolving a Promise", exc_info=True)
                        self._result = exc

                    self._trigger_promise_resolved_event()

        if isinstance(self._result, BaseException):
            raise self._result
        return self._result

    def __await__(self):
        return self.aresolve().__await__()

    def _trigger_promise_resolved_event(self):
        promising_context = PromisingContext.get_current()
        while promising_context:
            for handler in promising_context.on_promise_resolved_handlers:
                promising_context.start_asap(
                    handler(self, self._result),
                    suppress_errors=True,
                    log_level_for_errors=promising_context.log_level_for_errors,
                )
            promising_context = promising_context.parent


class StreamedPromise(Generic[PIECE, WHOLE], Promise[WHOLE]):
    def __init__(
        self,
        streamer: Optional[PromiseStreamer[PIECE]] = None,
        prefill_pieces: Union[Optional[Iterable[PIECE]], Sentinel] = NO_VALUE,
        resolver: Optional[PromiseResolver[T]] = None,
        prefill_result: Union[Optional[T], Sentinel] = NO_VALUE,
        start_asap: Union[bool, Sentinel] = DEFAULT,
    ) -> None:
        promising_context = PromisingContext.get_current()

        if start_asap is DEFAULT:
            start_asap = promising_context.start_everything_asap_by_default

        super().__init__(
            start_asap=start_asap,
            resolver=resolver,
            prefill_result=prefill_result,
        )

        if streamer:
            self._streamer = partial(streamer, self)

        if prefill_pieces is NO_VALUE:
            self._pieces_so_far: list[Union[PIECE, BaseException]] = []
        else:
            self._pieces_so_far: list[Union[PIECE, BaseException]] = [*prefill_pieces, StopAsyncIteration()]

        self._all_pieces_consumed = prefill_pieces is not NO_VALUE
        self._streamer_lock = asyncio.Lock()

        if start_asap and prefill_pieces is NO_VALUE:
            self._queue = asyncio.Queue()
            promising_context.start_asap(
                self._aconsume_the_stream(),
                suppress_errors=True,
                log_level_for_errors=promising_context.log_level_for_errors,
            )
        else:
            self._queue = None

        self._streamer_aiter: Union[Optional[AsyncIterator[PIECE]], Sentinel] = None

    def _streamer(self) -> AsyncIterator[PIECE]:
        raise FunctionNotProvidedError(
            "The `streamer` function should be provided either via the constructor "
            "or by subclassing the `StreamedPromise` class."
        )

    def __aiter__(self) -> AsyncIterator[PIECE]:
        return self._StreamReplayIterator(self)

    def __call__(self, *args, **kwargs) -> AsyncIterator[PIECE]:
        return self.__aiter__()

    async def _aconsume_the_stream(self) -> None:
        while True:
            piece = await self._streamer_aiter_anext()
            self._queue.put_nowait(piece)
            if isinstance(piece, StopAsyncIteration):
                break

    async def _streamer_aiter_anext(self) -> Union[PIECE, BaseException]:
        if self._streamer_aiter is None:
            try:
                self._streamer_aiter = self._streamer()
                if not callable(self._streamer_aiter.__anext__):
                    raise TypeError("The streamer must return an async iterator")
            except BaseException as exc:
                logger.debug("An error occurred while instantiating a streamer for a StreamedPromise", exc_info=True)
                self._streamer_aiter = FAILED
                return exc

        elif self._streamer_aiter is FAILED:
            return StopAsyncIteration()

        try:
            return await self._streamer_aiter.__anext__()
        except BaseException as exc:
            if not isinstance(exc, StopAsyncIteration):
                logger.debug(
                    'An error occurred while fetching a single "piece" of a StreamedPromise from its pieces streamer.',
                    exc_info=True,
                )
            return exc

    class _StreamReplayIterator(AsyncIterator[PIECE]):
        def __init__(self, streamed_promise: "StreamedPromise") -> None:
            self._streamed_promise = streamed_promise
            self._index = 0

        async def __anext__(self) -> PIECE:
            if self._index < len(self._streamed_promise._pieces_so_far):
                piece = self._streamed_promise._pieces_so_far[self._index]
            elif self._streamed_promise._all_pieces_consumed:
                raise self._streamed_promise._pieces_so_far[-1]
            else:
                async with self._streamed_promise._streamer_lock:
                    if self._index < len(self._streamed_promise._pieces_so_far):
                        piece = self._streamed_promise._pieces_so_far[self._index]
                    else:
                        piece = await self._real_anext()

            self._index += 1

            if isinstance(piece, BaseException):
                raise piece
            return piece

        async def _real_anext(self) -> Union[PIECE, BaseException]:
            if self._streamed_promise._queue is None:
                piece = await self._streamed_promise._streamer_aiter_anext()
            else:
                piece = await self._streamed_promise._queue.get()

            if isinstance(piece, StopAsyncIteration):
                self._streamed_promise._all_pieces_consumed = True

            self._streamed_promise._pieces_so_far.append(piece)
            return piece


class StreamAppender(Generic[PIECE], AsyncIterator[PIECE]):
    def __init__(self, capture_errors: Union[bool, Sentinel] = DEFAULT) -> None:
        self._queue = asyncio.Queue()
        self._append_open = False
        self._append_closed = False
        if capture_errors is DEFAULT:
            self._capture_errors = PromisingContext.get_current().appenders_capture_errors_by_default
        else:
            self._capture_errors = capture_errors

    def __enter__(self) -> "StreamAppender":
        return self.open()

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool:
        is_append_closed_error = isinstance(exc_value, AppenderClosedError)
        error_should_not_propagate = self._capture_errors and not is_append_closed_error

        if exc_value and error_should_not_propagate:
            logger.debug("An error occurred while appending pieces to a StreamAppender", exc_info=exc_value)
            self.append(exc_value)
        self.close()

        return error_should_not_propagate

    def append(self, piece: PIECE) -> "StreamAppender":
        if not self._append_open:
            raise AppenderNotOpenError(
                "You need to put the `append()` operation inside a `with StreamAppender()` block "
                "(or call `open()` and `close()` manually)."
            )
        if self._append_closed:
            raise AppenderClosedError("The StreamAppender has already been closed for appending.")
        self._queue.put_nowait(piece)
        return self

    def open(self) -> "StreamAppender":
        if self._append_closed:
            raise AppenderClosedError("Once closed, the StreamAppender cannot be opened again.")
        self._append_open = True
        return self

    def close(self) -> None:
        if self._append_closed:
            return
        self._append_closed = True
        self._queue.put_nowait(END_OF_QUEUE)

    async def __anext__(self) -> PIECE:
        if self._queue is None:
            raise StopAsyncIteration()

        piece = await self._queue.get()
        if piece is END_OF_QUEUE:
            self._queue = None
            raise StopAsyncIteration()

        return piece

    def __call__(self, *args, **kwargs) -> AsyncIterator[PIECE]:
        return self
```



miniagents/promising/sentinels.py
```python
class Sentinel:
    def __bool__(self) -> bool:
        raise RuntimeError("Sentinels should not be used in boolean expressions.")


NO_VALUE = Sentinel()
DEFAULT = Sentinel()
FAILED = Sentinel()
END_OF_QUEUE = Sentinel()
AWAIT = Sentinel()
CLEAR = Sentinel()
```



miniagents/promising/sequence.py
```python
from functools import partial
from typing import Generic, AsyncIterator, Union, Optional

from miniagents.promising.errors import FunctionNotProvidedError
from miniagents.promising.promise_typing import SequenceFlattener, IN, OUT, PromiseStreamer
from miniagents.promising.promising import StreamedPromise
from miniagents.promising.sentinels import Sentinel, DEFAULT


class FlatSequence(Generic[IN, OUT]):
    sequence_promise: StreamedPromise[OUT, tuple[OUT, ...]]

    def __init__(
        self,
        incoming_streamer: PromiseStreamer[IN],
        flattener: Optional[SequenceFlattener[IN, OUT]] = None,
        start_asap: Union[bool, Sentinel] = DEFAULT,
        sequence_promise_class: type[StreamedPromise[OUT, tuple[OUT, ...]]] = StreamedPromise[OUT, tuple[OUT, ...]],
    ) -> None:
        if flattener:
            self._flattener = partial(flattener, self)

        self._input_promise = StreamedPromise(
            streamer=self._streamer,
            resolver=lambda _: None,
            start_asap=False,
        )
        self._incoming_streamer_aiter = incoming_streamer(self)

        self.sequence_promise = sequence_promise_class(
            streamer=self._input_promise,
            resolver=self._resolver,
            start_asap=start_asap,
        )

    def _flattener(self, zero_or_more_items: IN) -> AsyncIterator[OUT]:
        raise FunctionNotProvidedError(
            "The `flattener` function should be provided either via the constructor "
            "or by subclassing the `FlatSequence` class."
        )

    async def _streamer(self, _) -> AsyncIterator[OUT]:
        async for zero_or_more_items in self._incoming_streamer_aiter:
            async for item in self._flattener(zero_or_more_items):
                yield item

    async def _resolver(self, _) -> tuple[OUT, ...]:
        return tuple([item async for item in self.sequence_promise])
```



miniagents/utils.py
```python
import logging
from typing import AsyncIterator, Any, Optional, Union, Iterable, Callable

from miniagents.messages import MessageSequencePromise
from miniagents.miniagents import MessageType, MessageSequence, MessagePromise, Message, MiniAgent
from miniagents.promising.promising import StreamAppender
from miniagents.promising.sentinels import Sentinel, DEFAULT, AWAIT, CLEAR

logger = logging.getLogger(__name__)


async def achain_loop(
    agents: Iterable[Union[MiniAgent, Callable[[MessageType, ...], MessageSequencePromise], Sentinel]],
    initial_input: Optional[MessageType] = None,
) -> None:
    agents = list(agents)
    if not any(agent is AWAIT for agent in agents):
        raise ValueError(
            "There should be at least one AWAIT sentinel in the list of agents in order for the loop not to "
            "schedule the turns infinitely without actually running them."
        )

    messages = initial_input
    while True:
        for agent in agents:
            if agent is AWAIT:
                if isinstance(messages, MessageSequencePromise):
                    messages = await messages.aresolve_messages()
            elif agent is CLEAR:
                messages = None
            elif callable(agent):
                messages = agent(messages)
            elif isinstance(agent, MiniAgent):
                messages = agent.inquire(messages)
            else:
                raise ValueError(f"Invalid agent: {agent}")


def join_messages(
    messages: MessageType,
    delimiter: Optional[str] = "\n\n",
    strip_leading_newlines: bool = False,
    reference_original_messages: bool = True,
    start_asap: Union[bool, Sentinel] = DEFAULT,
    **message_metadata,
) -> MessagePromise:
    async def token_streamer(metadata_so_far: dict[str, Any]) -> AsyncIterator[str]:
        metadata_so_far.update(message_metadata)
        if reference_original_messages:
            metadata_so_far["original_messages"] = []

        first_message = True
        async for message_promise in MessageSequence.turn_into_sequence_promise(messages):
            if delimiter and not first_message:
                yield delimiter

            lstrip_newlines = strip_leading_newlines
            async for token in message_promise:
                if lstrip_newlines:
                    token = token.lstrip("\n\r")
                if token:
                    lstrip_newlines = False
                    yield token

            if reference_original_messages:
                metadata_so_far["original_messages"].append(await message_promise)

            first_message = False

    return Message.promise(
        message_token_streamer=token_streamer,
        start_asap=start_asap,
    )


def split_messages(
    messages: MessageType,
    delimiter: str = "\n\n",
    code_block_delimiter: Optional[str] = "```",
    start_asap: Union[bool, Sentinel] = DEFAULT,
    **message_metadata,
) -> MessageSequencePromise:
    async def sequence_streamer(_) -> AsyncIterator[MessagePromise]:
        text_so_far = ""
        current_text_appender: Optional[StreamAppender[str]] = None
        inside_code_block = False

        def is_text_so_far_not_empty() -> bool:
            return bool(text_so_far.replace(delimiter, ""))

        def split_text_if_needed() -> bool:
            nonlocal text_so_far, current_text_appender, inside_code_block

            delimiter_idx = -1 if inside_code_block else text_so_far.find(delimiter)
            delimiter_len = len(delimiter)

            code_delimiter_idx = text_so_far.find(
                code_block_delimiter,
                len(code_block_delimiter) if inside_code_block else 0,
            )
            if code_delimiter_idx > -1 and (delimiter_idx < 0 or code_delimiter_idx < delimiter_idx):
                delimiter_len = 0
                if inside_code_block:
                    delimiter_idx = code_delimiter_idx + len(code_block_delimiter)
                else:
                    delimiter_idx = code_delimiter_idx
                inside_code_block = not inside_code_block

            if delimiter_idx < 0:
                return False

            text = text_so_far[:delimiter_idx]
            text_so_far = text_so_far[delimiter_idx + delimiter_len :]
            if text:
                with current_text_appender:
                    current_text_appender.append(text)
                current_text_appender = None
            return True

        def start_new_message_promise() -> MessagePromise:
            nonlocal current_text_appender
            current_text_appender = StreamAppender[str]()

            async def token_streamer(metadata_so_far: dict[str, Any]) -> AsyncIterator[str]:
                metadata_so_far.update(message_metadata)
                async for token in current_text_appender:
                    yield token

            return Message.promise(
                message_token_streamer=token_streamer,
                start_asap=start_asap,
            )

        try:
            if not current_text_appender:
                yield start_new_message_promise()

            async for token in join_messages(
                messages,
                delimiter=delimiter,
                reference_original_messages=False,
                start_asap=start_asap,
            ):
                text_so_far += token

                while True:
                    if not current_text_appender and is_text_so_far_not_empty():
                        yield start_new_message_promise()
                    if not split_text_if_needed():
                        break

            if is_text_so_far_not_empty():
                if current_text_appender:
                    with current_text_appender:
                        current_text_appender.append(text_so_far)
                else:
                    yield Message(text=text_so_far, **message_metadata).as_promise

        except BaseException as exc:
            logger.debug("Error while processing a message sequence inside `split_messages`", exc_info=True)
            if current_text_appender:
                with current_text_appender:
                    current_text_appender.append(exc)
            else:
                raise exc
        finally:
            if current_text_appender:
                current_text_appender.close()

    async def sequence_resolver(sequence_promise: MessageSequencePromise) -> tuple[MessagePromise, ...]:
        return tuple([item async for item in sequence_promise])

    return MessageSequencePromise(
        streamer=sequence_streamer,
        resolver=sequence_resolver,
        start_asap=True,
    )
```



pyproject.toml
```
[tool.black]
line-length = 119

[tool.coverage.run]
branch = true

[tool.poetry]
name = "miniagents"
version = "0.0.12"
description = """\
TODO Oleksandr\
"""
authors = ["Oleksandr Tereshchenko <toporok@gmail.com>"]
homepage = "https://github.com/teremterem/MiniAgents"
readme = "README.md"
license = "MIT"

[tool.poetry.dependencies]
python = ">=3.9,<4.0"
pydantic = ">=2.0.0,<3.0.0"

[tool.poetry.dev-dependencies]
anthropic = "*"
black = "*"
ipython = "*"
jupyterlab = "*"
notebook = "*"
openai = "*"
pre-commit = "*"
pylint = "*"
pytest = "*"
pytest-asyncio = "*"
pytest-cov = "*"
python-dotenv = "*"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```



tests/test_agents.py
```python
import asyncio
from typing import Union

import pytest

from miniagents.miniagents import MiniAgents, miniagent, InteractionContext
from miniagents.promising.sentinels import DEFAULT, Sentinel


@pytest.mark.parametrize("start_asap", [False, True, DEFAULT])
@pytest.mark.asyncio
async def test_agents_run_in_parallel(start_asap: Union[bool, Sentinel]) -> None:
    event_sequence = []

    @miniagent
    async def agent1(_) -> None:
        event_sequence.append("agent1 - start")
        await asyncio.sleep(0.1)
        event_sequence.append("agent1 - end")

    @miniagent
    async def agent2(_) -> None:
        event_sequence.append("agent2 - start")
        await asyncio.sleep(0.1)
        event_sequence.append("agent2 - end")

    async with MiniAgents():
        replies1 = agent1.inquire(start_asap=start_asap)
        replies2 = agent2.inquire(start_asap=start_asap)
        if start_asap is False:
            await replies1.aresolve_messages()
            await replies2.aresolve_messages()

    if start_asap is DEFAULT or start_asap is True:
        assert event_sequence == [
            "agent1 - start",
            "agent2 - start",
            "agent1 - end",
            "agent2 - end",
        ]
    else:
        assert event_sequence == [
            "agent1 - start",
            "agent1 - end",
            "agent2 - start",
            "agent2 - end",
        ]


@pytest.mark.parametrize("start_asap", [False, True, DEFAULT])
@pytest.mark.asyncio
async def test_sub_agents_run_in_parallel(start_asap: Union[bool, Sentinel]) -> None:
    event_sequence = []

    @miniagent
    async def agent1(_) -> None:
        event_sequence.append("agent1 - start")
        await asyncio.sleep(0.1)
        event_sequence.append("agent1 - end")

    @miniagent
    async def agent2(_) -> None:
        event_sequence.append("agent2 - start")
        await asyncio.sleep(0.1)
        event_sequence.append("agent2 - end")

    @miniagent
    async def aggregation_agent(ctx: InteractionContext) -> None:
        ctx.reply([agent.inquire(start_asap=start_asap) for agent in [agent1, agent2]])

    async with MiniAgents():
        replies = aggregation_agent.inquire(start_asap=start_asap)
        if start_asap is False:
            await replies.aresolve_messages()

    if start_asap is DEFAULT or start_asap is True:
        assert event_sequence == [
            "agent1 - start",
            "agent2 - start",
            "agent1 - end",
            "agent2 - end",
        ]
    else:
        assert event_sequence == [
            "agent1 - start",
            "agent1 - end",
            "agent2 - start",
            "agent2 - end",
        ]
```



tests/test_frozen.py
```python
import hashlib
from typing import Optional
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from miniagents.promising.ext.frozen import Frozen
from miniagents.promising.promising import PromisingContext


class SampleModel(Frozen):
    some_req_field: str
    some_opt_field: int = 2
    sub_model: Optional["SampleModel"] = None


def test_sample_model_frozen() -> None:
    sample = SampleModel(some_req_field="test")

    with pytest.raises(ValidationError):
        sample.some_req_field = "test2"
    with pytest.raises(ValidationError):
        sample.some_opt_field = 3

    assert sample.some_req_field == "test"
    assert sample.some_opt_field == 2


def test_model_frozen() -> None:
    model = Frozen(some_field="some value")

    with pytest.raises(ValidationError):
        model.some_other_field = "some other value"

    assert model.some_field == "some value"


@pytest.mark.asyncio
async def test_sample_model_hash_key() -> None:
    async with PromisingContext():
        sample = SampleModel(some_req_field="test", sub_model=SampleModel(some_req_field="юнікод", some_opt_field=3))
        sample._some_private_attribute = "some value"

        expected_hash_key = hashlib.sha256(
            '{"class_": "SampleModel", "some_opt_field": 2, "some_req_field": "test", "sub_model": '
            '{"class_": "SampleModel", "some_opt_field": 3, "some_req_field": "юнікод", "sub_model": null}}'
            "".encode("utf-8")
        ).hexdigest()[:40]
        assert sample.hash_key == expected_hash_key


@pytest.mark.asyncio
async def test_model_hash_key() -> None:
    async with PromisingContext():
        model = Frozen(content="test", final_sender_alias="user", custom_field={"role": "user"})
        expected_hash_key = hashlib.sha256(
            '{"class_": "Frozen", "content": "test", "custom_field": {"class_": "Frozen", "role": "user"}, '
            '"final_sender_alias": "user"}'.encode("utf-8")
        ).hexdigest()[:40]
        assert model.hash_key == expected_hash_key


def test_nested_object_not_copied() -> None:
    sub_model = SampleModel(some_req_field="test")
    sample = SampleModel(some_req_field="test", sub_model=sub_model)

    assert sample.sub_model is sub_model


@pytest.mark.asyncio
async def test_hash_key_calculated_once() -> None:
    original_sha256 = hashlib.sha256

    with patch("hashlib.sha256", side_effect=original_sha256) as mock_sha256:
        async with PromisingContext():
            sample = SampleModel(some_req_field="test")
            mock_sha256.assert_not_called()

            assert sample.hash_key == "2f9753c92f0452bacafaa606b6076d2bf266e095"
            mock_sha256.assert_called_once()

            assert sample.hash_key == "2f9753c92f0452bacafaa606b6076d2bf266e095"
            mock_sha256.assert_called_once()


@pytest.mark.asyncio
async def test_model_hash_key_vs_key_ordering() -> None:
    async with PromisingContext():
        model1 = Frozen(some_field="test", some_other_field=2)
        model2 = Frozen(some_other_field=2, some_field="test")

        assert model1.hash_key == model2.hash_key
```



tests/test_llm.py
```python
from typing import Callable

import pytest
from dotenv import load_dotenv

from miniagents.messages import Message
from miniagents.miniagents import MiniAgents, MiniAgent

load_dotenv()

from miniagents.ext.llm.anthropic import create_anthropic_agent
from miniagents.ext.llm.openai import create_openai_agent


def _check_openai_response(message: Message) -> None:
    assert message.text.strip() == "I AM ONLINE"
    assert message.choices[0].finish_reason == "stop"


def _check_anthropic_response(message: Message) -> None:
    assert message.text.strip() == "I AM ONLINE"
    assert message.stop_reason == "end_turn"


@pytest.mark.parametrize(
    "llm_agent, check_response_func",
    [
        (create_openai_agent(model="gpt-3.5-turbo-0125"), _check_openai_response),
        (create_anthropic_agent(model="claude-3-haiku-20240307"), _check_anthropic_response),
    ],
)
@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("start_asap", [False, True])
async def test_llm(
    start_asap: bool,
    stream: bool,
    llm_agent: MiniAgent,
    check_response_func: Callable[[Message], None],
) -> None:
    async with MiniAgents(start_everything_asap_by_default=start_asap):
        reply_sequence = llm_agent.inquire(
            Message(text="ANSWER:", role="assistant"),
            system=(
                "This is a test to verify that you are online. Your response will be validated using a strict "
                "program that does not tolerate any deviations from the expected output at all. Please respond "
                "with these exact words, all capitals and no punctuation: I AM ONLINE"
            ),
            stream=stream,
            max_tokens=20,
            temperature=0,
        )

        result = ""
        async for msg_promise in reply_sequence:
            async for token in msg_promise:
                result += token
            check_response_func(await msg_promise)
    assert result.strip() == "I AM ONLINE"
```



tests/test_message.py
```python
import hashlib
import json

import pytest

from miniagents.messages import Message
from miniagents.miniagents import MiniAgents
from miniagents.promising.ext.frozen import Frozen
from miniagents.promising.promising import PromisingContext, Promise
from miniagents.promising.sentinels import DEFAULT


@pytest.mark.asyncio
async def test_message_nesting_vs_hash_key() -> None:
    class SpecialNode(Frozen):
        ...

    async with PromisingContext():
        message = Message(
            text="юнікод",
            extra_field=[
                15,
                {
                    "role": "user",
                    "nested_nested": (Message(text="nested_text"), Message(text="nested_text2")),
                    "nested_nested2": [Message(text="nested_text2")],
                },
            ],
            extra_node=SpecialNode(nested_nested=Message(text="nested_text3")),
            nested_message=Message(text="nested_text"),
        )

        expected_structure = {
            "class_": "Message",
            "text": "юнікод",
            "text_template": None,
            "extra_field": (
                15,
                {
                    "class_": "Frozen",
                    "role": "user",
                    "nested_nested__hash_keys": (
                        "47e977f85cff13ea8980cf3d76959caec8a4984a",
                        "91868c8c8398b49deb9a04a73c4ea95bdb2eaa65",
                    ),
                    "nested_nested2__hash_keys": ("91868c8c8398b49deb9a04a73c4ea95bdb2eaa65",),
                },
            ),
            "extra_node": {
                "class_": "SpecialNode",
                "nested_nested__hash_key": "25a897f6457abf51fad6a28d86905918bb610038",
            },
            "nested_message__hash_key": "47e977f85cff13ea8980cf3d76959caec8a4984a",
        }
        assert message.serialize() == expected_structure

        expected_hash_key = hashlib.sha256(
            json.dumps(expected_structure, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:40]
        assert message.hash_key == expected_hash_key


@pytest.mark.parametrize("start_asap", [False, True, DEFAULT])
@pytest.mark.asyncio
async def test_on_persist_message_event_called_once(start_asap: bool) -> None:
    promise_resolved_calls = 0
    persist_message_calls = 0

    async def on_promise_resolved(_, __) -> None:
        nonlocal promise_resolved_calls
        promise_resolved_calls += 1

    async def on_persist_message(_, __) -> None:
        nonlocal persist_message_calls
        persist_message_calls += 1

    some_message = Message()

    async with MiniAgents(
        on_promise_resolved=on_promise_resolved,
        on_persist_message=on_persist_message,
    ):
        Promise(prefill_result=some_message, start_asap=start_asap)
        Promise(prefill_result=some_message, start_asap=start_asap)

    assert promise_resolved_calls == 2
    assert persist_message_calls == 1


@pytest.mark.parametrize("start_asap", [False, True, DEFAULT])
@pytest.mark.asyncio
async def test_on_persist_message_event_called_twice(start_asap: bool) -> None:
    promise_resolved_calls = 0
    persist_message_calls = 0

    async def on_promise_resolved(_, __) -> None:
        nonlocal promise_resolved_calls
        promise_resolved_calls += 1

    async def on_persist_message(_, __) -> None:
        nonlocal persist_message_calls
        persist_message_calls += 1

    message1 = Message()
    message2 = Message()

    async with MiniAgents(
        on_promise_resolved=on_promise_resolved,
        on_persist_message=on_persist_message,
    ):
        Promise(prefill_result=message1, start_asap=start_asap)
        Promise(prefill_result=message2, start_asap=start_asap)

    assert promise_resolved_calls == 2
    assert persist_message_calls == 2


@pytest.mark.parametrize("start_asap", [False, True, DEFAULT])
@pytest.mark.asyncio
async def test_on_persist_message_event_not_called(start_asap: bool) -> None:
    promise_resolved_calls = 0
    persist_message_calls = 0

    async def on_promise_resolved(_, __) -> None:
        nonlocal promise_resolved_calls
        promise_resolved_calls += 1

    async def on_persist_message(_, __) -> None:
        nonlocal persist_message_calls
        persist_message_calls += 1

    not_a_message = Frozen(some_field="not a message")

    async with MiniAgents(
        on_promise_resolved=on_promise_resolved,
        on_persist_message=on_persist_message,
    ):
        Promise(prefill_result=not_a_message, start_asap=start_asap)
        Promise(prefill_result=not_a_message, start_asap=start_asap)

    assert promise_resolved_calls == 2
    assert persist_message_calls == 0
```



tests/test_message_sequence.py
```python
import pytest

from miniagents.messages import Message
from miniagents.miniagents import MessageSequence
from miniagents.promising.promising import PromisingContext
from miniagents.promising.sentinels import DEFAULT


@pytest.mark.parametrize("start_asap", [False, True, DEFAULT])
@pytest.mark.asyncio
async def test_message_sequence(start_asap: bool) -> None:
    async with PromisingContext():
        msg_seq1 = MessageSequence(
            appender_capture_errors=True,
            start_asap=start_asap,
        )
        with msg_seq1.message_appender:
            msg_seq1.message_appender.append("msg1")
            msg_seq1.message_appender.append({"text": "msg2", "some_attr": 2})
            msg_seq1.message_appender.append(Message(text="msg3", another_attr=3))

            msg_seq2 = MessageSequence(
                appender_capture_errors=True,
                start_asap=start_asap,
            )
            with msg_seq2.message_appender:
                msg_seq2.message_appender.append("msg4")

                msg_seq3 = MessageSequence(
                    appender_capture_errors=True,
                    start_asap=start_asap,
                )
                with msg_seq3.message_appender:
                    msg_seq3.message_appender.append("msg5")
                    msg_seq3.message_appender.append(["msg6", "msg7"])
                    msg_seq3.message_appender.append([[Message(text="msg8", another_attr=8)]])

                msg_seq2.message_appender.append(msg_seq3.sequence_promise)
                msg_seq2.message_appender.append("msg9")

            msg_seq1.message_appender.append(msg_seq2.sequence_promise)
            msg_seq1.message_appender.append(Message.promise(text="msg10", yet_another_attr=10))

        message_result = [await msg_promise async for msg_promise in msg_seq1.sequence_promise]
        assert message_result == [
            Message(text="msg1"),
            Message(text="msg2", some_attr=2),
            Message(text="msg3", another_attr=3),
            Message(text="msg4"),
            Message(text="msg5"),
            Message(text="msg6"),
            Message(text="msg7"),
            Message(text="msg8", another_attr=8),
            Message(text="msg9"),
            Message(text="msg10", yet_another_attr=10),
        ]

        token_result = [token async for msg_promise in msg_seq1.sequence_promise async for token in msg_promise]
        assert token_result == [
            "msg1",
            "msg2",
            "msg3",
            "msg4",
            "msg5",
            "msg6",
            "msg7",
            "msg8",
            "msg9",
            "msg10",
        ]


@pytest.mark.parametrize("start_asap", [False, True, DEFAULT])
@pytest.mark.asyncio
async def test_message_sequence_error(start_asap: bool) -> None:
    async with PromisingContext(appenders_capture_errors_by_default=True):
        msg_seq1 = MessageSequence(start_asap=start_asap)
        with msg_seq1.message_appender:
            msg_seq1.message_appender.append("msg1")

            msg_seq2 = MessageSequence(start_asap=start_asap)
            with msg_seq2.message_appender:
                msg_seq2.message_appender.append("msg2")

                msg_seq3 = MessageSequence(start_asap=start_asap)
                with msg_seq3.message_appender:
                    msg_seq3.message_appender.append("msg3")
                    raise ValueError("msg5")

                msg_seq2.message_appender.append(msg_seq3.sequence_promise)
                msg_seq2.message_appender.append("msg6")

            msg_seq1.message_appender.append(msg_seq2.sequence_promise)
            msg_seq1.message_appender.append("msg7")

        message_result = []
        with pytest.raises(ValueError, match="msg5"):
            async for msg_promise in msg_seq1.sequence_promise:
                message_result.append(await msg_promise)

    assert message_result == [
        Message(text="msg1"),
        Message(text="msg2"),
        Message(text="msg3"),
    ]
```



tests/test_promise.py
```python
from typing import AsyncIterator

import pytest

from miniagents.promising.promising import StreamedPromise, StreamAppender, PromisingContext
from miniagents.promising.sentinels import DEFAULT


@pytest.mark.parametrize("start_asap", [False, True, DEFAULT])
@pytest.mark.asyncio
async def test_stream_replay_iterator(start_asap: bool) -> None:
    streamer_iterations = 0

    async def streamer(_streamed_promise: StreamedPromise) -> AsyncIterator[int]:
        nonlocal streamer_iterations
        for i in range(1, 6):
            streamer_iterations += 1
            yield i

    async def resolver(_streamed_promise: StreamedPromise) -> list[int]:
        return [piece async for piece in _streamed_promise]

    async with PromisingContext():
        streamed_promise = StreamedPromise(
            streamer=streamer,
            resolver=resolver,
            start_asap=start_asap,
        )

        assert [i async for i in streamed_promise] == [1, 2, 3, 4, 5]
        assert [i async for i in streamed_promise] == [1, 2, 3, 4, 5]

    assert streamer_iterations == 5


@pytest.mark.parametrize("start_asap", [False, True, DEFAULT])
@pytest.mark.asyncio
async def test_stream_replay_iterator_exception(start_asap: bool) -> None:
    with StreamAppender(capture_errors=True) as appender:
        for i in range(1, 6):
            if i == 3:
                raise ValueError("Test error")
            appender.append(i)

    async def resolver(_streamed_promise: StreamedPromise) -> list[int]:
        return [piece async for piece in _streamed_promise]

    async def iterate_over_promise():
        promise_iterator = streamed_promise.__aiter__()

        assert await promise_iterator.__anext__() == 1
        assert await promise_iterator.__anext__() == 2
        with pytest.raises(ValueError):
            await promise_iterator.__anext__()
        with pytest.raises(StopAsyncIteration):
            await promise_iterator.__anext__()
        with pytest.raises(StopAsyncIteration):
            await promise_iterator.__anext__()

    async with PromisingContext():
        streamed_promise = StreamedPromise(
            streamer=appender,
            resolver=resolver,
            start_asap=start_asap,
        )

        await iterate_over_promise()
        await iterate_over_promise()


async def _async_streamer_but_not_generator(_):
    return


@pytest.mark.parametrize(
    "broken_streamer",
    [
        lambda _: iter([]),
        _async_streamer_but_not_generator,
    ],
)
@pytest.mark.parametrize("start_asap", [False, True, DEFAULT])
@pytest.mark.asyncio
async def test_broken_streamer(broken_streamer, start_asap: bool) -> None:
    async def resolver(_streamed_promise: StreamedPromise) -> list[int]:
        return [piece async for piece in _streamed_promise]

    async def iterate_over_promise():
        promise_iterator = streamed_promise.__aiter__()

        with pytest.raises((TypeError, AttributeError)):
            await promise_iterator.__anext__()
        with pytest.raises(StopAsyncIteration):
            await promise_iterator.__anext__()
        with pytest.raises(StopAsyncIteration):
            await promise_iterator.__anext__()

    async with PromisingContext():
        streamed_promise = StreamedPromise(
            streamer=broken_streamer,
            resolver=resolver,
            start_asap=start_asap,
        )

        await iterate_over_promise()
        await iterate_over_promise()


@pytest.mark.parametrize(
    "broken_resolver",
    [
        lambda _: [],
        TypeError,
    ],
)
@pytest.mark.parametrize("start_asap", [False, True, DEFAULT])
@pytest.mark.asyncio
async def test_broken_stream_resolver(broken_resolver, start_asap: bool) -> None:
    expected_resolver_call_count = 0
    actual_resolver_call_count = 0
    if isinstance(broken_resolver, type):
        expected_resolver_call_count = 1
        error_class = broken_resolver

        async def broken_resolver(_streamed_promise: StreamedPromise) -> None:
            nonlocal actual_resolver_call_count
            actual_resolver_call_count += 1
            raise error_class("Test error")

    with StreamAppender(capture_errors=True) as appender:
        for i in range(1, 6):
            appender.append(i)

    async with PromisingContext():
        streamed_promise = StreamedPromise(
            streamer=appender,
            resolver=broken_resolver,
            start_asap=start_asap,
        )

        with pytest.raises(TypeError) as exc_info1:
            await streamed_promise
        error1 = exc_info1.value

        assert [i async for i in streamed_promise] == [1, 2, 3, 4, 5]

        with pytest.raises(TypeError) as exc_info2:
            await streamed_promise

    assert error1 is exc_info2.value

    assert actual_resolver_call_count == expected_resolver_call_count


@pytest.mark.parametrize("start_asap", [False, True, DEFAULT])
@pytest.mark.asyncio
async def test_streamed_promise_aresolve(start_asap: bool) -> None:
    resolver_calls = 0

    with StreamAppender(capture_errors=False) as appender:
        for i in range(1, 6):
            appender.append(i)

    async def resolver(_streamed_promise: StreamedPromise) -> list[int]:
        nonlocal resolver_calls
        resolver_calls += 1
        return [piece async for piece in _streamed_promise]

    async with PromisingContext():
        streamed_promise = StreamedPromise(
            streamer=appender,
            resolver=resolver,
            start_asap=start_asap,
        )

        result1 = await streamed_promise
        result2 = await streamed_promise

        assert resolver_calls == 1

        assert result1 == [1, 2, 3, 4, 5]
        assert result2 is result1


@pytest.mark.parametrize("start_asap", [False, True, DEFAULT])
@pytest.mark.asyncio
async def test_stream_appender_dont_capture_errors(start_asap: bool) -> None:
    with pytest.raises(ValueError):
        with StreamAppender(capture_errors=False) as appender:
            for i in range(1, 6):
                if i == 3:
                    raise ValueError("Test error")
                appender.append(i)

    async def resolver(_streamed_promise: StreamedPromise) -> list[int]:
        return [piece async for piece in _streamed_promise]

    async with PromisingContext():
        streamed_promise = StreamedPromise(
            streamer=appender,
            resolver=resolver,
            start_asap=start_asap,
        )

        assert await streamed_promise == [1, 2]


@pytest.mark.parametrize("start_asap", [False, True, DEFAULT])
@pytest.mark.asyncio
async def test_streamed_promise_same_instance(start_asap: bool) -> None:
    async def streamer(_streamed_promise: StreamedPromise) -> AsyncIterator[int]:
        assert _streamed_promise is streamed_promise
        yield 1

    async def resolver(_streamed_promise: StreamedPromise) -> list[int]:
        assert _streamed_promise is streamed_promise
        return [piece async for piece in _streamed_promise]

    async with PromisingContext():
        streamed_promise = StreamedPromise(
            streamer=streamer,
            resolver=resolver,
            start_asap=start_asap,
        )

        await streamed_promise
```



tests/test_sequence.py
```python
from typing import AsyncIterator

import pytest

from miniagents.promising.promising import PromisingContext, StreamAppender
from miniagents.promising.sentinels import DEFAULT
from miniagents.promising.sequence import FlatSequence


@pytest.mark.parametrize("start_asap", [False, True, DEFAULT])
@pytest.mark.asyncio
async def test_flat_sequence(start_asap: bool) -> None:
    async def flattener(_, number: int) -> AsyncIterator[int]:
        for _ in range(number):
            yield number

    async with PromisingContext():
        stream_appender = StreamAppender[int](capture_errors=True)
        flat_sequence = FlatSequence[int, int](
            incoming_streamer=stream_appender,
            flattener=flattener,
            start_asap=start_asap,
        )
        with stream_appender:
            stream_appender.append(0)
            stream_appender.append(1)
            stream_appender.append(2)
            stream_appender.append(3)

        assert await flat_sequence.sequence_promise == (1, 2, 2, 3, 3, 3)
        assert [i async for i in flat_sequence.sequence_promise] == [1, 2, 2, 3, 3, 3]
```