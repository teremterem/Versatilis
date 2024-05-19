"""
This module contains the models for the miniagents_copilot app.
"""

import logging
from pprint import pformat

from django.db import models, IntegrityError
from miniagents.ext.llm.llm_common import LangModelMessage
from miniagents.messages import Message

from miniagents_copilot.utils import current_time_utc_ms
from versatilis_config import mini_agents

logger = logging.getLogger(__name__)


class DataNode(models.Model):
    """
    A model to store Node objects from MiniAgents framework.
    """

    hash_key = models.CharField(max_length=64, unique=True)
    node_class = models.CharField(max_length=255, db_index=True)
    payload = models.JSONField()
    created_timestamp_ms = models.BigIntegerField(default=current_time_utc_ms)
    touched_timestamp_ms = models.BigIntegerField(default=current_time_utc_ms)

    def __str__(self) -> str:
        return str(self.hash_key)[:8]


class LangModelGenerationStats(models.Model):
    """
    A model to store statistics of text generations done by LLMs.
    """

    data_node = models.ForeignKey(DataNode, null=True, blank=True, on_delete=models.SET_NULL)
    timestamp_ms = models.BigIntegerField(default=current_time_utc_ms)
    model_name = models.TextField(null=True, blank=True)
    input_token_num = models.IntegerField(null=True, blank=True)
    output_token_num = models.IntegerField(null=True, blank=True)


@mini_agents.on_persist_message
async def on_persist_message(_, message: Message) -> None:
    """
    Persist Versatilis Messages in the database.
    """
    data_node = None

    try:
        if getattr(message, "persist_to_db", True):
            data_node = await DataNode.objects.acreate(
                hash_key=message.hash_key,
                node_class=message.class_,
                payload=message.serialize(),
            )

    except IntegrityError:
        data_node = await DataNode.objects.aget(hash_key=message.hash_key)
        data_node.touched_timestamp_ms = current_time_utc_ms()
        await data_node.asave(update_fields=["touched_timestamp_ms"])

    finally:
        if isinstance(message, LangModelMessage):
            try:
                input_token_num = message.usage.input_tokens
                output_token_num = message.usage.output_tokens
            except AttributeError:
                try:
                    input_token_num = message.usage.prompt_tokens
                    output_token_num = message.usage.completion_tokens
                except AttributeError:
                    input_token_num = None
                    output_token_num = None

            await LangModelGenerationStats.objects.acreate(
                data_node=data_node,
                model_name=message.model,
                input_token_num=input_token_num,
                output_token_num=output_token_num,
            )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "COLLECTED: %s\n\n%s\n\n%s\n",
                message.class_,
                message.hash_key,
                pformat(message.serialize(), width=119),
            )
