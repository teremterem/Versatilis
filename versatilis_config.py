"""
Versatilis-specific configurations.
"""

import os

from dotenv import load_dotenv
from miniagents.ext.llm.anthropic import create_anthropic_agent
from miniagents.ext.llm.openai import create_openai_agent
from miniagents.miniagents import MiniAgents

load_dotenv()

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]

DJANGO_SECRET_KEY = os.environ["DJANGO_SECRET_KEY"]

DJANGO_DEBUG = os.environ.get("DJANGO_DEBUG", "False").lower() == "true"
DEBUG_DJANGO_REQUESTS = os.environ.get("DEBUG_DJANGO_REQUESTS", "False").lower() == "true"
DJANGO_LOG_LEVEL = os.environ.get("DJANGO_LOG_LEVEL") or "INFO"
VERSATILIS_LOG_LEVEL = os.environ.get("VERSATILIS_LOG_LEVEL") or "INFO"

DJANGO_HOSTNAME = os.environ["DJANGO_HOSTNAME"]
WEBHOOK_BASE_URL = f"https://{DJANGO_HOSTNAME}"

mini_agents = MiniAgents()

if os.getenv("PROMPTLAYER_API_KEY"):
    from promptlayer import PromptLayer  # pylint: disable=import-outside-toplevel

    promptlayer_client = PromptLayer()

    PL_TAGS_KW = {"pl_tags": [os.getenv("PL_TAG") or "versatilis-no-tag"]}
    anthropic_agent = create_anthropic_agent(
        async_client=promptlayer_client.anthropic.AsyncAnthropic(),
    )
    openai_agent = create_openai_agent(
        async_client=promptlayer_client.openai.AsyncOpenAI(),
    )
else:
    PL_TAGS_KW = {}
    anthropic_agent = create_anthropic_agent()
    openai_agent = create_openai_agent()
