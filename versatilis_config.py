"""
Versatilis-specific configurations.
"""

import os

from dotenv import load_dotenv
from miniagents.ext.llms.anthropic import create_anthropic_agent
from miniagents.miniagents import MiniAgents

load_dotenv()

DJANGO_SECRET_KEY = os.environ["DJANGO_SECRET_KEY"]

DJANGO_DEBUG = os.environ.get("DJANGO_DEBUG", "False").lower() == "true"
DEBUG_DJANGO_REQUESTS = os.environ.get("DEBUG_DJANGO_REQUESTS", "False").lower() == "true"
DJANGO_LOG_LEVEL = os.environ.get("DJANGO_LOG_LEVEL") or "INFO"
VERSATILIS_LOG_LEVEL = os.environ.get("VERSATILIS_LOG_LEVEL") or "INFO"

DJANGO_HOSTNAME = os.environ.get("DJANGO_HOSTNAME")

mini_agents = MiniAgents()

if os.getenv("PROMPTLAYER_API_KEY"):
    import promptlayer  # pylint: disable=import-outside-toplevel

    PL_TAGS_KW = {"pl_tags": [os.getenv("PL_TAG") or "versatilis-no-tag"]}
    anthropic_agent = create_anthropic_agent(
        async_client=promptlayer.anthropic.AsyncAnthropic(),
    )
else:
    PL_TAGS_KW = {}
    anthropic_agent = create_anthropic_agent()
