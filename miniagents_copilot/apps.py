"""
App configuration for miniagents_copilot Django app.
"""

import logging

from django.apps import AppConfig

from miniagents_copilot.telegram_bot import telegram_app
from versatilis_config import mini_agents, TELEGRAM_TOKEN, WEBHOOK_BASE_URL

logger = logging.getLogger(__name__)


class MiniAgentsCopilotConfig(AppConfig):
    """
    App configuration for miniagents_copilot Django app.
    """

    default_auto_field = "django.db.models.BigAutoField"
    name = "miniagents_copilot"

    def ready(self) -> None:
        """
        Called when Django starts.
        """
        mini_agents.activate()

        logger.info("MiniAgentsCopilotConfig.ready() - entered")

        async def _init_telegram_bot():
            logger.info("MiniAgentsCopilotConfig.ready() - preparing to init telegram bot")
            await telegram_app.initialize()
            logger.info("MiniAgentsCopilotConfig.ready() - telegram bot initialized")

            webhook_url = f"{WEBHOOK_BASE_URL}/{TELEGRAM_TOKEN}/"
            # TODO oleksandr: use secret_token parameter of set_webhook instead of the token in the url
            await telegram_app.bot.set_webhook(webhook_url)
            logger.info("MiniAgentsCopilotConfig.ready() - telegram bot webhook set")

        mini_agents.schedule_task(_init_telegram_bot())

        logger.info("MiniAgentsCopilotConfig.ready() - exited")
