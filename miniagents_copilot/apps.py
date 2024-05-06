"""
App configuration for miniagents_copilot Django app.
"""

from django.apps import AppConfig

from versatilis_config import mini_agents


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
