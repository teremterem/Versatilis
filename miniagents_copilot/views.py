"""
This file contains the views for the miniagents_copilot app.
"""

import json
import logging

from django.http import HttpRequest, HttpResponse
from django.views.decorators.csrf import csrf_exempt

from miniagents_copilot.agents.telegram_agent import telegram_agent
from miniagents_copilot.telegram_bot import TelegramUpdateMessage

logger = logging.getLogger(__name__)


@csrf_exempt
async def telegram_webhook(request: HttpRequest) -> HttpResponse:
    """
    Handle the incoming Telegram update.
    """
    # noinspection PyBroadException
    try:
        request_json = json.loads(request.body)
        # TODO Oleksandr: `delegate` instead of `inquire`
        telegram_agent.inquire(TelegramUpdateMessage(**request_json))
    except Exception:  # pylint: disable=broad-except
        logger.exception("FAILED TO PROCESS TELEGRAM UPDATE")

    return HttpResponse("OK")
