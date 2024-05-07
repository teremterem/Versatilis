"""
This file is used to define the URL patterns for the miniagents_copilot app.
"""

from django.urls import path

from miniagents_copilot import views
from versatilis_config import TELEGRAM_TOKEN

urlpatterns = [
    path(f"{TELEGRAM_TOKEN}/", views.telegram_webhook, name="telegram_webhook"),
]
