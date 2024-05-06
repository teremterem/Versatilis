"""
ASGI config for Versatilis project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.0/howto/deployment/asgi/
"""

import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "versatilis.settings")

# pylint: disable=wrong-import-position
from django.core.asgi import get_asgi_application

application = get_asgi_application()
