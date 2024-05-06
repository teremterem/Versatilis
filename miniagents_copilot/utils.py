"""
Utility functions.
"""

import datetime
from json import JSONDecodeError
from pprint import pformat
from typing import Any

import httpx
from django.utils.html import format_html

from versatilis_config import DJANGO_DEBUG

HTML_PRE_START = '<pre style="white-space: pre-wrap">'
HTML_PRE_END = "</pre>"
HTML_PRE_TEMPLATE = HTML_PRE_START + "{}" + HTML_PRE_END


async def adownload_json(url: str, raise_for_status: bool = True) -> Any:
    """
    Download a JSON object from a URL.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if raise_for_status:
            response.raise_for_status()
        try:
            return response.json()
        except JSONDecodeError as exc:
            if DJANGO_DEBUG:
                raise ValueError(f"FAILED TO DECODE JSON:\n\n{response.text}") from exc
            raise


def string_preview(data: Any, preview_chars: int = 100) -> str:
    """
    Return a string preview of data.
    """
    data = str(data)
    return data[: preview_chars - 3] + "..." if len(data) > preview_chars else data


def format_pre_html(text: str | None) -> str:
    """
    Format a string as a pre-formatted HTML block.
    """
    return "-" if text is None else format_html(HTML_PRE_TEMPLATE, text)


def pformat_pre_html(data: Any | None, width: int = 119) -> str:
    """
    Pretty format a data structure as a pre-formatted HTML block.
    """
    text = None if data is None else pformat(data, width=width)
    return format_pre_html(text)


def current_time_utc_ms():
    """
    Returns current time in UTC in milliseconds.
    """
    return int(datetime.datetime.now(datetime.UTC).timestamp() * 1000)


def format_time_utc(utc_timestamp_ms: int | None) -> str:
    """
    Format a timestamp in milliseconds.
    """
    if utc_timestamp_ms is None:
        return "-"
    return datetime.datetime.fromtimestamp(utc_timestamp_ms / 1000, datetime.UTC).strftime("%Y-%m-%d %H:%M:%S")
