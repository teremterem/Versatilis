"""
This module contains the admin classes for the models in the miniagents_copilot app.
"""

from django.contrib import admin

from miniagents_copilot.models import DataNode, LangModelGenerationStats
from miniagents_copilot.utils import pformat_pre_html, string_preview, format_time_utc


class DataNodeAdmin(admin.ModelAdmin):
    """
    Admin class for DataNode model.
    """

    ordering = ["-touched_timestamp_ms"]
    list_filter = [
        "node_class",
    ]
    search_fields = [
        "hash_key",
    ]
    list_display = [
        "hash_key",
        # "node_class",
        "touched_time_utc",
        "pretty_payload",
    ]
    list_display_links = [
        "hash_key",
        # "node_class",
        "touched_time_utc",
        # "pretty_payload",
    ]
    fields = [
        "hash_key",
        "node_class",
        "pretty_payload",
        "created_time_utc",
        "touched_time_utc",
        "created_timestamp_ms",
        "touched_timestamp_ms",
    ]

    @admin.display(description="Created Time (UTC)")
    def created_time_utc(self, obj: DataNode) -> str:
        """
        Return a human-readable version of the created_timestamp_ms field.
        """
        return format_time_utc(obj.created_timestamp_ms)

    @admin.display(description="Touched Time (UTC)")
    def touched_time_utc(self, obj: DataNode) -> str:
        """
        Return a human-readable version of the touched_timestamp_ms field.
        """
        return format_time_utc(obj.touched_timestamp_ms)

    @admin.display(description="Payload")
    def pretty_payload(self, obj: DataNode) -> str:
        """
        Return a pretty-printed version of the payload dict.
        """
        return pformat_pre_html(obj.payload, width=80)

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False


class LangModelGenerationStatsAdmin(admin.ModelAdmin):
    """
    Admin class for LangModelGenerationStats model.
    """

    ordering = ["-timestamp_ms"]
    list_display = [
        "time_utc",
        "model_name",
        "input_token_num",
        "output_token_num",
        "payload_preview",
    ]
    list_display_links = list_display
    fields = [
        "time_utc",
        "model_name",
        "input_token_num",
        "output_token_num",
        "data_node",
        "pretty_payload",
        "timestamp_ms",
    ]

    @admin.display(description="Time (UTC)")
    def time_utc(self, obj: LangModelGenerationStats) -> str:
        """
        Return a human-readable version of the timestamp_ms field.
        """
        return format_time_utc(obj.timestamp_ms)

    @admin.display(description="Payload")
    def pretty_payload(self, obj: LangModelGenerationStats) -> str:
        """
        Return a pretty-printed version of the payload dict.
        """
        return pformat_pre_html(obj.data_node.payload if obj.data_node else None)

    @admin.display(description="Payload")
    def payload_preview(self, obj: LangModelGenerationStats) -> str:
        """
        Return a preview of the payload dict.
        """
        return string_preview(obj.data_node.payload if obj.data_node else None, preview_chars=100)

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False


admin.site.register(DataNode, DataNodeAdmin)
admin.site.register(LangModelGenerationStats, LangModelGenerationStatsAdmin)
