"""MCP identity resolution and schema augmentation for tool calls."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.core.config import Settings

EXCLUDED_TOOLS = frozenset({"finish"})

USER_SCOPED_TOOLS = frozenset(
    {
        "list_notes",
        "get_note",
        "create_note",
        "update_note",
        "list_translations",
        "translate_text",
    }
)

TASK_SCOPED_TOOLS = frozenset(
    {
        "ingest_task_document",
        "list_task_documents",
        "search_task_documents",
    }
)


class IdentityError(ValueError):
    """Raised when a tool call is missing required user_id or task_id."""


def augment_schema(tool_name: str, parameters: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow-copied schema with optional MCP identity properties."""
    schema = deepcopy(parameters) if parameters else {"type": "object", "properties": {}}
    properties = schema.setdefault("properties", {})
    if not isinstance(properties, dict):
        properties = {}
        schema["properties"] = properties

    if tool_name in USER_SCOPED_TOOLS and "user_id" not in properties:
        properties["user_id"] = {
            "type": "string",
            "description": ("Owner user id for this call. Falls back to MCP_USER_ID when omitted."),
        }
    if tool_name in TASK_SCOPED_TOOLS and "task_id" not in properties:
        properties["task_id"] = {
            "type": "string",
            "description": (
                "Task id for RAG-scoped tools. Falls back to MCP_TASK_ID when omitted."
            ),
        }
    return schema


def resolve_identity(
    tool_name: str,
    arguments: dict[str, Any],
    settings: Settings,
) -> tuple[dict[str, Any], str | None, str | None]:
    """Pop identity args and resolve them against settings.

    Returns ``(remaining_arguments, user_id, task_id)``. Raises ``IdentityError``
    when a scoped tool is missing the required identity.
    """
    remaining = dict(arguments or {})
    arg_user = remaining.pop("user_id", None)
    arg_task = remaining.pop("task_id", None)

    user_id = arg_user if isinstance(arg_user, str) and arg_user.strip() else None
    if user_id is None:
        user_id = settings.mcp_user_id

    task_id = arg_task if isinstance(arg_task, str) and arg_task.strip() else None
    if task_id is None:
        task_id = settings.mcp_task_id

    if tool_name in USER_SCOPED_TOOLS and not user_id:
        raise IdentityError(f"tool '{tool_name}' requires user_id via arguments or MCP_USER_ID")
    if tool_name in TASK_SCOPED_TOOLS and not task_id:
        raise IdentityError(f"tool '{tool_name}' requires task_id via arguments or MCP_TASK_ID")
    return remaining, user_id, task_id
