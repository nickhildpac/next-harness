from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.repositories.notes import NoteRepository
from app.repositories.translations import TranslationRepository
from app.tools.registry import Tool, ToolContext, ToolError


def _require_session(context: ToolContext):
    if context.session is None:
        raise ToolError("no database session available for this tool")
    return context.session


def _require_user(context: ToolContext) -> str:
    if not context.user_id:
        raise ToolError("no user_id available on the tool context")
    return context.user_id


async def _now(_: dict[str, Any], __: ToolContext) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {"iso": now.isoformat(), "epoch": int(now.timestamp())}


async def _list_notes(args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    session = _require_session(context)
    user_id = args.get("user_id") or _require_user(context)
    notes = await NoteRepository(session).list_for_user(user_id)
    limit = int(args.get("limit") or 20)
    items = [
        {"id": n.id, "title": n.title, "style_name": n.style_name, "updated_at": n.updated_at.isoformat()}
        for n in notes[:limit]
    ]
    return {"count": len(items), "items": items}


async def _get_note(args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    session = _require_session(context)
    note_id = args.get("note_id")
    if not note_id:
        raise ToolError("note_id is required")
    note = await NoteRepository(session).get(note_id)
    if note is None:
        raise ToolError(f"note '{note_id}' not found")
    return {
        "id": note.id,
        "title": note.title,
        "content": note.content,
        "style_name": note.style_name,
        "custom_instructions": note.custom_instructions,
    }


async def _create_note(args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    session = _require_session(context)
    user_id = _require_user(context)
    content = args.get("content")
    if not content:
        raise ToolError("content is required")
    note = await NoteRepository(session).create(
        user_id=user_id,
        title=args.get("title"),
        content=content,
        style_name=args.get("style_name") or "default",
        custom_instructions=args.get("custom_instructions"),
    )
    return {"id": note.id, "title": note.title}


async def _list_translations(args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    session = _require_session(context)
    user_id = args.get("user_id") or _require_user(context)
    translations = await TranslationRepository(session).list_for_user(user_id)
    limit = int(args.get("limit") or 20)
    items = [
        {
            "id": t.id,
            "title": t.title,
            "target_language": t.target_language,
            "updated_at": t.updated_at.isoformat(),
        }
        for t in translations[:limit]
    ]
    return {"count": len(items), "items": items}


async def _http_fetch(args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    url = args.get("url")
    if not url:
        raise ToolError("url is required")
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ToolError("url must be http(s)")
    max_bytes = int(args.get("max_bytes") or 8192)
    if context.http_client is None:
        raise ToolError("no http client available in this run")
    response = await context.http_client.get(url, timeout=10.0)
    body = response.text[:max_bytes]
    return {
        "url": url,
        "status": response.status_code,
        "content_type": response.headers.get("content-type"),
        "body": body,
        "truncated": len(response.text) > max_bytes,
    }


async def _finish(args: dict[str, Any], _: ToolContext) -> dict[str, Any]:
    return {"summary": args.get("summary") or "done"}


def all_tools() -> list[Tool]:
    return [
        Tool(
            name="now",
            description="Return the current UTC time.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            handler=_now,
        ),
        Tool(
            name="list_notes",
            description="List markdown notes owned by a user (defaults to the task's user).",
            parameters={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                "additionalProperties": False,
            },
            handler=_list_notes,
        ),
        Tool(
            name="get_note",
            description="Fetch a single note by id, including its full markdown content.",
            parameters={
                "type": "object",
                "properties": {"note_id": {"type": "string"}},
                "required": ["note_id"],
                "additionalProperties": False,
            },
            handler=_get_note,
        ),
        Tool(
            name="create_note",
            description="Persist a new markdown note for the task's user.",
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                    "style_name": {"type": "string"},
                    "custom_instructions": {"type": "string"},
                },
                "required": ["content"],
                "additionalProperties": False,
            },
            handler=_create_note,
        ),
        Tool(
            name="list_translations",
            description="List saved translations for a user.",
            parameters={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                "additionalProperties": False,
            },
            handler=_list_translations,
        ),
        Tool(
            name="http_fetch",
            description="HTTP GET a URL and return the response body (truncated).",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "max_bytes": {"type": "integer", "minimum": 1, "maximum": 65536},
                },
                "required": ["url"],
                "additionalProperties": False,
            },
            handler=_http_fetch,
        ),
        Tool(
            name="finish",
            description=(
                "Mark the task as complete. Call this exactly once, at the end, with a short "
                "user-facing summary of what was accomplished."
            ),
            parameters={
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
                "additionalProperties": False,
            },
            handler=_finish,
        ),
    ]
