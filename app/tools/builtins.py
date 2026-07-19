from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException

from app.repositories.notes import NoteRepository
from app.repositories.translations import TranslationRepository
from app.schemas.note import NoteCreate, NoteUpdate
from app.schemas.translation import TranslationCreate
from app.services.rag import RagService
from app.services.translations import TranslationService
from app.tools.registry import Tool, ToolContext, ToolError


def _require_session(context: ToolContext):
    if context.session is None:
        raise ToolError("no database session available for this tool")
    return context.session


def _require_user(context: ToolContext) -> str:
    if not context.user_id:
        raise ToolError("no user_id available on the tool context")
    return context.user_id


def _require_task(context: ToolContext) -> str:
    if not context.task_id:
        raise ToolError("no task_id available on the tool context")
    return context.task_id


def _require_rag(context: ToolContext) -> RagService:
    session = _require_session(context)
    if context.settings is None or context.embeddings is None or context.vectorstore is None:
        raise ToolError("RAG services are not available on the tool context")
    return RagService(session, context.settings, context.embeddings, context.vectorstore)


def _require_translation_service(context: ToolContext) -> TranslationService:
    session = _require_session(context)
    if context.settings is None or context.llm is None:
        raise ToolError("LLM services are not available on the tool context")
    return TranslationService(session, context.settings, context.llm)


def _tool_error_from_http(exc: HTTPException) -> ToolError:
    return ToolError(str(exc.detail))


async def _now(_: dict[str, Any], __: ToolContext) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {"iso": now.isoformat(), "epoch": int(now.timestamp())}


async def _list_notes(args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    session = _require_session(context)
    user_id = _require_user(context)
    notes = await NoteRepository(session).list_for_user(user_id)
    limit = int(args.get("limit") or 20)
    items = [
        {
            "id": n.id,
            "title": n.title,
            "style_name": n.style_name,
            "updated_at": n.updated_at.isoformat(),
        }
        for n in notes[:limit]
    ]
    return {"count": len(items), "items": items}


async def _get_note(args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    session = _require_session(context)
    note_id = args.get("note_id")
    if not note_id:
        raise ToolError("note_id is required")
    note = await NoteRepository(session).get(note_id)
    if note is None or note.user_id != _require_user(context):
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
    try:
        payload = NoteCreate.model_validate(
            {
                "user_id": user_id,
                "title": args.get("title"),
                "content": args.get("content") or "",
                "style": {
                    "style_name": args.get("style_name") or "default",
                    "custom_instructions": args.get("custom_instructions"),
                },
            }
        )
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    if not payload.content:
        raise ToolError("content is required")
    note = await NoteRepository(session).create(
        user_id=payload.user_id,
        title=payload.title,
        content=payload.content,
        style_name=payload.style.style_name,
        custom_instructions=payload.style.custom_instructions,
    )
    return {
        "id": note.id,
        "title": note.title,
        "style_name": note.style_name,
        "updated_at": note.updated_at.isoformat(),
    }


async def _update_note(args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    session = _require_session(context)
    note_id = args.get("note_id")
    if not note_id:
        raise ToolError("note_id is required")
    note = await NoteRepository(session).get(note_id)
    if note is None or note.user_id != _require_user(context):
        raise ToolError(f"note '{note_id}' not found")
    style = None
    if args.get("style_name") or args.get("custom_instructions"):
        style = {
            "style_name": args.get("style_name") or note.style_name,
            "custom_instructions": args.get("custom_instructions"),
        }
    try:
        payload = NoteUpdate.model_validate(
            {
                key: value
                for key, value in {
                    "title": args.get("title"),
                    "content": args.get("content"),
                    "style": style,
                }.items()
                if value is not None
            }
        )
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    updates = payload.model_dump(exclude_unset=True)
    style_updates = updates.pop("style", None)
    if style_updates is not None:
        updates["style_name"] = style_updates["style_name"]
        updates["custom_instructions"] = style_updates.get("custom_instructions")
    if not updates:
        raise ToolError("at least one update field is required")
    updated = await NoteRepository(session).apply_updates(note, updates)
    return {
        "id": updated.id,
        "title": updated.title,
        "style_name": updated.style_name,
        "updated_at": updated.updated_at.isoformat(),
    }


async def _list_translations(args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    session = _require_session(context)
    user_id = _require_user(context)
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


async def _translate_text(args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    user_id = _require_user(context)
    service = _require_translation_service(context)
    try:
        payload = TranslationCreate.model_validate(
            {
                "user_id": user_id,
                "source_text": args.get("source_text"),
                "target_language": args.get("target_language"),
            }
        )
        result = await service.translate(payload, commit=False)
    except HTTPException as exc:
        raise _tool_error_from_http(exc) from exc
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    save = args.get("save")
    if save is False:
        row = await service.repo.get(result.translation_id)
        if row is not None:
            await service.repo.archive(row)
    return {
        "translation_id": result.translation_id if save is not False else None,
        "saved": save is not False,
        "translated_text": result.translated_text,
        "romanized_text": result.romanized_text,
        "target_language": result.target_language,
        "model": result.model,
        "token_usage": result.token_usage,
    }


async def _ingest_task_document(args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    task_id = _require_task(context)
    rag = _require_rag(context)
    filename = args.get("filename")
    content = args.get("content")
    if not filename:
        raise ToolError("filename is required")
    if not isinstance(content, str) or not content.strip():
        raise ToolError("content is required")
    try:
        document = await rag.ingest_task_document(
            task_id,
            filename=filename,
            content_type=args.get("content_type"),
            data=content.encode("utf-8"),
            commit=False,
        )
    except HTTPException as exc:
        raise _tool_error_from_http(exc) from exc
    return {
        "id": document.id,
        "filename": document.filename,
        "content_type": document.content_type,
        "size_bytes": document.size_bytes,
        "chunk_count": document.chunk_count,
        "created_at": document.created_at.isoformat(),
    }


async def _list_task_documents(_: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    task_id = _require_task(context)
    rag = _require_rag(context)
    try:
        documents = await rag.list_task_documents(task_id)
    except HTTPException as exc:
        raise _tool_error_from_http(exc) from exc
    return {
        "count": len(documents),
        "items": [
            {
                "id": document.id,
                "filename": document.filename,
                "content_type": document.content_type,
                "size_bytes": document.size_bytes,
                "chunk_count": document.chunk_count,
                "created_at": document.created_at.isoformat(),
            }
            for document in documents
        ],
    }


async def _search_task_documents(args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    task_id = _require_task(context)
    rag = _require_rag(context)
    query = args.get("query")
    if not query:
        raise ToolError("query is required")
    try:
        token_budget = int(args.get("token_budget") or context.settings.rag_token_budget)
    except (TypeError, ValueError) as exc:
        raise ToolError("token_budget must be an integer") from exc
    try:
        chunks = await rag.retrieve_task(task_id, query, token_budget=token_budget)
    except HTTPException as exc:
        raise _tool_error_from_http(exc) from exc
    citations = rag.build_citations(chunks)
    return {
        "count": len(chunks),
        "context": rag.format_context(chunks) if chunks else "",
        "citations": [citation.model_dump() for citation in citations],
    }


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


REGISTRY: list[Tool] = [
    Tool(
        name="now",
        description="Return the current UTC time.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        executor=_now,
    ),
    Tool(
        name="list_notes",
        description="List markdown notes owned by the task user.",
        input_schema={
            "type": "object",
            "properties": {
                "limit": {"type": ["integer", "null"], "minimum": 1, "maximum": 100},
            },
            "required": ["limit"],
            "additionalProperties": False,
        },
        executor=_list_notes,
    ),
    Tool(
        name="get_note",
        description="Fetch a single note by id, including its full markdown content.",
        input_schema={
            "type": "object",
            "properties": {"note_id": {"type": "string"}},
            "required": ["note_id"],
            "additionalProperties": False,
        },
        executor=_get_note,
    ),
    Tool(
        name="create_note",
        description="Persist a new markdown note for the task's user.",
        input_schema={
            "type": "object",
            "properties": {
                "title": {"type": ["string", "null"]},
                "content": {"type": "string"},
                "style_name": {"type": ["string", "null"]},
                "custom_instructions": {"type": ["string", "null"]},
            },
            "required": ["title", "content", "style_name", "custom_instructions"],
            "additionalProperties": False,
        },
        executor=_create_note,
        consequential=True,
    ),
    Tool(
        name="update_note",
        description="Update a markdown note owned by the task user.",
        input_schema={
            "type": "object",
            "properties": {
                "note_id": {"type": "string"},
                "title": {"type": ["string", "null"]},
                "content": {"type": ["string", "null"]},
                "style_name": {"type": ["string", "null"]},
                "custom_instructions": {"type": ["string", "null"]},
            },
            "required": ["note_id", "title", "content", "style_name", "custom_instructions"],
            "additionalProperties": False,
        },
        executor=_update_note,
        consequential=True,
    ),
    Tool(
        name="list_translations",
        description="List saved translations for the task user.",
        input_schema={
            "type": "object",
            "properties": {
                "limit": {"type": ["integer", "null"], "minimum": 1, "maximum": 100},
            },
            "required": ["limit"],
            "additionalProperties": False,
        },
        executor=_list_translations,
    ),
    Tool(
        name="translate_text",
        description="Translate text for the task user and save the translation by default.",
        input_schema={
            "type": "object",
            "properties": {
                "source_text": {"type": "string"},
                "target_language": {"type": "string"},
                "save": {"type": ["boolean", "null"]},
            },
            "required": ["source_text", "target_language", "save"],
            "additionalProperties": False,
        },
        executor=_translate_text,
        consequential=True,
    ),
    Tool(
        name="ingest_task_document",
        description=(
            "Create a task-scoped RAG document from text or markdown content supplied in JSON."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "filename": {"type": "string"},
                "content": {"type": "string"},
                "content_type": {"type": ["string", "null"]},
            },
            "required": ["filename", "content", "content_type"],
            "additionalProperties": False,
        },
        executor=_ingest_task_document,
        consequential=True,
    ),
    Tool(
        name="list_task_documents",
        description="List RAG documents ingested for the current task.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        executor=_list_task_documents,
    ),
    Tool(
        name="search_task_documents",
        description=(
            "Search task-scoped RAG documents. Use this before answering questions that "
            "depend on ingested document content."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "token_budget": {"type": ["integer", "null"], "minimum": 1},
            },
            "required": ["query", "token_budget"],
            "additionalProperties": False,
        },
        executor=_search_task_documents,
    ),
    Tool(
        name="http_fetch",
        description="HTTP GET a URL and return the response body (truncated).",
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "max_bytes": {"type": ["integer", "null"], "minimum": 1, "maximum": 65536},
            },
            "required": ["url", "max_bytes"],
            "additionalProperties": False,
        },
        executor=_http_fetch,
    ),
    Tool(
        name="finish",
        description=(
            "Mark the task as complete. Call this exactly once, at the end, with a short "
            "user-facing summary of what was accomplished."
        ),
        input_schema={
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
            "additionalProperties": False,
        },
        executor=_finish,
    ),
]
