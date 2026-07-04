import pytest
from fastapi import HTTPException

from app.core.config import Settings
from app.db.models import Note
from app.schemas.note import (
    NoteCreate,
    NoteRegenerateRequest,
    NoteStyleConfig,
    NoteUpdate,
)
from app.services.notes import NoteService

from conftest import FakeLLM


def make_service(session, llm) -> NoteService:
    return NoteService(session, Settings(), llm)


async def test_create_and_list_scoped_to_user(session):
    service = make_service(session, FakeLLM())

    alice_note = await service.create(NoteCreate(user_id="alice", title="A", content="hello"))
    await service.create(NoteCreate(user_id="bob", title="B", content="world"))

    alice_notes = await service.list_for_user("alice")
    bob_notes = await service.list_for_user("bob")

    assert [n.id for n in alice_notes] == [alice_note.id]
    assert len(bob_notes) == 1
    assert bob_notes[0].title == "B"


async def test_update_partial_fields(session):
    service = make_service(session, FakeLLM())
    note = await service.create(NoteCreate(user_id="alice", title="A", content="hello"))

    updated = await service.update(
        note.id, "alice", NoteUpdate(title="renamed")
    )

    assert updated.title == "renamed"
    assert updated.content == "hello"


async def test_update_style_persists(session):
    service = make_service(session, FakeLLM())
    note = await service.create(NoteCreate(user_id="alice"))

    updated = await service.update(
        note.id,
        "alice",
        NoteUpdate(style=NoteStyleConfig(style_name="meeting")),
    )

    assert updated.style_name == "meeting"


async def test_get_and_update_forbidden_for_other_user(session):
    service = make_service(session, FakeLLM())
    note = await service.create(NoteCreate(user_id="alice", content="secret"))

    with pytest.raises(HTTPException) as exc:
        await service.get(note.id, "bob")
    assert exc.value.status_code == 404

    with pytest.raises(HTTPException) as exc:
        await service.update(note.id, "bob", NoteUpdate(title="hacked"))
    assert exc.value.status_code == 404


async def test_delete_archives_note(session):
    service = make_service(session, FakeLLM())
    note = await service.create(NoteCreate(user_id="alice"))

    await service.delete(note.id, "alice")

    assert await service.list_for_user("alice") == []
    with pytest.raises(HTTPException) as exc:
        await service.get(note.id, "alice")
    assert exc.value.status_code == 404


async def test_regenerate_replaces_content_and_uses_style_system_prompt(session):
    llm = FakeLLM(reply="# Rewritten\n\nBetter body.")
    service = make_service(session, llm)
    note = await service.create(
        NoteCreate(
            user_id="alice",
            content="# Original\n\nRough draft.",
            style=NoteStyleConfig(style_name="academic"),
        )
    )

    response = await service.regenerate(
        note.id,
        NoteRegenerateRequest(user_id="alice", prompt="Add an abstract section"),
    )

    assert response.content == "# Rewritten\n\nBetter body."
    assert response.style_name == "academic"
    context = llm.calls[0]
    assert context[0].role == "system"
    assert "academic" in context[0].content.lower()
    assert context[1].role == "user"
    assert "# Original" in context[1].content
    assert "Add an abstract section" in context[1].content
    # Content is persisted.
    fetched = await service.get(note.id, "alice")
    assert fetched.content == "# Rewritten\n\nBetter body."


async def test_regenerate_unknown_style_falls_back_to_default(session):
    llm = FakeLLM(reply="ok")
    service = make_service(session, llm)
    note = await service.create(NoteCreate(user_id="alice"))
    # Simulate a style name that is no longer registered (e.g. removed from settings).
    db_note = await session.get(Note, note.id)
    db_note.style_name = "retired-style"
    await session.commit()

    response = await service.regenerate(
        note.id, NoteRegenerateRequest(user_id="alice", prompt="rewrite")
    )

    assert response.content == "ok"
    # System prompt should match the default template even though the note names an unknown style.
    system_content = llm.calls[0][0].content.lower()
    assert "clean, well-structured markdown" in system_content


async def test_regenerate_forbidden_for_other_user(session):
    service = make_service(session, FakeLLM())
    note = await service.create(NoteCreate(user_id="alice"))

    with pytest.raises(HTTPException) as exc:
        await service.regenerate(
            note.id, NoteRegenerateRequest(user_id="bob", prompt="hijack")
        )
    assert exc.value.status_code == 404


async def test_regenerate_uses_style_override(session):
    llm = FakeLLM(reply="ok")
    service = make_service(session, llm)
    note = await service.create(
        NoteCreate(user_id="alice", style=NoteStyleConfig(style_name="default"))
    )

    await service.regenerate(
        note.id,
        NoteRegenerateRequest(
            user_id="alice",
            prompt="convert to meeting minutes",
            style_override=NoteStyleConfig(style_name="meeting"),
        ),
    )

    system_content = llm.calls[0][0].content.lower()
    assert "meeting notes" in system_content
