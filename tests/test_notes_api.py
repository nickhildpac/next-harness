from app.main import app
from app.schemas.note import (
    NoteCreate,
    NoteRegenerateRequest,
    NoteStyleConfig,
    NoteUpdate,
)


def test_openapi_contains_note_routes():
    paths = app.openapi()["paths"]

    assert "/notes" in paths
    assert "/notes/{note_id}" in paths
    assert "/notes/{note_id}/regenerate" in paths
    assert "/note-styles" in paths


def test_note_style_alias_normalization():
    style = NoteStyleConfig.model_validate({"style_name": "  Scholarly "})
    assert style.style_name == "academic"

    style = NoteStyleConfig.model_validate({"style_name": "Minutes"})
    assert style.style_name == "meeting"


def test_note_style_custom_instructions_sanitized():
    style = NoteStyleConfig.model_validate(
        {"style_name": "custom", "custom_instructions": "  Use {bold} words  \n"}
    )
    assert style.custom_instructions == "Use bold words"


def test_note_create_defaults():
    payload = NoteCreate.model_validate({})
    assert payload.user_id == "anonymous"
    assert payload.content == ""
    assert payload.style.style_name == "default"


def test_note_update_allows_partial_payload():
    payload = NoteUpdate.model_validate({"title": "renamed"})
    dumped = payload.model_dump(exclude_unset=True)
    assert dumped == {"title": "renamed"}


def test_note_regenerate_requires_prompt():
    import pytest

    with pytest.raises(ValueError):
        NoteRegenerateRequest.model_validate({"user_id": "alice"})
