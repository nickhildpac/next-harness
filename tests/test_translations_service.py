from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from app.core.config import Settings
from app.schemas.translation import TranslationCreate
from app.services.translations import TranslationService, _parse_response

from conftest import FakeLLM


def make_service(session, llm) -> TranslationService:
    return TranslationService(session, Settings(), llm)


def test_parse_response_splits_translation_and_romanized():
    text = "TRANSLATION:\nこんにちは 世界\n\nROMANIZED:\nkonnichiwa sekai"
    translated, romanized = _parse_response(text)
    assert translated == "こんにちは 世界"
    assert romanized == "konnichiwa sekai"


def test_parse_response_translation_only():
    text = "TRANSLATION:\nHola mundo"
    translated, romanized = _parse_response(text)
    assert translated == "Hola mundo"
    assert romanized == ""


def test_parse_response_no_markers_returns_raw():
    text = "just some text"
    translated, romanized = _parse_response(text)
    assert translated == "just some text"
    assert romanized == ""


def test_parse_response_romanized_before_translation_marker_absent():
    text = "hello\nROMANIZED:\nhola"
    translated, romanized = _parse_response(text)
    assert translated == "hello"
    assert romanized == "hola"


async def test_translate_persists_and_returns_result(session):
    llm = FakeLLM(reply="TRANSLATION:\nHola\n\nROMANIZED:\nHola")
    service = make_service(session, llm)

    result = await service.translate(
        TranslationCreate(user_id="alice", source_text="Hello", target_language="Spanish")
    )

    assert result.translated_text == "Hola"
    assert result.romanized_text == "Hola"
    assert result.target_language == "Spanish"
    assert result.model == "fake-model"
    assert result.session_id
    assert result.turn_id

    listed = await service.list_for_user("alice")
    assert len(listed) == 1
    assert listed[0].preview == "Hola"
    assert listed[0].target_language == "Spanish"
    assert listed[0].turn_count == 1

    detail = await service.get(result.session_id, "alice")
    assert len(detail.turns) == 1
    assert detail.turns[0].source_text == "Hello"


async def test_translate_can_defer_commit_to_caller(session, monkeypatch):
    llm = FakeLLM(reply="TRANSLATION:\nHola\n\nROMANIZED:\nHola")
    service = make_service(session, llm)
    commit = AsyncMock()
    monkeypatch.setattr(session, "commit", commit)

    result = await service.translate(
        TranslationCreate(user_id="alice", source_text="Hello", target_language="Spanish"),
        commit=False,
    )

    commit.assert_not_awaited()
    detail = await service.get(result.session_id, "alice")
    assert detail.turns[0].translated_text == "Hola"


async def test_translate_appends_turn_to_existing_session(session):
    llm = FakeLLM(reply="TRANSLATION:\nHola\n\nROMANIZED:\nHola")
    service = make_service(session, llm)

    first = await service.translate(
        TranslationCreate(user_id="alice", source_text="Hello", target_language="Spanish")
    )
    llm.reply = "TRANSLATION:\nAdiós\n\nROMANIZED:\nAdios"
    second = await service.translate(
        TranslationCreate(
            user_id="alice",
            source_text="Goodbye",
            session_id=first.session_id,
        )
    )

    assert second.session_id == first.session_id
    assert second.translated_text == "Adiós"
    assert len(llm.calls) == 2
    assert len(llm.calls[1]) == 4
    assert llm.calls[1][-1].content == "Goodbye"

    detail = await service.get(first.session_id, "alice")
    assert len(detail.turns) == 2
    assert detail.turns[1].source_text == "Goodbye"


async def test_translate_scoped_to_user(session):
    service = make_service(session, FakeLLM(reply="TRANSLATION:\nX\nROMANIZED:\nX"))
    created = await service.translate(
        TranslationCreate(user_id="alice", source_text="hi", target_language="French")
    )

    with pytest.raises(HTTPException) as exc:
        await service.get(created.session_id, "bob")
    assert exc.value.status_code == 404


async def test_delete_archives_translation(session):
    service = make_service(session, FakeLLM(reply="TRANSLATION:\nX\nROMANIZED:\nX"))
    created = await service.translate(
        TranslationCreate(user_id="alice", source_text="hi", target_language="French")
    )

    await service.delete(created.session_id, "alice")

    assert await service.list_for_user("alice") == []
