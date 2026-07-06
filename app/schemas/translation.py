from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TranslationCreate(BaseModel):
    user_id: str = Field(default="anonymous", min_length=1, max_length=128)
    source_text: str = Field(min_length=1, max_length=10_000)
    target_language: str | None = Field(default=None, min_length=1, max_length=64)
    session_id: str | None = None

    @model_validator(mode="after")
    def require_language_for_new_session(self) -> "TranslationCreate":
        if self.session_id is None and not self.target_language:
            raise ValueError("target_language is required when starting a new translation session")
        return self


class TranslationTurnResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    turn_index: int
    source_text: str
    target_language: str
    translated_text: str
    romanized_text: str
    model: str | None
    created_at: datetime


class TranslationSessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str
    title: str | None
    target_language: str
    is_archived: bool
    created_at: datetime
    updated_at: datetime
    turns: list[TranslationTurnResponse] = Field(default_factory=list)


class TranslationSessionSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str
    title: str | None
    target_language: str
    preview: str
    turn_count: int
    is_archived: bool
    created_at: datetime
    updated_at: datetime


class TranslateResponse(BaseModel):
    session_id: str
    turn_id: str
    translated_text: str
    romanized_text: str
    model: str
    target_language: str
    token_usage: dict[str, int]

    @property
    def translation_id(self) -> str:
        """Backward-compatible alias for session_id."""
        return self.session_id
