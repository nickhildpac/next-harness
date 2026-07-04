from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class TranslationCreate(BaseModel):
    user_id: str = Field(default="anonymous", min_length=1, max_length=128)
    source_text: str = Field(min_length=1, max_length=10_000)
    target_language: str = Field(min_length=1, max_length=64)


class TranslationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str
    title: str | None
    source_text: str
    target_language: str
    translated_text: str
    romanized_text: str
    model: str | None
    is_archived: bool
    created_at: datetime
    updated_at: datetime


class TranslateResponse(BaseModel):
    translation_id: str
    translated_text: str
    romanized_text: str
    model: str
    target_language: str
    token_usage: dict[str, int]
