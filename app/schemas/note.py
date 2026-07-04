from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


NoteStyleName = Literal["default", "academic", "meeting", "blog", "custom"]

NOTE_STYLE_ALIASES = {
    "default": "default",
    "standard": "default",
    "academic": "academic",
    "scholarly": "academic",
    "meeting": "meeting",
    "minutes": "meeting",
    "blog": "blog",
    "article": "blog",
    "custom": "custom",
}


def normalize_style(value: str) -> str:
    return NOTE_STYLE_ALIASES.get(value.strip().lower(), value.strip().lower())


class NoteStyleConfig(BaseModel):
    style_name: NoteStyleName = "default"
    custom_instructions: str | None = Field(default=None, max_length=800)

    @field_validator("style_name", mode="before")
    @classmethod
    def sanitize_style_name(cls, value: str) -> str:
        return normalize_style(value)

    @field_validator("custom_instructions")
    @classmethod
    def sanitize_instructions(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = " ".join(value.replace("{", "").replace("}", "").split())
        return cleaned or None


class NoteCreate(BaseModel):
    user_id: str = Field(default="anonymous", min_length=1, max_length=128)
    title: str | None = Field(default=None, max_length=255)
    content: str = Field(default="", max_length=200_000)
    style: NoteStyleConfig = Field(default_factory=NoteStyleConfig)


class NoteUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=255)
    content: str | None = Field(default=None, max_length=200_000)
    style: NoteStyleConfig | None = None


class NoteResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str
    title: str | None
    content: str
    style_name: str
    custom_instructions: str | None
    is_archived: bool
    created_at: datetime
    updated_at: datetime


class NoteRegenerateRequest(BaseModel):
    user_id: str = Field(default="anonymous", min_length=1, max_length=128)
    prompt: str = Field(min_length=1, max_length=4000)
    style_override: NoteStyleConfig | None = None


class NoteRegenerateResponse(BaseModel):
    note_id: str
    content: str
    model: str
    style_name: str
    token_usage: dict[str, int]
