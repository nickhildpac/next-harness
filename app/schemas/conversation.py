from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ToneName = Literal["professional", "friendly", "concise", "empathetic", "technical", "humorous", "custom"]

TONE_ALIASES = {
    "friendly": "friendly",
    "formal": "professional",
    "playful": "humorous",
    "empathetic": "empathetic",
    "direct": "concise",
    "professional": "professional",
    "concise": "concise",
    "technical": "technical",
    "humorous": "humorous",
    "custom": "custom",
}


def normalize_tone(value: str) -> str:
    return TONE_ALIASES.get(value.strip().lower(), value.strip().lower())


class ToneConfig(BaseModel):
    tone_name: ToneName = "professional"
    custom_persona: str | None = Field(default=None, max_length=800)

    @model_validator(mode="before")
    @classmethod
    def accept_tone_alias(cls, data):
        if isinstance(data, str):
            return {"tone_name": normalize_tone(data)}
        if isinstance(data, dict) and "tone" in data and "tone_name" not in data:
            data = {**data, "tone_name": normalize_tone(data["tone"])}
        return data

    @field_validator("tone_name", mode="before")
    @classmethod
    def sanitize_tone_name(cls, value: str) -> str:
        return normalize_tone(value)

    @field_validator("custom_persona")
    @classmethod
    def sanitize_persona(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = " ".join(value.replace("{", "").replace("}", "").split())
        if not cleaned:
            return None
        return cleaned


class ConversationCreate(BaseModel):
    user_id: str = Field(default="anonymous", min_length=1, max_length=128)
    title: str | None = Field(default=None, max_length=255)
    tone: ToneConfig = Field(default_factory=ToneConfig)
    participants: list[str] | None = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def accept_tone_field(cls, data):
        if isinstance(data, dict) and "tone" in data and isinstance(data["tone"], str):
            return {**data, "tone": ToneConfig(tone_name=normalize_tone(data["tone"]))}
        return data

    @field_validator("participants")
    @classmethod
    def validate_participants(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        cleaned = [participant.strip() for participant in value]
        if len(cleaned) != 2 or len(set(cleaned)) != 2:
            raise ValueError("participants must be exactly two distinct users")
        if any(not participant or len(participant) > 128 for participant in cleaned):
            raise ValueError("each participant must be 1-128 characters")
        return cleaned


class ConversationToneUpdate(ToneConfig):
    pass


class ConversationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str
    second_user_id: str | None = None
    kind: str = "assistant"
    title: str | None
    tone_name: str
    custom_persona: str | None
    is_archived: bool
    created_at: datetime
    updated_at: datetime


class MessageCreate(BaseModel):
    user_id: str = Field(default="anonymous", min_length=1, max_length=128)
    content: str = Field(min_length=1, max_length=12000)
    tone_override: ToneConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def accept_text_field(cls, data):
        if isinstance(data, dict) and "text" in data and "content" not in data:
            return {**data, "content": data["text"]}
        return data


class MessageResponse(BaseModel):
    id: str
    conversation_id: str
    user_id: str
    role: str
    content: str
    token_count: int
    model: str | None
    created_at: datetime


class ConversationDetail(ConversationResponse):
    messages: list[MessageResponse]
    summary: str | None = None


class ChatResponse(BaseModel):
    conversation_id: str
    user_message: MessageResponse
    # None for two-user conversations, where the LLM only replies via /suggest.
    assistant_message: MessageResponse | None = None
    token_usage: dict[str, int]


class SuggestRequest(BaseModel):
    for_user: str = Field(min_length=1, max_length=128)
    tone_override: ToneConfig | None = None
    persist: bool = False

    @model_validator(mode="before")
    @classmethod
    def accept_user_aliases(cls, data):
        if isinstance(data, dict) and "for_user" not in data:
            for alias in ("as_user", "user_id"):
                if alias in data:
                    return {**data, "for_user": data[alias]}
        return data


class SuggestResponse(BaseModel):
    conversation_id: str
    for_user: str
    content: str
    model: str
    message: MessageResponse | None = None
    token_usage: dict[str, int]


class PaginatedMessages(BaseModel):
    items: list[MessageResponse]
    limit: int
    offset: int
    total: int
