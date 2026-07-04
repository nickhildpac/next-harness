from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


ToneName = Literal["professional", "friendly", "concise", "empathetic", "technical", "humorous", "custom"]


class ToneConfig(BaseModel):
    tone_name: ToneName = "professional"
    custom_persona: str | None = Field(default=None, max_length=800)

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
    user_id: str = Field(min_length=1, max_length=128)
    title: str | None = Field(default=None, max_length=255)
    tone: ToneConfig = Field(default_factory=ToneConfig)


class ConversationToneUpdate(ToneConfig):
    pass


class ConversationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str
    title: str | None
    tone_name: str
    custom_persona: str | None
    is_archived: bool
    created_at: datetime
    updated_at: datetime


class MessageCreate(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    content: str = Field(min_length=1, max_length=12000)
    tone_override: ToneConfig | None = None


class MessageResponse(BaseModel):
    id: str
    conversation_id: str
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
    assistant_message: MessageResponse
    token_usage: dict[str, int]


class PaginatedMessages(BaseModel):
    items: list[MessageResponse]
    limit: int
    offset: int
    total: int

