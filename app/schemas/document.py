from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    conversation_id: str
    filename: str
    content_type: str | None = None
    size_bytes: int
    chunk_count: int
    created_at: datetime


class Citation(BaseModel):
    marker: int
    document_id: str
    filename: str
    page: int | None = None
    chunk_index: int
    score: float
    snippet: str
