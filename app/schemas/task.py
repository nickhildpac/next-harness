from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TaskCreate(BaseModel):
    goal: str = Field(min_length=1, max_length=4000)
    user_id: str = Field(default="anonymous", min_length=1, max_length=128)
    thread_id: str | None = None
    max_steps: int = Field(default=8, ge=1, le=32)
    allowed_tools: list[str] | None = None
    run: bool = True

    @model_validator(mode="before")
    @classmethod
    def accept_prompt_alias(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for alias in ("prompt", "objective", "task"):
                if alias in data and "goal" not in data:
                    return {**data, "goal": data[alias]}
        return data

    @field_validator("allowed_tools")
    @classmethod
    def clean_allowed(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        cleaned = [t.strip() for t in value if isinstance(t, str) and t.strip()]
        return cleaned or None


class ThreadCreate(TaskCreate):
    title: str | None = Field(default=None, max_length=255)


class TaskStepResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    step_index: int
    kind: str
    tool_name: str | None = None
    content: str | None = None
    payload: Any = None
    ok: bool | None = None
    created_at: datetime


class TaskResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str
    thread_id: str | None = None
    sequence_index: int
    goal: str
    status: str
    max_steps: int
    steps_taken: int
    model: str | None = None
    result_summary: str | None = None
    error: str | None = None
    allowed_tools: list[str] | None = None
    created_at: datetime
    updated_at: datetime


class TaskDetail(TaskResponse):
    steps: list[TaskStepResponse] = Field(default_factory=list)


class ThreadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str
    title: str | None = None
    created_at: datetime
    updated_at: datetime
    tasks: list[TaskResponse] = Field(default_factory=list)


class ThreadDetail(ThreadResponse):
    tasks: list[TaskDetail] = Field(default_factory=list)


class ToolInfo(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]
