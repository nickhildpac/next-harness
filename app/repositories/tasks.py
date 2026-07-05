from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import AgentTask, AgentTaskStep, TaskStatus, TaskStepKind


class TaskRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        *,
        user_id: str,
        goal: str,
        max_steps: int,
        allowed_tools: list[str] | None,
    ) -> AgentTask:
        task = AgentTask(
            user_id=user_id,
            goal=goal,
            max_steps=max_steps,
            allowed_tools=allowed_tools,
            status=TaskStatus.pending.value,
        )
        self.session.add(task)
        await self.session.flush()
        return task

    async def get(self, task_id: str) -> AgentTask | None:
        stmt = select(AgentTask).where(AgentTask.id == task_id)
        return await self.session.scalar(stmt)

    async def get_with_steps(self, task_id: str) -> AgentTask | None:
        stmt = (
            select(AgentTask)
            .options(selectinload(AgentTask.steps))
            .where(AgentTask.id == task_id)
        )
        return await self.session.scalar(stmt)

    async def list_for_user(self, user_id: str, limit: int = 50) -> list[AgentTask]:
        stmt = (
            select(AgentTask)
            .where(AgentTask.user_id == user_id)
            .order_by(AgentTask.updated_at.desc(), AgentTask.created_at.desc())
            .limit(limit)
        )
        return list(await self.session.scalars(stmt))

    async def set_status(
        self,
        task: AgentTask,
        status: TaskStatus,
        *,
        result_summary: str | None = None,
        error: str | None = None,
        model: str | None = None,
    ) -> AgentTask:
        task.status = status.value
        if result_summary is not None:
            task.result_summary = result_summary
        if error is not None:
            task.error = error
        if model is not None:
            task.model = model
        await self.session.flush()
        return task

    async def bump_steps(self, task: AgentTask, steps_taken: int) -> None:
        task.steps_taken = steps_taken
        await self.session.flush()

    async def add_step(
        self,
        task: AgentTask,
        *,
        step_index: int,
        kind: TaskStepKind,
        tool_name: str | None = None,
        content: str | None = None,
        payload: Any = None,
        ok: bool | None = None,
    ) -> AgentTaskStep:
        step = AgentTaskStep(
            task_id=task.id,
            step_index=step_index,
            kind=kind.value,
            tool_name=tool_name,
            content=content,
            payload=payload,
            ok=ok,
        )
        self.session.add(step)
        await self.session.flush()
        return step
