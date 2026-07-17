from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.mcp_client import McpStdioSession, McpStreamableHttpSession
from app.api.ndjson import ndjson_event
from app.core.config import Settings
from app.db.base import utcnow
from app.db.models import AgentTask, AgentThread, TaskStatus, TaskStepKind
from app.orchestration.agent_graph import AgentGraph, AgentRun, StepRecord
from app.ports.embeddings import EmbeddingsClient
from app.ports.llm import GenerationParams, LLMClient
from app.ports.tools import ToolInvoker
from app.ports.vectorstore import VectorStore
from app.repositories.documents import DocumentRepository
from app.repositories.tasks import TaskRepository
from app.schemas.task import (
    TaskCreate,
    TaskDetail,
    TaskResponse,
    TaskStepResponse,
    ThreadCreate,
    ThreadDetail,
    ThreadResponse,
    ToolInfo,
)
from app.services.rag import RagService
from app.tools.mcp_invoker import HybridToolInvoker
from app.tools.registry import Tool, ToolContext, ToolRegistry, build_default_registry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskDocumentUpload:
    filename: str
    content_type: str | None
    data: bytes


class TaskService:
    def __init__(
        self,
        session: AsyncSession,
        settings: Settings,
        llm: LLMClient,
        http_client: httpx.AsyncClient | None,
        embeddings: EmbeddingsClient | None = None,
        vectorstore: VectorStore | None = None,
        *,
        use_mcp_tools: bool = True,
        mcp_auth_token: str | None = None,
        mcp_http_client: httpx.AsyncClient | None = None,
    ):
        self.session = session
        self.settings = settings
        self.llm = llm
        self.http_client = http_client
        self.embeddings = embeddings
        self.vectorstore = vectorstore
        self.use_mcp_tools = use_mcp_tools
        self.mcp_auth_token = mcp_auth_token
        self.mcp_http_client = mcp_http_client
        self.repo = TaskRepository(session)
        self._registry = build_default_registry()

    def available_tools(self) -> list[ToolInfo]:
        return [
            ToolInfo(name=spec.name, description=spec.description, parameters=spec.parameters)
            for spec in self._registry.specs()
        ]

    async def create_task(self, payload: TaskCreate) -> TaskDetail:
        task = await self._create_task_record(payload)
        if not payload.run:
            await self.session.commit()
            return await self._detail(task.id)
        return await self._run_task(task)

    async def stream_create_and_run(self, payload: TaskCreate) -> AsyncIterator[str]:
        """Create a task and stream the agent run as NDJSON events."""
        try:
            task = await self._create_task_record(payload)
            await self.session.commit()
        except HTTPException as exc:
            await self.session.rollback()
            yield ndjson_event("error", {"error": exc.detail})
            return
        except Exception as exc:  # noqa: BLE001
            await self.session.rollback()
            yield ndjson_event("error", {"error": f"{exc.__class__.__name__}: {exc}"})
            return
        async for frame in self.stream_run(task):
            yield frame

    async def create_thread(self, payload: ThreadCreate) -> ThreadDetail:
        thread = await self.repo.create_thread(
            user_id=payload.user_id,
            title=payload.title or payload.goal[:255],
        )
        task_payload = TaskCreate(**payload.model_dump(exclude={"title"}))
        task = await self._create_task_record(task_payload, thread=thread)
        if payload.run:
            await self._run_task(task)
        else:
            await self.session.commit()
        return await self._thread_detail(thread.id, payload.user_id)

    async def stream_create_thread_and_run(self, payload: ThreadCreate) -> AsyncIterator[str]:
        try:
            thread = await self.repo.create_thread(
                user_id=payload.user_id,
                title=payload.title or payload.goal[:255],
            )
            task_payload = TaskCreate(**payload.model_dump(exclude={"title"}))
            task = await self._create_task_record(task_payload, thread=thread)
            await self.session.commit()
        except HTTPException as exc:
            await self.session.rollback()
            yield ndjson_event("error", {"error": exc.detail})
            return
        except Exception as exc:  # noqa: BLE001
            await self.session.rollback()
            yield ndjson_event("error", {"error": f"{exc.__class__.__name__}: {exc}"})
            return
        async for frame in self.stream_run(task):
            yield frame

    async def create_thread_task(
        self, thread_id: str, payload: TaskCreate, user_id: str
    ) -> TaskDetail:
        thread = await self._thread_for_user(thread_id, user_id)
        task = await self._create_task_record(payload, thread=thread)
        if not payload.run:
            await self.session.commit()
            return await self._detail(task.id)
        return await self._run_task(task)

    async def stream_create_thread_task_and_run(
        self, thread_id: str, payload: TaskCreate, user_id: str
    ) -> AsyncIterator[str]:
        try:
            thread = await self._thread_for_user(thread_id, user_id)
            task = await self._create_task_record(payload, thread=thread)
            await self.session.commit()
        except HTTPException as exc:
            await self.session.rollback()
            yield ndjson_event("error", {"error": exc.detail})
            return
        except Exception as exc:  # noqa: BLE001
            await self.session.rollback()
            yield ndjson_event("error", {"error": f"{exc.__class__.__name__}: {exc}"})
            return
        async for frame in self.stream_run(task):
            yield frame

    async def run_task(self, task_id: str, user_id: str) -> TaskDetail:
        task = await self._task_for_user(task_id, user_id)
        if task.status != TaskStatus.pending.value:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Task cannot be run from status '{task.status}'.",
            )
        return await self._run_task(task)

    async def stream_run_task(self, task_id: str, user_id: str) -> AsyncIterator[str]:
        try:
            task = await self._task_for_user(task_id, user_id)
            if task.status != TaskStatus.pending.value:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Task cannot be run from status '{task.status}'.",
                )
        except HTTPException as exc:
            yield ndjson_event("error", {"error": exc.detail})
            return
        async for frame in self.stream_run(task):
            yield frame

    async def upload_task_document(self, task_id: str, user_id: str, document: TaskDocumentUpload):
        task = await self._task_for_user(task_id, user_id)
        if task.status != TaskStatus.pending.value:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Documents can only be attached while task is pending, not '{task.status}'.",
            )
        uploaded = await self._ingest_documents(task.id, [document])
        if uploaded != 1:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Document upload failed.",
            )
        documents = await DocumentRepository(self.session).list_by_task(task.id)
        return documents[-1]

    async def stream_run(self, task: AgentTask) -> AsyncIterator[str]:
        """Run the agent loop, persisting and yielding each step as NDJSON."""
        try:
            self._validate_allowed_tools(task.allowed_tools)
            uploaded_document_count = await self._task_document_count(task.id)
            await self.repo.set_status(task, TaskStatus.running)
            await self.session.commit()
            yield ndjson_event(
                "task",
                {
                    **TaskResponse.model_validate(task).model_dump(mode="json"),
                    "steps": [],
                },
            )

            context = self._tool_context(task, uploaded_document_count=uploaded_document_count)
            prior_context = await self._build_thread_context(task)
            final_run: AgentRun | None = None
            step_index = 0

            async with self._tool_invoker(task) as invoker:
                graph = AgentGraph(self.llm, invoker, max_steps=task.max_steps)
                async for mode, chunk in graph.stream(
                    task.goal,
                    self._params(),
                    context,
                    prior_context=prior_context,
                ):
                    if mode == "custom" and isinstance(chunk, StepRecord):
                        row = await self.repo.add_step(
                            task,
                            step_index=step_index,
                            kind=self._step_kind(chunk),
                            tool_name=chunk.tool_name,
                            content=chunk.content,
                            payload=chunk.payload,
                            ok=chunk.ok,
                        )
                        step_index += 1
                        await self.session.commit()
                        yield ndjson_event(
                            "step",
                            TaskStepResponse.model_validate(row).model_dump(mode="json"),
                        )
                    elif mode == "values" and isinstance(chunk, dict) and "run" in chunk:
                        final_run = chunk["run"]

            if final_run is None:
                final_run = AgentRun(
                    goal=task.goal,
                    errored=True,
                    error="run ended without a result",
                )

            await self._finalize_run_status(task, final_run)
            await self.session.commit()
            detail = await self._detail(task.id)
            yield ndjson_event("done", detail.model_dump(mode="json"))
            logger.info(
                "task_run_completed",
                extra={
                    "task_id": task.id,
                    "status": task.status,
                    "turns_used": final_run.turns_used,
                    "step_limit_hit": final_run.step_limit_hit,
                },
            )
        except HTTPException as exc:
            await self.session.rollback()
            yield ndjson_event("error", {"error": exc.detail})
        except Exception as exc:  # noqa: BLE001
            await self.session.rollback()
            logger.exception("task_stream_failed", extra={"task_id": getattr(task, "id", None)})
            try:
                await self.repo.set_status(
                    task, TaskStatus.failed, error=f"{exc.__class__.__name__}: {exc}"
                )
                await self.session.commit()
            except Exception:  # noqa: BLE001
                await self.session.rollback()
            yield ndjson_event("error", {"error": f"{exc.__class__.__name__}: {exc}"})

    async def _run_task(self, task: AgentTask) -> TaskDetail:
        self._validate_allowed_tools(task.allowed_tools)
        uploaded_document_count = await self._task_document_count(task.id)
        await self.repo.set_status(task, TaskStatus.running)
        await self.session.commit()

        context = self._tool_context(task, uploaded_document_count=uploaded_document_count)
        prior_context = await self._build_thread_context(task)
        async with self._tool_invoker(task) as invoker:
            graph = AgentGraph(self.llm, invoker, max_steps=task.max_steps)
            run = await graph.run(
                task.goal,
                self._params(),
                context,
                prior_context=prior_context,
            )

        await self._persist_run(task, run)
        await self.session.commit()
        logger.info(
            "task_run_completed",
            extra={
                "task_id": task.id,
                "status": task.status,
                "turns_used": run.turns_used,
                "step_limit_hit": run.step_limit_hit,
            },
        )
        return await self._detail(task.id)

    async def _create_task_record(
        self,
        payload: TaskCreate,
        *,
        thread: AgentThread | None = None,
    ) -> AgentTask:
        if thread is None and payload.thread_id:
            thread = await self._thread_for_user(payload.thread_id, payload.user_id)
        if thread is not None and thread.user_id != payload.user_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found")

        sequence_index = (
            await self.repo.next_sequence_index(thread.id) if thread is not None else 0
        )
        task = await self.repo.create(
            user_id=payload.user_id,
            thread_id=thread.id if thread is not None else None,
            sequence_index=sequence_index,
            goal=payload.goal,
            max_steps=payload.max_steps,
            allowed_tools=payload.allowed_tools,
        )
        if thread is not None:
            thread.updated_at = utcnow()
            await self.session.flush()
        return task

    @asynccontextmanager
    async def _tool_invoker(self, task: AgentTask) -> AsyncIterator[ToolInvoker]:
        if self.use_mcp_tools:
            if self.settings.mcp_transport == "stdio":
                async with McpStdioSession.from_settings(
                    self.settings,
                    user_id=task.user_id,
                    task_id=task.id,
                ) as mcp:
                    yield await HybridToolInvoker.create(mcp, allowed_tools=task.allowed_tools)
            else:
                async with McpStreamableHttpSession.from_settings(
                    self.settings,
                    auth_token=self.mcp_auth_token,
                    http_client=self.mcp_http_client,
                ) as mcp:
                    yield await HybridToolInvoker.create(mcp, allowed_tools=task.allowed_tools)
        else:
            yield self._scoped_registry(task.allowed_tools)

    def _tool_context(self, task: AgentTask, *, uploaded_document_count: int) -> ToolContext:
        # MCP tools open their own DB session in the child process; local registry
        # path still needs the request-scoped session and DI bag for builtins.
        if self.use_mcp_tools:
            return ToolContext(
                user_id=task.user_id,
                task_id=task.id,
                metadata={"uploaded_document_count": uploaded_document_count},
            )
        return ToolContext(
            session=self.session,
            http_client=self.http_client,
            user_id=task.user_id,
            task_id=task.id,
            settings=self.settings,
            llm=self.llm,
            embeddings=self.embeddings,
            vectorstore=self.vectorstore,
            metadata={"uploaded_document_count": uploaded_document_count},
        )

    async def list_tasks(self, user_id: str) -> list[TaskResponse]:
        tasks = await self.repo.list_for_user(user_id)
        return [TaskResponse.model_validate(t) for t in tasks]

    async def list_threads(self, user_id: str) -> list[ThreadResponse]:
        threads = await self.repo.list_threads_for_user(user_id)
        return [ThreadResponse.model_validate(thread) for thread in threads]

    async def get_thread(self, thread_id: str, user_id: str) -> ThreadDetail:
        return await self._thread_detail(thread_id, user_id)

    async def delete_thread(self, thread_id: str, user_id: str) -> None:
        thread = await self._thread_for_user(thread_id, user_id)
        await self.repo.delete_thread(thread)
        await self.session.commit()

    async def get_task(self, task_id: str, user_id: str | None = None) -> TaskDetail:
        return await self._detail(task_id, user_id)

    def _params(self) -> GenerationParams:
        return GenerationParams(
            model=self.settings.default_model,
            temperature=0.2,
            top_p=0.9,
            timeout_seconds=self.settings.request_timeout_seconds,
        )

    def _validate_allowed_tools(self, allowed: list[str] | None) -> None:
        if not allowed:
            return
        allowed_set = set(allowed)
        unknown = sorted(name for name in allowed_set if self._registry.get(name) is None)
        if unknown:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"allowed_tools contains unknown tool(s): {', '.join(unknown)}",
            )

    def _scoped_registry(self, allowed: list[str] | None) -> ToolRegistry:
        self._validate_allowed_tools(allowed)
        if not allowed:
            return self._registry
        allowed_set = set(allowed)
        allowed_set.add("finish")  # finish is always reachable
        tools: list[Tool] = []
        for name in self._registry.names():
            if name in allowed_set:
                tool = self._registry.get(name)
                if tool is not None:
                    tools.append(tool)
        if not tools:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="allowed_tools did not match any registered tool.",
            )
        return ToolRegistry(tools)

    async def _persist_run(self, task: AgentTask, run: AgentRun) -> None:
        for index, step in enumerate(run.steps):
            await self.repo.add_step(
                task,
                step_index=index,
                kind=self._step_kind(step),
                tool_name=step.tool_name,
                content=step.content,
                payload=step.payload,
                ok=step.ok,
            )
        await self._finalize_run_status(task, run)

    async def _finalize_run_status(self, task: AgentTask, run: AgentRun) -> None:
        await self.repo.bump_steps(task, run.turns_used)
        if run.errored:
            await self.repo.set_status(task, TaskStatus.failed, error=run.error, model=run.model)
        elif run.completed:
            await self.repo.set_status(
                task,
                TaskStatus.completed,
                result_summary=run.final_summary,
                model=run.model,
            )
        elif run.step_limit_hit:
            await self.repo.set_status(
                task,
                TaskStatus.failed,
                error=f"step limit ({task.max_steps}) reached before task completion",
                model=run.model,
            )
        else:
            await self.repo.set_status(
                task, TaskStatus.failed, error="run ended without a result", model=run.model
            )

    async def _ingest_documents(self, task_id: str, documents: list[TaskDocumentUpload]) -> int:
        if self.embeddings is None or self.vectorstore is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Document ingestion is unavailable.",
            )
        rag = RagService(self.session, self.settings, self.embeddings, self.vectorstore)
        count = 0
        for document in documents:
            await rag.ingest_task_document(
                task_id,
                filename=document.filename,
                content_type=document.content_type,
                data=document.data,
                commit=True,
            )
            count += 1
        return count

    async def _task_for_user(self, task_id: str, user_id: str) -> AgentTask:
        task = await self.repo.get(task_id)
        if task is None or task.user_id != user_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
        return task

    async def _thread_for_user(self, thread_id: str, user_id: str) -> AgentThread:
        thread = await self.repo.get_thread(thread_id)
        if thread is None or thread.user_id != user_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found")
        return thread

    async def _build_thread_context(self, task: AgentTask) -> str | None:
        if task.thread_id is None:
            return None
        prior_tasks = await self.repo.list_completed_tasks_in_thread(
            task.thread_id,
            before_sequence=task.sequence_index,
        )
        if not prior_tasks:
            return None
        return "\n".join(
            f"[{prior.sequence_index + 1}] Goal: {prior.goal}\n"
            f"Summary: {prior.result_summary or 'Completed without a summary.'}"
            for prior in prior_tasks
        )

    async def _task_document_count(self, task_id: str) -> int:
        return len(await DocumentRepository(self.session).list_by_task(task_id))

    def _step_kind(self, step: StepRecord) -> TaskStepKind:
        return TaskStepKind(step.kind)

    async def _detail(self, task_id: str, user_id: str | None = None) -> TaskDetail:
        task = await self.repo.get_with_steps(task_id)
        if task is None or (user_id is not None and task.user_id != user_id):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
        return TaskDetail(
            **TaskResponse.model_validate(task).model_dump(),
            steps=[TaskStepResponse.model_validate(s) for s in task.steps],
        )

    async def _thread_detail(self, thread_id: str, user_id: str) -> ThreadDetail:
        thread = await self.repo.get_thread_with_tasks(thread_id)
        if thread is None or thread.user_id != user_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found")
        return ThreadDetail.model_validate(thread)
