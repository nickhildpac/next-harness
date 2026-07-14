"""Shared MCP runtime services and per-call ToolContext construction."""

from __future__ import annotations

from typing import Any

import httpx
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.adapters.chroma import ChromaVectorStore
from app.adapters.openai_embeddings import OpenAIEmbeddingsClient
from app.api.dependencies import build_llm_client
from app.core.config import Settings, get_settings
from app.db.session import SessionLocal
from app.ports.embeddings import EmbeddingsClient
from app.ports.llm import LLMClient
from app.ports.vectorstore import VectorStore
from app.tools.registry import ToolContext, ToolRegistry, ToolResult


class McpRuntime:
    """Process-scoped clients reused across MCP tool calls."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        session_factory: async_sessionmaker[AsyncSession] | None = None,
        llm: LLMClient | None = None,
        embeddings: EmbeddingsClient | None = None,
        vectorstore: VectorStore | None = None,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.settings = settings or get_settings()
        self.session_factory = session_factory or SessionLocal
        self._owns_http = http_client is None
        self.http_client = http_client
        self._llm = llm
        self._llm_failed = False
        self._embeddings = embeddings
        self._vectorstore = vectorstore

    async def __aenter__(self) -> McpRuntime:
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(timeout=self.settings.request_timeout_seconds)
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._owns_http and self.http_client is not None:
            await self.http_client.aclose()
            self.http_client = None

    def resolve_llm(self) -> LLMClient | None:
        if self._llm is not None or self._llm_failed:
            return self._llm
        try:
            self._llm = build_llm_client(
                self.settings,
                self.settings.task_llm_provider,
                http_client=self.http_client,
                openai_model_override=self.settings.task_openai_model,
            )
        except Exception:  # noqa: BLE001 — optional until a tool needs the LLM
            self._llm_failed = True
            return None
        return self._llm

    def resolve_embeddings(self) -> EmbeddingsClient:
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddingsClient(self.settings, self.http_client)
        return self._embeddings

    def resolve_vectorstore(self) -> VectorStore:
        if self._vectorstore is None:
            self._vectorstore = ChromaVectorStore(self.settings)
        return self._vectorstore

    def build_context(
        self,
        session: AsyncSession,
        *,
        user_id: str | None,
        task_id: str | None,
    ) -> ToolContext:
        return ToolContext(
            session=session,
            http_client=self.http_client,
            user_id=user_id,
            task_id=task_id,
            settings=self.settings,
            llm=self.resolve_llm(),
            embeddings=self.resolve_embeddings(),
            vectorstore=self.resolve_vectorstore(),
        )

    async def invoke(
        self,
        registry: ToolRegistry,
        name: str,
        arguments: dict[str, Any],
        *,
        user_id: str | None,
        task_id: str | None,
    ) -> ToolResult:
        async with self.session_factory() as session:
            context = self.build_context(session, user_id=user_id, task_id=task_id)
            result = await registry.invoke(name, arguments, context)
            if result.ok:
                await session.commit()
            else:
                await session.rollback()
            return result
