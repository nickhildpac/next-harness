import json

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import Settings
from app.db.base import Base
from app.mcp.context import McpRuntime
from app.mcp.identity import USER_SCOPED_TOOLS, TASK_SCOPED_TOOLS, augment_schema
from app.mcp.server import handle_call_tool, mcp_tools
from app.repositories.notes import NoteRepository
from app.tools.registry import ToolRegistry, build_default_registry
from tests.conftest import FakeEmbeddings, FakeLLM, FakeVectorStore


@pytest.fixture
def settings() -> Settings:
    return Settings(
        database_url="sqlite+aiosqlite:///:memory:",
        task_llm_provider="ollama",
        mcp_user_id=None,
        mcp_task_id=None,
    )


@pytest.fixture
async def session_factory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    yield maker
    await engine.dispose()


@pytest.fixture
def registry() -> ToolRegistry:
    return build_default_registry()


@pytest.fixture
def runtime(settings, session_factory) -> McpRuntime:
    return McpRuntime(
        settings,
        session_factory=session_factory,
        llm=FakeLLM(),
        embeddings=FakeEmbeddings(),
        vectorstore=FakeVectorStore(),
        http_client=None,
    )


async def test_mcp_tools_mirror_registry_minus_finish(registry):
    advertised = mcp_tools(registry)
    names = [tool.name for tool in advertised]
    expected = [name for name in registry.names() if name != "finish"]
    assert names == expected
    assert "finish" not in names


async def test_mcp_schemas_match_spec_with_additive_identity(registry):
    by_name = {tool.name: tool for tool in mcp_tools(registry)}
    for tool in registry.tools():
        if tool.name == "finish":
            continue
        spec = tool.spec()
        advertised = by_name[tool.name]
        assert advertised.description == spec.description
        schema = advertised.inputSchema
        base_props = (spec.parameters or {}).get("properties") or {}
        for key, value in base_props.items():
            assert schema["properties"][key] == value
        if tool.name in USER_SCOPED_TOOLS:
            assert "user_id" in schema["properties"]
        else:
            assert "user_id" not in schema.get("properties", {})
        if tool.name in TASK_SCOPED_TOOLS:
            assert "task_id" in schema["properties"]
        else:
            assert "task_id" not in schema.get("properties", {})


def test_augment_schema_does_not_mutate_original():
    original = {"type": "object", "properties": {"limit": {"type": "integer"}}}
    copy = dict(original)
    augmented = augment_schema("list_notes", original)
    assert original == copy
    assert "user_id" in augmented["properties"]
    assert "user_id" not in original["properties"]


async def test_now_works_without_identity(registry, settings, runtime):
    result = await handle_call_tool(
        "now",
        {},
        registry=registry,
        settings=settings,
        runtime=runtime,
    )
    assert result.isError is False
    payload = json.loads(result.content[0].text)
    assert "iso" in payload
    assert "epoch" in payload


async def test_list_notes_requires_identity(registry, settings, runtime):
    result = await handle_call_tool(
        "list_notes",
        {},
        registry=registry,
        settings=settings,
        runtime=runtime,
    )
    assert result.isError is True
    assert "user_id" in result.content[0].text
    assert "MCP_USER_ID" in result.content[0].text


async def test_list_notes_succeeds_with_arg_user_id(registry, settings, runtime, session_factory):
    async with session_factory() as session:
        await NoteRepository(session).create(
            user_id="alice",
            title="Hello",
            content="world",
            style_name="default",
            custom_instructions=None,
        )
        await session.commit()

    result = await handle_call_tool(
        "list_notes",
        {"user_id": "alice", "limit": 10},
        registry=registry,
        settings=settings,
        runtime=runtime,
    )
    assert result.isError is False
    payload = json.loads(result.content[0].text)
    assert payload["count"] == 1
    assert payload["items"][0]["title"] == "Hello"


async def test_list_notes_succeeds_with_settings_user_id(registry, session_factory, settings):
    settings = settings.model_copy(update={"mcp_user_id": "bob"})
    runtime = McpRuntime(
        settings,
        session_factory=session_factory,
        llm=FakeLLM(),
        embeddings=FakeEmbeddings(),
        vectorstore=FakeVectorStore(),
    )
    async with session_factory() as session:
        await NoteRepository(session).create(
            user_id="bob",
            title="Bob note",
            content="x",
            style_name="default",
            custom_instructions=None,
        )
        await session.commit()

    result = await handle_call_tool(
        "list_notes",
        {},
        registry=registry,
        settings=settings,
        runtime=runtime,
    )
    assert result.isError is False
    payload = json.loads(result.content[0].text)
    assert payload["count"] == 1
    assert payload["items"][0]["title"] == "Bob note"


async def test_finish_is_rejected(registry, settings, runtime):
    result = await handle_call_tool(
        "finish",
        {"summary": "done"},
        registry=registry,
        settings=settings,
        runtime=runtime,
    )
    assert result.isError is True
    assert "not available via MCP" in result.content[0].text


async def test_unknown_tool_is_rejected(registry, settings, runtime):
    result = await handle_call_tool(
        "nope",
        {},
        registry=registry,
        settings=settings,
        runtime=runtime,
    )
    assert result.isError is True
    assert "unknown tool" in result.content[0].text


async def test_task_scoped_tool_requires_task_id(registry, settings, runtime):
    result = await handle_call_tool(
        "list_task_documents",
        {},
        registry=registry,
        settings=settings,
        runtime=runtime,
    )
    assert result.isError is True
    assert "task_id" in result.content[0].text
    assert "MCP_TASK_ID" in result.content[0].text
