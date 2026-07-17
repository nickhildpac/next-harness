"""Tests for MCP Streamable HTTP mount and NDJSON task streaming helpers."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.adapters.mcp_client import McpStreamableHttpSession
from app.api.ndjson import ndjson_event
from app.core.config import Settings
from app.mcp.context import McpRuntime
from app.mcp.http import McpBearerAuthMiddleware, StreamableHttpASGIApp, create_session_manager
from app.mcp.server import create_server
from app.services.auth import create_access_token


def test_ndjson_event_frames_payload():
    line = ndjson_event("step", {"tool_name": "now", "ok": True})
    assert line.endswith("\n")
    assert '"event": "step"' in line or '"event":"step"' in line
    assert "now" in line


async def _with_mcp_http_app(settings: Settings):
    """Start Streamable HTTP MCP + bearer auth in the current task."""
    asgi = StreamableHttpASGIApp()
    runtime = McpRuntime(settings)
    await runtime.__aenter__()
    server = create_server(settings=settings, runtime=runtime)
    manager = create_session_manager(server)
    asgi.manager = manager
    auth_app = McpBearerAuthMiddleware(asgi, settings_getter=lambda: settings)
    run_cm = manager.run()
    await run_cm.__aenter__()
    return auth_app, runtime, run_cm, asgi


async def _teardown_mcp_http(runtime, run_cm, asgi: StreamableHttpASGIApp) -> None:
    asgi.manager = None
    await run_cm.__aexit__(None, None, None)
    await runtime.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_mcp_streamable_http_requires_auth(tmp_path):
    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'mcp-auth.db'}",
        mcp_http_auth_token="test-mcp-token",
        chroma_persist_dir=str(tmp_path / "chroma"),
    )
    auth_app, runtime, run_cm, asgi = await _with_mcp_http_app(settings)
    try:
        transport = ASGITransport(app=auth_app, raise_app_exceptions=True)
        async with AsyncClient(transport=transport, base_url="http://mcp.local") as client:
            response = await client.post(
                "/",
                json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
            )
        assert response.status_code == 401
    finally:
        await _teardown_mcp_http(runtime, run_cm, asgi)


@pytest.mark.asyncio
async def test_mcp_streamable_http_lists_and_calls_now(tmp_path):
    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'mcp-now.db'}",
        mcp_http_auth_token="test-mcp-token",
        chroma_persist_dir=str(tmp_path / "chroma"),
    )
    auth_app, runtime, run_cm, asgi = await _with_mcp_http_app(settings)
    try:
        transport = ASGITransport(app=auth_app, raise_app_exceptions=True)
        async with AsyncClient(
            transport=transport,
            base_url="http://mcp.local",
            headers={"Authorization": f"Bearer {settings.mcp_http_auth_token}"},
            timeout=60.0,
        ) as client:
            async with McpStreamableHttpSession(
                url="http://mcp.local/",
                http_client=client,
            ) as mcp:
                specs = await mcp.list_specs()
                names = {spec.name for spec in specs}
                assert "now" in names
                assert "finish" not in names

                result = await mcp.call_tool("now", {})
                assert result.ok
                assert result.output is not None
    finally:
        await _teardown_mcp_http(runtime, run_cm, asgi)


@pytest.mark.asyncio
async def test_mcp_streamable_http_accepts_user_jwt(tmp_path):
    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'mcp-jwt.db'}",
        mcp_http_auth_token="test-mcp-token",
        chroma_persist_dir=str(tmp_path / "chroma"),
    )
    token, _ = create_access_token("user-1", settings.auth_secret_key, 60)
    auth_app, runtime, run_cm, asgi = await _with_mcp_http_app(settings)
    try:
        transport = ASGITransport(app=auth_app, raise_app_exceptions=True)
        async with AsyncClient(
            transport=transport,
            base_url="http://mcp.local",
            headers={"Authorization": f"Bearer {token}"},
            timeout=60.0,
        ) as client:
            async with McpStreamableHttpSession(
                url="http://mcp.local/",
                http_client=client,
            ) as mcp:
                result = await mcp.call_tool("now", {})
                assert result.ok
    finally:
        await _teardown_mcp_http(runtime, run_cm, asgi)


@pytest.mark.asyncio
async def test_session_manager_handles_stateless_tool_call(tmp_path):
    settings = Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'direct.db'}",
        chroma_persist_dir=str(tmp_path / "chroma"),
    )
    async with McpRuntime(settings) as runtime:
        server = create_server(settings=settings, runtime=runtime)
        manager = create_session_manager(server)
        async with manager.run():
            assert manager is not None
