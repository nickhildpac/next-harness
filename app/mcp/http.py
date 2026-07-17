"""Streamable HTTP ASGI surface for the MCP tool server."""

from __future__ import annotations

from collections.abc import Callable

from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.types import ASGIApp, Receive, Scope, Send

from app.core.config import Settings
from app.services.auth import decode_access_token


def create_session_manager(
    server: Server,
    *,
    stateless: bool = True,
    json_response: bool = True,
) -> StreamableHTTPSessionManager:
    """Build a Streamable HTTP session manager for the low-level MCP Server."""
    return StreamableHTTPSessionManager(
        app=server,
        json_response=json_response,
        stateless=stateless,
        security_settings=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )


class StreamableHttpASGIApp:
    """ASGI app that delegates to a session manager set during FastAPI lifespan."""

    def __init__(self) -> None:
        self.manager: StreamableHTTPSessionManager | None = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self.manager is None:
            await _send_json_error(send, 503, "MCP Streamable HTTP is not ready")
            return
        await self.manager.handle_request(scope, receive, send)


class McpBearerAuthMiddleware:
    """Require Bearer JWT or the shared ``mcp_http_auth_token`` for ``/mcp``."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        settings_getter: Callable[[], Settings],
    ) -> None:
        self.app = app
        self._settings_getter = settings_getter

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        authorization = headers.get("authorization", "")
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token:
            await _send_json_error(send, 401, "Missing bearer token")
            return

        settings = self._settings_getter()
        if settings.mcp_http_auth_token and token == settings.mcp_http_auth_token:
            await self.app(scope, receive, send)
            return

        try:
            decode_access_token(token, settings.auth_secret_key)
        except Exception:  # noqa: BLE001 — map any auth failure to 401
            await _send_json_error(send, 401, "Invalid token")
            return

        await self.app(scope, receive, send)


async def _send_json_error(send: Send, status: int, detail: str) -> None:
    body = f'{{"detail":"{detail}"}}'.encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
