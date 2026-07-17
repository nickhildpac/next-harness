from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.api.routes import (
    auth,
    conversations,
    documents,
    health,
    notes,
    providers,
    tasks,
    tones,
    translations,
)
from app.core.config import get_settings
from app.core.logging import RequestContextMiddleware, configure_logging
from app.db.session import create_db_and_tables
from app.mcp.context import McpRuntime
from app.mcp.http import McpBearerAuthMiddleware, StreamableHttpASGIApp, create_session_manager
from app.mcp.server import create_server


mcp_asgi = StreamableHttpASGIApp()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_db_and_tables()
    settings = get_settings()
    app.state.http_client = httpx.AsyncClient()
    async with McpRuntime(settings, http_client=app.state.http_client) as runtime:
        server = create_server(settings=settings, runtime=runtime)
        manager = create_session_manager(server)
        mcp_asgi.manager = manager
        app.state.mcp_session_manager = manager
        # In-process transport so TaskService can call /mcp without a network hop.
        app.state.mcp_asgi_transport = httpx.ASGITransport(app=app, raise_app_exceptions=True)
        async with manager.run():
            try:
                yield
            finally:
                mcp_asgi.manager = None
                app.state.mcp_session_manager = None
                app.state.mcp_asgi_transport = None
        await app.state.http_client.aclose()


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    app.add_middleware(RequestContextMiddleware)
    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(tasks.router)
    app.include_router(tones.router)
    app.include_router(providers.router)
    app.include_router(conversations.router)
    app.include_router(documents.router)
    app.include_router(notes.router)
    app.include_router(translations.router)
    app.mount(
        "/mcp",
        McpBearerAuthMiddleware(mcp_asgi, settings_getter=get_settings),
        name="mcp",
    )

    @app.get("/", include_in_schema=False)
    async def index_redirect() -> RedirectResponse:
        return RedirectResponse(url="/app/tasks.html")

    return app


app = create_app()
