from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import conversations, health
from app.core.config import get_settings
from app.core.logging import RequestContextMiddleware, configure_logging
from app.db.session import create_db_and_tables


@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_db_and_tables()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    app.add_middleware(RequestContextMiddleware)
    app.include_router(health.router)
    app.include_router(conversations.router)
    return app


app = create_app()

