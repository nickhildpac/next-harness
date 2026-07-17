import os
from collections.abc import AsyncIterator

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import get_settings
from app.db.base import Base

settings = get_settings()
_is_sqlite = settings.database_url.startswith("sqlite")
connect_args = (
    {"check_same_thread": False, "timeout": 30} if _is_sqlite else {}
)
engine_kwargs: dict = {
    "echo": False,
    "pool_pre_ping": True,
    "connect_args": connect_args,
}
if _is_sqlite:
    # File-backed SQLite with async needs one connection per checkout so MCP tool
    # calls and long-lived streaming requests do not share a locked connection.
    engine_kwargs["poolclass"] = NullPool
engine = create_async_engine(settings.database_url, **engine_kwargs)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


@event.listens_for(engine.sync_engine, "connect")
def _configure_sqlite_connection(dbapi_connection, _connection_record) -> None:
    if not _is_sqlite:
        return
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=30000")
    cursor.close()


async def create_db_and_tables() -> None:
    if settings.database_url.startswith("sqlite"):
        db_path = settings.database_url.removeprefix("sqlite+aiosqlite:///")
        if db_path and db_path != ":memory:":
            os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session

