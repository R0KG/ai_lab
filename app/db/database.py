from collections.abc import AsyncIterator
from typing import Any

from pgvector.psycopg import register_vector_async
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings

settings = get_settings()


class Base(DeclarativeBase):
    pass


engine: AsyncEngine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    pool_pre_ping=True,
)


@event.listens_for(engine.sync_engine, "connect")
def register_vector(
    dbapi_connection: Any,
    _connection_record: Any,
) -> None:
    """Register pgvector types for async psycopg connections."""

    dbapi_connection.run_async(register_vector_async)


SessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session


async def check_database() -> bool:
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
    except Exception:
        return False
    return True


async def close_database() -> None:
    await engine.dispose()
