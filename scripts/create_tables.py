import asyncio

from sqlalchemy import text

from app.core.config import get_settings
from app.db.database import Base, engine
from app.models.chunk import (
    HNSW_EF_CONSTRUCTION,
    HNSW_INDEX_NAME,
    HNSW_M,
    Chunk,  # noqa: F401
)
from app.models.document import Document  # noqa: F401

settings = get_settings()


async def create_tables() -> None:
    async with engine.begin() as connection:
        await connection.execute(
            text("CREATE EXTENSION IF NOT EXISTS vector")
        )
        await connection.run_sync(Base.metadata.create_all)
        await connection.execute(
            text(
                "ALTER TABLE chunks "
                f"ADD COLUMN IF NOT EXISTS embedding "
                f"vector({settings.embedding_dimension})"
            )
        )
        await connection.execute(
            text(
                f"CREATE INDEX IF NOT EXISTS {HNSW_INDEX_NAME} "
                "ON chunks USING hnsw (embedding vector_cosine_ops) "
                f"WITH (m = {HNSW_M}, "
                f"ef_construction = {HNSW_EF_CONSTRUCTION})"
            )
        )

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(create_tables())
