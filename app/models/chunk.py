"""Document chunk persistence model."""

import uuid

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import ForeignKey, Index, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.config import get_settings
from app.db.database import Base

settings = get_settings()

HNSW_INDEX_NAME = "ix_chunks_embedding_hnsw_cosine"
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 64


class Chunk(Base):
    """A searchable text fragment belonging to one document."""

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "chunk_index",
            name="uq_chunks_document_index",
        ),
        Index(
            HNSW_INDEX_NAME,
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={
                "m": HNSW_M,
                "ef_construction": HNSW_EF_CONSTRUCTION,
            },
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    page_number: Mapped[int] = mapped_column(nullable=False)

    chunk_index: Mapped[int] = mapped_column(nullable=False)

    text: Mapped[str] = mapped_column(Text, nullable=False)

    embedding: Mapped[list[float] | None] = mapped_column(
        VECTOR(settings.embedding_dimension),
        nullable=True,
    )
