from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.embeddings.base import EmbeddingVector
from app.models.chunk import Chunk
from app.rag.chunker import TextChunk


@dataclass
class ChunkRepository:
    session: AsyncSession

    async def create_many(
        self,
        document_id: UUID,
        text_chunks: list[TextChunk],
        embeddings: list[EmbeddingVector],
    ) -> list[Chunk]:
        if len(text_chunks) != len(embeddings):
            raise ValueError(
                "The number of text chunks must match the number of embeddings"
            )

        if not text_chunks:
            return []

        rows = [
            Chunk(
                document_id=document_id,
                page_number=text_chunk.page_number,
                chunk_index=text_chunk.chunk_index,
                text=text_chunk.text,
                embedding=embedding,
            )
            for text_chunk, embedding in zip(
                text_chunks,
                embeddings,
                strict=True,
            )
        ]

        self.session.add_all(rows)

        try:
            await self.session.commit()
        except IntegrityError:
            await self.session.rollback()
            raise

        return rows

    async def search_similar(
        self,
        query_embedding: EmbeddingVector,
        limit: int,
        document_id: UUID | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Return chunks ordered by cosine distance to a query vector."""

        if not query_embedding:
            raise ValueError("query_embedding must not be empty")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")

        distance = Chunk.embedding.cosine_distance(query_embedding)
        statement = (
            select(Chunk, distance.label("distance"))
            .where(Chunk.embedding.is_not(None))
            .order_by(distance, Chunk.id)
            .limit(limit)
        )

        if document_id is not None:
            statement = statement.where(Chunk.document_id == document_id)

        result = await self.session.execute(statement)

        matches: list[tuple[Chunk, float]] = []
        for chunk, raw_distance in result.all():
            if raw_distance is not None:
                matches.append((chunk, float(raw_distance)))

        return matches
