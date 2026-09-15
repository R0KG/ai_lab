"""Semantic retrieval over indexed document chunks."""

from dataclasses import dataclass
from uuid import UUID

from app.embeddings.base import EmbeddingProvider
from app.models.chunk import Chunk
from app.repositories.chunk_repository import ChunkRepository

DEFAULT_SEARCH_LIMIT = 5
MAX_SEARCH_LIMIT = 50


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by vector search with its cosine distance."""

    chunk: Chunk
    distance: float


@dataclass
class Retriever:
    """Embed a query and retrieve the nearest stored chunks."""

    chunk_repository: ChunkRepository
    embedding_provider: EmbeddingProvider

    async def search(
        self,
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
        document_id: UUID | None = None,
    ) -> list[RetrievedChunk]:
        """Return chunks nearest to a natural-language query."""

        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")
        if not 1 <= limit <= MAX_SEARCH_LIMIT:
            raise ValueError(
                f"limit must be between 1 and {MAX_SEARCH_LIMIT}"
            )

        embeddings = await self.embedding_provider.embed([normalized_query])
        if len(embeddings) != 1:
            raise ValueError(
                "Embedding provider must return one vector for the query"
            )

        matches = await self.chunk_repository.search_similar(
            embeddings[0],
            limit,
            document_id,
        )

        return [
            RetrievedChunk(chunk=chunk, distance=distance)
            for chunk, distance in matches
        ]
