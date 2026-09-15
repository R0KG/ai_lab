"""Tests for semantic retrieval orchestration."""

from uuid import UUID, uuid4

import pytest

from app.embeddings.base import EmbeddingVector
from app.models.chunk import Chunk
from app.rag.retriever import Retriever


class FakeEmbeddingProvider:
    """Capture the query and return one deterministic vector."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    async def embed(self, texts: list[str]) -> list[EmbeddingVector]:
        self.texts = texts
        return [[0.1, 0.2, 0.3]]


class FakeChunkRepository:
    """Return one deterministic database match."""

    def __init__(self, document_id: UUID) -> None:
        self.document_id = document_id
        self.query_embedding: EmbeddingVector | None = None
        self.limit: int | None = None

    async def search_similar(
        self,
        query_embedding: EmbeddingVector,
        limit: int,
        document_id: UUID | None = None,
    ) -> list[tuple[Chunk, float]]:
        self.query_embedding = query_embedding
        self.limit = limit

        chunk = Chunk(
            document_id=document_id or self.document_id,
            page_number=1,
            chunk_index=0,
            text="Python backend experience",
            embedding=query_embedding,
        )
        return [(chunk, 0.12)]


async def test_retriever_embeds_query_and_returns_matches() -> None:
    document_id = uuid4()
    provider = FakeEmbeddingProvider()
    repository = FakeChunkRepository(document_id)
    retriever = Retriever(
        chunk_repository=repository,  # type: ignore[arg-type]
        embedding_provider=provider,  # type: ignore[arg-type]
    )

    matches = await retriever.search("  python backend  ", limit=3)

    assert provider.texts == ["python backend"]
    assert repository.query_embedding == [0.1, 0.2, 0.3]
    assert repository.limit == 3
    assert len(matches) == 1
    assert matches[0].chunk.text == "Python backend experience"
    assert matches[0].distance == 0.12


async def test_retriever_rejects_empty_query() -> None:
    retriever = Retriever(
        chunk_repository=FakeChunkRepository(uuid4()),  # type: ignore[arg-type]
        embedding_provider=FakeEmbeddingProvider(),  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="query must not be empty"):
        await retriever.search("   ")
