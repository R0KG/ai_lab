"""Tests for the document ingestion service."""

from pathlib import Path
from uuid import UUID, uuid4

import pymupdf
import pytest

from app.models.chunk import Chunk
from app.rag.chunker import TextChunk
from app.services.ingestion_service import IngestionService


class FakeEmbeddingProvider:
    """Return deterministic vectors without calling Ollama."""

    def __init__(self, dimension: int = 3) -> None:
        self.dimension = dimension
        self.texts: list[str] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.texts = texts
        return [
            [float(index)] * self.dimension
            for index, _text in enumerate(texts)
        ]


class FakeChunkRepository:
    """Capture repository input without requiring a database."""

    def __init__(self) -> None:
        self.document_id: UUID | None = None
        self.text_chunks: list[TextChunk] = []
        self.embeddings: list[list[float]] = []

    async def create_many(
        self,
        document_id: UUID,
        text_chunks: list[TextChunk],
        embeddings: list[list[float]],
    ) -> list[Chunk]:
        self.document_id = document_id
        self.text_chunks = text_chunks
        self.embeddings = embeddings

        return [
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


def create_test_pdf(path: Path) -> None:
    """Create a small text PDF for deterministic extraction tests."""

    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text(
            (72, 72),
            "one two three four five six seven eight nine ten",
        )
        document.save(path)


async def test_ingest_pdf_extracts_chunks_and_calls_repository(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "sample.pdf"
    create_test_pdf(pdf_path)
    document_id = uuid4()
    repository = FakeChunkRepository()
    embedding_provider = FakeEmbeddingProvider()
    service = IngestionService(
        chunk_repository=repository,  # type: ignore[arg-type]
        embedding_provider=embedding_provider,  # type: ignore[arg-type]
        chunk_size=4,
        chunk_overlap=1,
        embedding_dimension=3,
    )

    saved_chunks = await service.ingest_pdf(document_id, pdf_path)

    assert repository.document_id == document_id
    assert [chunk.text for chunk in repository.text_chunks] == [
        "one two three four",
        "four five six seven",
        "seven eight nine ten",
    ]
    assert embedding_provider.texts == [chunk.text for chunk in repository.text_chunks]
    assert [len(embedding) for embedding in repository.embeddings] == [3, 3, 3]
    assert len(saved_chunks) == 3


async def test_ingest_pdf_rejects_missing_file(tmp_path: Path) -> None:
    repository = FakeChunkRepository()
    embedding_provider = FakeEmbeddingProvider()
    service = IngestionService(
        chunk_repository=repository,  # type: ignore[arg-type]
        embedding_provider=embedding_provider,  # type: ignore[arg-type]
    )

    with pytest.raises(FileNotFoundError):
        await service.ingest_pdf(uuid4(), tmp_path / "missing.pdf")
