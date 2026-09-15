"""Document ingestion orchestration."""

from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from app.embeddings.base import EmbeddingProvider
from app.models.chunk import Chunk
from app.rag.chunker import chunk_text
from app.rag.extractor import extract_pdf
from app.repositories.chunk_repository import ChunkRepository


@dataclass
class IngestionService:
    """Extract, chunk, and persist one document."""

    chunk_repository: ChunkRepository
    embedding_provider: EmbeddingProvider
    chunk_size: int = 600
    chunk_overlap: int = 100
    embedding_dimension: int = 768

    async def ingest_pdf(
        self,
        document_id: UUID,
        file_path: Path,
    ) -> list[Chunk]:
        """Extract a PDF, create chunks, and save them for a document."""

        if not file_path.is_file():
            raise FileNotFoundError(f"Document file not found: {file_path}")

        pages = extract_pdf(file_path)
        text_chunks = chunk_text(
            pages,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        embeddings = await self.embedding_provider.embed(
            [text_chunk.text for text_chunk in text_chunks]
        )

        if len(embeddings) != len(text_chunks):
            raise ValueError(
                "Embedding provider returned a different number of vectors"
            )

        invalid_dimensions = {
            len(embedding)
            for embedding in embeddings
            if len(embedding) != self.embedding_dimension
        }
        if invalid_dimensions:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"expected {self.embedding_dimension}, "
                f"received {sorted(invalid_dimensions)}"
            )

        return await self.chunk_repository.create_many(
            document_id,
            text_chunks,
            embeddings,
        )
