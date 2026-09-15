"""Chat request and response schemas."""

from uuid import UUID

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request for a RAG answer."""

    query: str = Field(min_length=1, max_length=2_000)
    limit: int | None = Field(default=None, ge=1, le=50)
    document_id: UUID | None = None


class ChatSource(BaseModel):
    """A document chunk used to generate the answer."""

    chunk_id: UUID
    document_id: UUID
    page_number: int
    chunk_index: int
    distance: float


class ChatResponse(BaseModel):
    """RAG answer with source references."""

    query: str
    answer: str
    sources: list[ChatSource]
