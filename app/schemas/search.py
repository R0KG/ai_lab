"""Vector search request and response schemas."""

from uuid import UUID

from pydantic import BaseModel


class SearchResult(BaseModel):
    """One chunk returned by semantic search."""

    chunk_id: UUID
    document_id: UUID
    page_number: int
    chunk_index: int
    text: str
    distance: float


class SearchResponse(BaseModel):
    """Semantic search response."""

    query: str
    results: list[SearchResult]
