"""Semantic search endpoints."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.database import get_db
from app.embeddings.local import OllamaEmbeddingProvider
from app.rag.retriever import Retriever
from app.repositories.chunk_repository import ChunkRepository
from app.schemas.search import SearchResponse, SearchResult

router = APIRouter(
    prefix="/v1/search",
    tags=["search"],
)


def get_retriever(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Retriever:
    """Build the retriever with the configured embedding provider."""

    settings = get_settings()
    return Retriever(
        chunk_repository=ChunkRepository(db),
        embedding_provider=OllamaEmbeddingProvider(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embedding_model,
        ),
    )


def resolve_search_limit(requested_limit: int | None) -> int:
    """Use the configured top_k unless the request explicitly overrides it."""

    if requested_limit is not None:
        return requested_limit

    return get_settings().top_k


@router.get("", response_model=SearchResponse)
async def search_documents(
    retriever: Annotated[Retriever, Depends(get_retriever)],
    query: Annotated[
        str,
        Query(alias="q", min_length=1, max_length=2_000),
    ],
    limit: Annotated[int | None, Query(ge=1, le=50)] = None,
    document_id: Annotated[UUID | None, Query()] = None,
) -> SearchResponse:
    """Find document chunks semantically similar to a query."""

    effective_limit = resolve_search_limit(limit)
    matches = await retriever.search(
        query=query,
        limit=effective_limit,
        document_id=document_id,
    )

    return SearchResponse(
        query=query,
        results=[
            SearchResult(
                chunk_id=match.chunk.id,
                document_id=match.chunk.document_id,
                page_number=match.chunk.page_number,
                chunk_index=match.chunk.chunk_index,
                text=match.chunk.text,
                distance=match.distance,
            )
            for match in matches
        ],
    )
