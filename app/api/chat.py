"""RAG chat endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.db.database import get_db
from app.embeddings.local import OllamaEmbeddingProvider
from app.llm.base import LLMProvider, LLMProviderError
from app.llm.bedrock import BedrockLLMProvider
from app.llm.ollama import OllamaLLMProvider
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import Retriever
from app.repositories.chunk_repository import ChunkRepository
from app.schemas.chat import ChatRequest, ChatResponse, ChatSource

router = APIRouter(
    prefix="/v1/chat",
    tags=["chat"],
)


def build_llm_provider(settings: Settings) -> LLMProvider:
    """Build the configured text-generation provider."""

    if settings.llm_backend == "ollama":
        return OllamaLLMProvider(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
        )
    if settings.llm_backend == "bedrock":
        return BedrockLLMProvider(
            region=settings.aws_region,
            model_id=settings.aws_bedrock_model_id,
            profile=settings.aws_profile,
        )
    raise ValueError(f"Unsupported LLM backend: {settings.llm_backend}")


def get_rag_pipeline(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RAGPipeline:
    """Build a RAG pipeline using the configured providers."""

    settings = get_settings()

    embedding_provider = OllamaEmbeddingProvider(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embedding_model,
    )
    llm_provider = build_llm_provider(settings)
    retriever = Retriever(
        chunk_repository=ChunkRepository(db),
        embedding_provider=embedding_provider,
    )

    return RAGPipeline(
        retriever=retriever,
        llm_provider=llm_provider,
    )


@router.post("", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    pipeline: Annotated[RAGPipeline, Depends(get_rag_pipeline)],
) -> ChatResponse:
    """Answer a question using retrieved document context."""

    settings = get_settings()
    effective_limit = payload.limit if payload.limit is not None else settings.top_k

    try:
        result = await pipeline.answer(
            query=payload.query,
            limit=effective_limit,
            document_id=payload.document_id,
        )
    except LLMProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ChatResponse(
        query=payload.query,
        answer=result.answer,
        sources=[
            ChatSource(
                chunk_id=item.chunk.id,
                document_id=item.chunk.document_id,
                page_number=item.chunk.page_number,
                chunk_index=item.chunk.chunk_index,
                distance=item.distance,
            )
            for item in result.sources
        ],
    )
