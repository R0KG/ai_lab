"""Tests for RAG orchestration."""

from uuid import uuid4

from app.llm.base import LLMResponse
from app.models.chunk import Chunk
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import RetrievedChunk


class FakeRetriever:
    """Return one deterministic document chunk."""

    def __init__(self) -> None:
        self.query: str | None = None
        self.limit: int | None = None

    async def search(
        self,
        query: str,
        limit: int,
        document_id=None,
    ) -> list[RetrievedChunk]:
        self.query = query
        self.limit = limit
        chunk = Chunk(
            id=uuid4(),
            document_id=document_id or uuid4(),
            page_number=2,
            chunk_index=0,
            text="Python is used for backend development.",
        )
        return [RetrievedChunk(chunk=chunk, distance=0.15)]


class FakeLLMProvider:
    """Capture the generated prompt and return a deterministic answer."""

    def __init__(self) -> None:
        self.prompt: str | None = None

    async def generate(self, prompt: str) -> LLMResponse:
        self.prompt = prompt
        return LLMResponse(
            text="Python is used for backend development.",
            model="fake-model",
        )


async def test_pipeline_retrieves_context_and_generates_answer() -> None:
    retriever = FakeRetriever()
    llm_provider = FakeLLMProvider()
    pipeline = RAGPipeline(
        retriever=retriever,  # type: ignore[arg-type]
        llm_provider=llm_provider,  # type: ignore[arg-type]
    )

    result = await pipeline.answer(
        query="What is Python used for?",
        limit=5,
    )

    assert retriever.query == "What is Python used for?"
    assert retriever.limit == 5
    assert result.answer == "Python is used for backend development."
    assert len(result.sources) == 1
    assert llm_provider.prompt is not None
    assert "Python is used for backend development." in llm_provider.prompt
