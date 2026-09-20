"""Tests for the Ollama text-generation provider."""

from typing import Any

import httpx
import pytest

from app.llm.ollama import OllamaLLMProvider


class FakeResponse:
    """Minimal HTTP response used by the provider test."""

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {
            "message": {
                "role": "assistant",
                "content": "  grounded answer  ",
            }
        }


class FakeAsyncClient:
    """Capture the request without connecting to Ollama."""

    last_request: dict[str, Any] | None = None

    def __init__(self, **_kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> "FakeAsyncClient":
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    async def post(self, path: str, json: dict[str, Any]) -> FakeResponse:
        self.last_request = {"path": path, "json": json}
        FakeAsyncClient.last_request = self.last_request
        return FakeResponse()


@pytest.mark.asyncio
async def test_ollama_provider_sends_chat_request(monkeypatch) -> None:
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    provider = OllamaLLMProvider(
        base_url="http://localhost:11434",
        model="qwen2.5:7b",
    )

    response = await provider.generate("What is Python?")

    assert response.text == "grounded answer"
    assert response.model == "qwen2.5:7b"
    assert FakeAsyncClient.last_request == {
        "path": "/api/chat",
        "json": {
            "model": "qwen2.5:7b",
            "messages": [
                {"role": "user", "content": "What is Python?"},
            ],
            "stream": False,
        },
    }


@pytest.mark.asyncio
async def test_ollama_provider_rejects_empty_prompt() -> None:
    provider = OllamaLLMProvider(
        base_url="http://localhost:11434",
        model="qwen2.5:7b",
    )

    with pytest.raises(ValueError, match="prompt must not be empty"):
        await provider.generate("   ")
