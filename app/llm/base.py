"""Common interface for text-generation providers."""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None


class LLMProviderError(RuntimeError):
    """Raised when an LLM provider cannot generate an answer."""


class LLMProvider(Protocol):
    """Interface implemented by Ollama, Bedrock, vLLM, and other providers."""

    async def generate(self, prompt: str) -> LLMResponse:
        """Generate an answer from a prompt."""
        ...
