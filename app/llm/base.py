"""Common interface for text-generation providers."""

from typing import Protocol


class LLMProviderError(RuntimeError):
    """Raised when an LLM provider cannot generate an answer."""


class LLMProvider(Protocol):
    """Interface implemented by Ollama, Bedrock, vLLM, and other providers."""

    async def generate(self, prompt: str) -> str:
        """Generate an answer from a prompt."""
        ...
