"""Ollama text-generation provider."""

from dataclasses import dataclass
from typing import Any

import httpx

from app.llm.base import LLMProviderError, LLMResponse


@dataclass
class OllamaLLMProvider:
    """Generate text using an Ollama chat model."""

    base_url: str
    model: str
    timeout: float = 120.0

    async def generate(self, prompt: str) -> LLMResponse:
        """Send a non-streaming chat request to Ollama."""

        normalized_prompt = prompt.strip()

        if not normalized_prompt:
            raise ValueError("prompt must not be empty")

        async with httpx.AsyncClient(
            base_url=self.base_url.rstrip("/"),
            timeout=self.timeout,
        ) as client:
            response = await client.post(
                "/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": normalized_prompt,
                        }
                    ],
                    "stream": False,
                },
            )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    raise LLMProviderError(
                        f"Ollama model '{self.model}' was not found. "
                        f"Run `ollama pull {self.model}` first."
                    ) from exc

                raise LLMProviderError(
                    "Ollama returned an error while generating the answer"
                ) from exc
            except httpx.HTTPError as exc:
                raise LLMProviderError("Could not connect to Ollama") from exc

            payload: Any = response.json()

        if not isinstance(payload, dict):
            raise TypeError("Ollama returned an invalid response")

        message = payload.get("message")

        if not isinstance(message, dict):
            raise TypeError("Ollama response does not contain a message")

        content = message.get("content")

        if not isinstance(content, str):
            raise TypeError("Ollama message does not contain text content")

        return LLMResponse(
            text=content.strip(),
            model=self.model,
        )
