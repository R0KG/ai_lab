"""Local embedding model adapter."""

from dataclasses import dataclass
from typing import Any

import httpx

from app.embeddings.base import EmbeddingVector


@dataclass
class OllamaEmbeddingProvider:
    base_url: str
    model: str
    timeout: float = 60.0

    async def embed(
        self,
        texts: list[str],
    ) -> list[EmbeddingVector]:
        if not texts:
            return []

        async with httpx.AsyncClient(
            base_url=self.base_url.rstrip("/"),
            timeout=self.timeout,
        ) as client:
            response = await client.post(
                "/api/embed",
                json={
                    "model": self.model,
                    "input": texts,
                    "truncate": False,
                },
            )
            response.raise_for_status()
            payload: Any = response.json()

        if not isinstance(payload, dict):
            raise TypeError("Ollama returned an invalid response")

        raw_embeddings = payload.get("embeddings")

        if not isinstance(raw_embeddings, list):
            raise TypeError("Ollama response does not contain embeddings")

        if len(raw_embeddings) != len(texts):
            raise ValueError("Ollama returned a different number of embeddings")

        embeddings: list[EmbeddingVector] = []

        for raw_embedding in raw_embeddings:
            if not isinstance(raw_embedding, list):
                raise TypeError("Embeddings must be a list")

            if not all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in raw_embedding
            ):
                raise TypeError("Embedding must be numeric")
            embeddings.append([float(value) for value in raw_embedding])

        return embeddings
