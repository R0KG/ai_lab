"""Common embedding provider interface."""

from typing import Protocol

type EmbeddingVector = list[float]


class EmbeddingProvider(Protocol):
    async def embed(
        self,
        texts: list[str],
    ) -> list[EmbeddingVector]: ...
