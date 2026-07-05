from typing import Protocol


class EmbeddingsClient(Protocol):
    """Seam between the app and any text-embedding provider."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed each text, returning vectors in the same order as the input."""
        ...

    async def health(self) -> bool: ...
