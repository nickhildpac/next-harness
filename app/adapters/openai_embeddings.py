import httpx
from openai import AsyncOpenAI

from app.core.config import Settings
from app.ports.embeddings import EmbeddingsClient


class EmbeddingsUnavailable(RuntimeError):
    pass


class OpenAIEmbeddingsClient(EmbeddingsClient):
    def __init__(self, settings: Settings, http_client: httpx.AsyncClient | None = None):
        self.settings = settings
        self._http = http_client
        self._client_instance: AsyncOpenAI | None = None

    def _client(self) -> AsyncOpenAI:
        if self._client_instance is None:
            if not self.settings.openai_api_key:
                raise EmbeddingsUnavailable("OpenAI API key is not configured")
            self._client_instance = AsyncOpenAI(
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_base_url or None,
            )
        return self._client_instance

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        kwargs: dict = {"model": self.settings.embedding_model, "input": texts}
        if self.settings.embedding_dimensions:
            kwargs["dimensions"] = self.settings.embedding_dimensions
        response = await self._client().embeddings.create(**kwargs)
        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]

    async def health(self) -> bool:
        return bool(self.settings.openai_api_key)
