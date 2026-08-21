from dataclasses import dataclass
from typing import Protocol

import httpx
import numpy as np

from backend.app.core.ollama import get_ollama_runtime_info


@dataclass(frozen=True)
class EmbeddingDescriptor:
    provider: str
    model: str
    digest: str
    dimension: int


class EmbeddingProvider(Protocol):
    async def prepare(self) -> EmbeddingDescriptor: ...

    async def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    async def embed_query(self, query: str) -> np.ndarray: ...

    async def close(self) -> None: ...


def validate_vectors(vectors, rows: int, dimension: int) -> np.ndarray:
    try:
        result = np.asarray(vectors, dtype="float32")
    except (TypeError, ValueError) as exc:
        raise ValueError("Embedding 响应结构无效") from exc
    if result.shape != (rows, dimension):
        raise ValueError("Embedding 返回数量或维度无效")
    if not np.isfinite(result).all():
        raise ValueError("Embedding 返回了非有限数值")
    return result


class OllamaEmbeddingProvider:
    def __init__(self, settings):
        self.settings = settings
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.client: httpx.AsyncClient | None = None
        self.descriptor: EmbeddingDescriptor | None = None

    async def _client(self) -> httpx.AsyncClient:
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=self.settings.ollama_request_timeout_seconds)
        return self.client

    async def _models(self) -> list[dict]:
        response = await (await self._client()).get(f"{self.base_url}/api/tags")
        response.raise_for_status()
        models = response.json().get("models")
        if not isinstance(models, list):
            raise TypeError("Ollama /api/tags 未返回模型列表")
        return models

    async def prepare(self) -> EmbeddingDescriptor:
        runtime = get_ollama_runtime_info()
        if runtime and runtime.available and runtime.model == self.settings.embedding_model:
            digest = runtime.digest
        else:
            digest = ""
            for item in await self._models():
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or item.get("model") or "")
                if name == self.settings.embedding_model:
                    digest = str(item.get("digest") or "")
                    break
        if not digest:
            raise RuntimeError(
                f"未找到 {self.settings.embedding_model}，请先执行："
                f"ollama pull {self.settings.embedding_model}"
            )
        self.descriptor = EmbeddingDescriptor(
            provider="ollama",
            model=self.settings.embedding_model,
            digest=digest,
            dimension=self.settings.embedding_dimension,
        )
        return self.descriptor

    async def _embed(self, texts: list[str]) -> np.ndarray:
        response = await (await self._client()).post(
            f"{self.base_url}/api/embed",
            json={
                "model": self.settings.embedding_model,
                "input": texts,
                "dimensions": self.settings.embedding_dimension,
                "truncate": False,
                "keep_alive": self.settings.ollama_keep_alive,
            },
        )
        response.raise_for_status()
        return validate_vectors(
            response.json().get("embeddings"),
            len(texts),
            self.settings.embedding_dimension,
        )

    async def embed_documents(self, texts: list[str]) -> np.ndarray:
        return await self._embed(texts)

    async def embed_query(self, query: str) -> np.ndarray:
        instruction = self.settings.ollama_query_instruction.strip()
        text = f"Instruct: {instruction}\nQuery: {query}" if instruction else query
        return await self._embed([text])

    async def close(self) -> None:
        if self.client is not None:
            await self.client.aclose()
            self.client = None


class DashScopeEmbeddingProvider:
    def __init__(self, settings):
        self.settings = settings
        self.client: httpx.AsyncClient | None = None

    async def _client(self) -> httpx.AsyncClient:
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=self.settings.index_embedding_timeout_seconds)
        return self.client

    async def prepare(self) -> EmbeddingDescriptor:
        if not self.settings.dashscope_api_key:
            raise RuntimeError("缺少 DASHSCOPE_API_KEY")
        return EmbeddingDescriptor(
            provider="dashscope",
            model=self.settings.embedding_model,
            digest=self.settings.embedding_model,
            dimension=self.settings.embedding_dimension,
        )

    async def _embed(self, texts: list[str]) -> np.ndarray:
        response = await (await self._client()).post(
            self.settings.dashscope_base_url.rstrip("/") + "/embeddings",
            headers={"Authorization": f"Bearer {self.settings.dashscope_api_key}"},
            json={
                "model": self.settings.embedding_model,
                "input": texts,
                "dimensions": self.settings.embedding_dimension,
            },
        )
        response.raise_for_status()
        data = response.json().get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            raise ValueError("Embedding 返回数量与请求数量不一致")
        try:
            ordered = sorted(data, key=lambda item: int(item["index"]))
            indexes = [int(item["index"]) for item in ordered]
            vectors = [item["embedding"] for item in ordered]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Embedding 响应结构无效") from exc
        if indexes != list(range(len(texts))):
            raise ValueError("Embedding 返回顺序无效")
        return validate_vectors(vectors, len(texts), self.settings.embedding_dimension)

    async def embed_documents(self, texts: list[str]) -> np.ndarray:
        return await self._embed(texts)

    async def embed_query(self, query: str) -> np.ndarray:
        return await self._embed([query])

    async def close(self) -> None:
        if self.client is not None:
            await self.client.aclose()
            self.client = None


def create_embedding_provider(settings) -> EmbeddingProvider:
    provider = settings.embedding_provider.strip().lower()
    if provider == "ollama":
        return OllamaEmbeddingProvider(settings)
    if provider == "dashscope":
        return DashScopeEmbeddingProvider(settings)
    raise ValueError(f"不支持的 EMBEDDING_PROVIDER：{settings.embedding_provider}")
