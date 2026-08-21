from types import SimpleNamespace

import numpy as np
import pytest

import mcp_servers.law_rag.embeddings as embedding_module


def settings():
    return SimpleNamespace(
        ollama_base_url="http://127.0.0.1:11434",
        ollama_request_timeout_seconds=300,
        ollama_keep_alive="30m",
        ollama_query_instruction="Retrieve relevant Chinese laws",
        embedding_model="qwen3-embedding:0.6b",
        embedding_dimension=4,
    )


class Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class Client:
    def __init__(self, vectors):
        self.vectors = vectors
        self.requests = []

    async def post(self, url, json):
        self.requests.append((url, json))
        return Response({"embeddings": self.vectors})

    async def aclose(self):
        return None


@pytest.mark.asyncio
async def test_ollama_documents_and_query_use_different_inputs(monkeypatch):
    runtime = SimpleNamespace(
        available=True,
        model="qwen3-embedding:0.6b",
        digest="digest-1",
    )
    monkeypatch.setattr(embedding_module, "get_ollama_runtime_info", lambda: runtime)
    provider = embedding_module.OllamaEmbeddingProvider(settings())
    client = Client([[1, 0, 0, 0], [0, 1, 0, 0]])
    provider.client = client
    descriptor = await provider.prepare()
    assert descriptor.digest == "digest-1"

    documents = await provider.embed_documents(["法条一", "法条二"])
    assert documents.shape == (2, 4)
    assert client.requests[0][1] == {
        "model": "qwen3-embedding:0.6b",
        "input": ["法条一", "法条二"],
        "dimensions": 4,
        "truncate": False,
        "keep_alive": "30m",
    }

    client.vectors = [[0, 0, 1, 0]]
    query = await provider.embed_query("解除劳动合同")
    assert query.shape == (1, 4)
    assert client.requests[1][1]["input"] == [
        "Instruct: Retrieve relevant Chinese laws\nQuery: 解除劳动合同"
    ]


@pytest.mark.asyncio
async def test_ollama_rejects_invalid_vector_dimension(monkeypatch):
    runtime = SimpleNamespace(
        available=True,
        model="qwen3-embedding:0.6b",
        digest="digest-1",
    )
    monkeypatch.setattr(embedding_module, "get_ollama_runtime_info", lambda: runtime)
    provider = embedding_module.OllamaEmbeddingProvider(settings())
    provider.client = Client([[1, 2]])
    await provider.prepare()
    with pytest.raises(ValueError, match="数量或维度"):
        await provider.embed_documents(["法条"])


def test_validate_vectors_rejects_non_finite_values():
    with pytest.raises(ValueError, match="非有限"):
        embedding_module.validate_vectors([[1, 2, np.nan, 4]], 1, 4)
