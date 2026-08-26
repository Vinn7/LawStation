from types import SimpleNamespace

import httpx
import numpy as np
import pytest

import mcp_servers.law_rag.engine as engine_module
from mcp_servers.law_rag.embeddings import EmbeddingDescriptor
from mcp_servers.law_rag.reranker import (
    RerankBatch,
    RerankerDescriptor,
    RerankerUnavailable,
    RerankResult,
)


def settings(source, index_dir):
    return SimpleNamespace(
        law_data_path=str(source), index_dir=str(index_dir),
        index_chunk_max_chars=1000, index_chunk_overlap_chars=150,
        embedding_provider="dashscope", embedding_model="test-embedding", embedding_dimension=4,
        index_auto_build=True, dashscope_api_key="test", dashscope_base_url="http://unused",
        ollama_embedding_batch_size=8,
        index_build_batch_size=2,
        index_embedding_timeout_seconds=120,
        index_embedding_max_retries=2,
        index_embedding_retry_base_seconds=0,
        index_embedding_retry_max_seconds=0,
        rag_bm25_min_score=0.01,
        rag_dense_min_score=0.20,
        rag_rrf_min_score=0.01,
    )


@pytest.mark.asyncio
async def test_valid_manifest_skips_embedding(tmp_path, monkeypatch):
    source = tmp_path / "law.json"
    source.write_text('{"民法典第一条":"第一条内容","刑法第二条":"第二条内容"}', encoding="utf-8")
    config = settings(source, tmp_path / "indexes")
    monkeypatch.setattr(engine_module, "get_settings", lambda: config)
    monkeypatch.setattr(engine_module, "audit", lambda *args, **kwargs: None)
    first = engine_module.LawSearchEngine()
    first._record_manifest = lambda: None
    calls = 0

    async def embed(texts):
        nonlocal calls
        calls += 1
        return np.ones((len(texts), 4), dtype="float32")

    first._embed = embed
    monkeypatch.setattr(np, "vstack", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("不得全量 vstack")))
    await first.initialize_index(wait=True)
    assert first.status()["status"] == "ready"
    assert calls == 1

    second = engine_module.LawSearchEngine()
    async def should_not_run(_):
        raise AssertionError("有效索引不应再次调用 Embedding")
    second._embed = should_not_run
    await second.initialize_index(wait=True)
    assert second.status()["status"] == "ready"

    forced = engine_module.LawSearchEngine()
    forced._record_manifest = lambda: None
    forced_calls = 0
    async def forced_embed(texts):
        nonlocal forced_calls
        forced_calls += 1
        return np.ones((len(texts), 4), dtype="float32")
    forced._embed = forced_embed
    await forced.initialize_index(wait=True, force=True)
    assert forced_calls == 1


def test_long_article_splits_with_stable_ids(tmp_path, monkeypatch):
    source = tmp_path / "law.json"
    source.write_text('{"测试法第一条":"' + "甲" * 1200 + '"}', encoding="utf-8")
    monkeypatch.setattr(engine_module, "get_settings", lambda: settings(source, tmp_path / "indexes"))
    first = engine_module.LawSearchEngine()
    second = engine_module.LawSearchEngine()
    assert len(first.docs) == 2
    assert [item["chunk_id"] for item in first.docs] == [item["chunk_id"] for item in second.docs]


@pytest.mark.asyncio
async def test_source_fingerprint_change_triggers_rebuild(tmp_path, monkeypatch):
    sample = tmp_path / "law_sample.json"
    full = tmp_path / "law.json"
    sample.write_text('{"民法典第一条":"样本内容"}', encoding="utf-8")
    full.write_text('{"民法典第一条":"全量内容","刑法第二条":"第二条内容"}', encoding="utf-8")
    index_dir = tmp_path / "indexes"
    monkeypatch.setattr(engine_module, "audit", lambda *args, **kwargs: None)

    monkeypatch.setattr(engine_module, "get_settings", lambda: settings(sample, index_dir))
    sample_engine = engine_module.LawSearchEngine()
    sample_engine._record_manifest = lambda: None

    async def sample_embed(texts):
        return np.ones((len(texts), 4), dtype="float32")

    sample_engine._embed = sample_embed
    await sample_engine.initialize_index(wait=True)

    monkeypatch.setattr(engine_module, "get_settings", lambda: settings(full, index_dir))
    full_engine = engine_module.LawSearchEngine()
    full_engine._record_manifest = lambda: None
    calls = []

    async def full_embed(texts):
        calls.append(len(texts))
        return np.ones((len(texts), 4), dtype="float32")

    full_engine._embed = full_embed
    await full_engine.initialize_index(wait=True)
    assert calls == [2]
    assert full_engine.status()["status"] == "ready"


@pytest.mark.asyncio
async def test_corrupt_or_missing_batches_are_regenerated(tmp_path, monkeypatch):
    source = tmp_path / "law.json"
    source.write_text(
        '{"法第一条":"一","法第二条":"二","法第三条":"三"}',
        encoding="utf-8",
    )
    config = settings(source, tmp_path / "indexes")
    monkeypatch.setattr(engine_module, "get_settings", lambda: config)
    monkeypatch.setattr(engine_module, "audit", lambda *args, **kwargs: None)
    first = engine_module.LawSearchEngine()
    first._record_manifest = lambda: None
    calls = 0

    async def interrupted_embed(texts):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return np.ones((len(texts), 4), dtype="float32")

    first._embed = interrupted_embed
    await first.initialize_index(wait=True)
    assert first.status()["status"] == "failed"

    staging = config.index_dir + f"/.staging-{first.fingerprint}"
    corrupt = engine_module.Path(staging) / "batch-00000000.npy"
    corrupt.write_bytes(b"invalid")
    second = engine_module.LawSearchEngine()
    second._record_manifest = lambda: None
    rebuilt = []

    async def resumed_embed(texts):
        rebuilt.append(len(texts))
        return np.ones((len(texts), 4), dtype="float32")

    second._embed = resumed_embed
    await second.initialize_index(wait=True)
    assert rebuilt == [2, 1]
    assert second.status()["status"] == "ready"


@pytest.mark.asyncio
async def test_build_caps_embedding_batches_at_twenty(tmp_path, monkeypatch):
    source = tmp_path / "law.json"
    source.write_text(
        "{" + ",".join(f'"法第{i}条":"内容{i}"' for i in range(21)) + "}",
        encoding="utf-8",
    )
    config = settings(source, tmp_path / "indexes")
    config.index_build_batch_size = 100
    monkeypatch.setattr(engine_module, "get_settings", lambda: config)
    monkeypatch.setattr(engine_module, "audit", lambda *args, **kwargs: None)
    engine = engine_module.LawSearchEngine()
    engine._record_manifest = lambda: None
    batches = []

    async def embed(texts):
        batches.append(len(texts))
        return np.ones((len(texts), 4), dtype="float32")

    engine._embed = embed
    await engine.initialize_index(wait=True)
    assert batches == [20, 1]


@pytest.mark.asyncio
async def test_embedding_retry_only_for_retryable_errors(tmp_path, monkeypatch):
    source = tmp_path / "law.json"
    source.write_text('{"法第一条":"一"}', encoding="utf-8")
    monkeypatch.setattr(engine_module, "get_settings", lambda: settings(source, tmp_path / "indexes"))
    monkeypatch.setattr(engine_module, "audit", lambda *args, **kwargs: None)
    monkeypatch.setattr(engine_module.asyncio, "sleep", lambda _delay: _completed())
    engine = engine_module.LawSearchEngine()
    calls = 0

    async def retryable(_texts):
        nonlocal calls
        calls += 1
        if calls == 1:
            request = httpx.Request("POST", "http://unused/embeddings")
            response = httpx.Response(429, request=request)
            raise httpx.HTTPStatusError("rate limited", request=request, response=response)
        return np.ones((1, 4), dtype="float32")

    engine._embed = retryable
    result = await engine._embed_batch(["一"], batch_start=0)
    assert result.shape == (1, 4)
    assert calls == 2


@pytest.mark.asyncio
async def test_embedding_does_not_retry_deterministic_client_error(tmp_path, monkeypatch):
    source = tmp_path / "law.json"
    source.write_text('{"法第一条":"一"}', encoding="utf-8")
    monkeypatch.setattr(engine_module, "get_settings", lambda: settings(source, tmp_path / "indexes"))
    monkeypatch.setattr(engine_module, "audit", lambda *args, **kwargs: None)
    engine = engine_module.LawSearchEngine()
    calls = 0

    async def invalid_request(_texts):
        nonlocal calls
        calls += 1
        request = httpx.Request("POST", "http://unused/embeddings")
        response = httpx.Response(400, request=request)
        raise httpx.HTTPStatusError("invalid request", request=request, response=response)

    engine._embed = invalid_request
    with pytest.raises(httpx.HTTPStatusError):
        await engine._embed_batch(["一"], batch_start=0)
    assert calls == 1


@pytest.mark.asyncio
async def test_ollama_dense_build_does_not_require_dashscope_key(tmp_path, monkeypatch):
    source = tmp_path / "law.json"
    source.write_text('{"法第一条":"一","法第二条":"二","法第三条":"三"}', encoding="utf-8")
    config = settings(source, tmp_path / "indexes")
    config.embedding_provider = "ollama"
    config.embedding_model = "qwen3-embedding:0.6b"
    config.dashscope_api_key = ""
    config.ollama_embedding_batch_size = 2
    config.ollama_base_url = "http://unused"
    config.ollama_request_timeout_seconds = 1
    config.ollama_keep_alive = "30m"
    config.ollama_query_instruction = "Retrieve laws"
    monkeypatch.setattr(engine_module, "get_settings", lambda: config)
    monkeypatch.setattr(engine_module, "audit", lambda *args, **kwargs: None)
    engine = engine_module.LawSearchEngine()
    engine._record_manifest = lambda: None
    batches = []

    class Provider:
        async def prepare(self):
            return EmbeddingDescriptor("ollama", config.embedding_model, "digest-1", 4)

        async def embed_documents(self, texts):
            batches.append(len(texts))
            return np.ones((len(texts), 4), dtype="float32")

        async def embed_query(self, _query):
            return np.ones((1, 4), dtype="float32")

        async def close(self):
            return None

    engine.embedding_provider = Provider()
    await engine.initialize_index(wait=True)
    assert batches == [2, 1]
    assert engine.status()["status"] == "ready"
    assert engine.status()["embedding_provider"] == "ollama"
    assert engine.status()["embedding_model_digest"] == "digest-1"
    assert engine.status()["ollama_available"] is True


@pytest.mark.asyncio
async def test_ollama_build_caps_batches_at_configured_eight(tmp_path, monkeypatch):
    source = tmp_path / "law.json"
    source.write_text(
        "{" + ",".join(f'"法第{i}条":"内容{i}"' for i in range(17)) + "}",
        encoding="utf-8",
    )
    config = settings(source, tmp_path / "indexes")
    config.embedding_provider = "ollama"
    config.embedding_model = "qwen3-embedding:0.6b"
    config.ollama_embedding_batch_size = 8
    config.index_build_batch_size = 100
    config.ollama_base_url = "http://unused"
    config.ollama_request_timeout_seconds = 1
    config.ollama_keep_alive = "30m"
    config.ollama_query_instruction = "Retrieve laws"
    monkeypatch.setattr(engine_module, "get_settings", lambda: config)
    monkeypatch.setattr(engine_module, "audit", lambda *args, **kwargs: None)
    engine = engine_module.LawSearchEngine()
    engine._record_manifest = lambda: None
    batches = []

    class Provider:
        async def prepare(self):
            return EmbeddingDescriptor("ollama", config.embedding_model, "digest-1", 4)

        async def embed_documents(self, texts):
            batches.append(len(texts))
            return np.ones((len(texts), 4), dtype="float32")

        async def embed_query(self, _query):
            return np.ones((1, 4), dtype="float32")

        async def close(self):
            return None

    engine.embedding_provider = Provider()
    await engine.initialize_index(wait=True)
    assert batches == [8, 8, 1]


async def _completed():
    return None


@pytest.mark.asyncio
async def test_irrelevant_lexical_query_returns_normal_empty_result(tmp_path, monkeypatch):
    source = tmp_path / "law.json"
    source.write_text('{"劳动法第一条":"劳动权益内容"}', encoding="utf-8")
    monkeypatch.setattr(engine_module, "get_settings", lambda: settings(source, tmp_path / "indexes"))
    engine = engine_module.LawSearchEngine()

    result = await engine.search("量子天体物理")

    assert result == []


@pytest.mark.asyncio
async def test_law_name_filter_is_applied_before_ranking(tmp_path, monkeypatch):
    source = tmp_path / "law.json"
    records = {f"普通法第{i}条": f"普通内容{i}" for i in range(40)}
    records["中华人民共和国目标法第一条"] = "特殊救济目标"
    source.write_text(engine_module.json.dumps(records, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(engine_module, "get_settings", lambda: settings(source, tmp_path / "indexes"))
    engine = engine_module.LawSearchEngine()

    result = await engine.search("特殊救济", filters={"law_name": "目标法"})

    assert len(result) == 1
    assert result[0]["law_name"] == "中华人民共和国目标法"
    assert result[0]["retrieval_scores"]["bm25"] > 0


@pytest.mark.asyncio
async def test_unknown_filter_is_rejected(tmp_path, monkeypatch):
    source = tmp_path / "law.json"
    source.write_text('{"测试法第一条":"内容"}', encoding="utf-8")
    monkeypatch.setattr(engine_module, "get_settings", lambda: settings(source, tmp_path / "indexes"))
    engine = engine_module.LawSearchEngine()

    with pytest.raises(ValueError, match="不支持的过滤字段"):
        await engine.search("内容", filters={"region": "北京"})


@pytest.mark.asyncio
async def test_bm25_evaluation_mode_does_not_call_dense(tmp_path, monkeypatch):
    source = tmp_path / "law.json"
    source.write_text('{"测试法第一条":"劳动合同解除赔偿"}', encoding="utf-8")
    config = settings(source, tmp_path / "indexes")
    config.rag_bm25_min_score = -1
    monkeypatch.setattr(engine_module, "get_settings", lambda: config)
    engine = engine_module.LawSearchEngine()
    engine.faiss = object()
    engine.embedding_descriptor = EmbeddingDescriptor("ollama", "test", "digest", 4)

    async def forbidden(_query):
        raise AssertionError("BM25-only 评测不得调用 Dense Embedding")

    engine._embed_query = forbidden
    result = await engine.search("劳动合同解除", retrieval_mode="bm25")

    assert result
    assert result[0]["retrieval_mode"] == "bm25"
    assert result[0]["retrieval_sources"] == ["bm25"]


@pytest.mark.asyncio
async def test_unknown_retrieval_mode_is_rejected(tmp_path, monkeypatch):
    source = tmp_path / "law.json"
    source.write_text('{"测试法第一条":"内容"}', encoding="utf-8")
    monkeypatch.setattr(engine_module, "get_settings", lambda: settings(source, tmp_path / "indexes"))
    engine = engine_module.LawSearchEngine()

    with pytest.raises(ValueError, match="retrieval_mode"):
        await engine.search("内容", retrieval_mode="invalid")


@pytest.mark.asyncio
async def test_reranker_reorders_rrf_candidates_without_changing_schema(tmp_path, monkeypatch):
    source = tmp_path / "law.json"
    source.write_text(
        '{"测试法第一条":"劳动合同解除","测试法第二条":"劳动赔偿责任"}',
        encoding="utf-8",
    )
    config = settings(source, tmp_path / "indexes")
    config.rag_bm25_min_score = -1
    config.rag_rerank_candidate_count = 12
    config.rag_rerank_min_score = 0
    monkeypatch.setattr(engine_module, "get_settings", lambda: config)
    engine = engine_module.LawSearchEngine()

    class FakeReranker:
        async def rerank(self, _query, candidates):
            return RerankBatch([
                RerankResult(
                    candidates[1].index, candidates[1].chunk_id, 0.9, candidates[1].rrf_score
                ),
                RerankResult(
                    candidates[0].index, candidates[0].chunk_id, 0.1, candidates[0].rrf_score
                ),
            ], 12.5, 40)

        def status(self):
            return {
                "status": "ready", "message": "ready", "provider": "ollama",
                "model": "reranker", "model_digest": "digest", "ranking_version": "reranker@digest",
            }

        async def close(self):
            return None

    engine.reranker = FakeReranker()
    engine.reranker_descriptor = RerankerDescriptor("ollama", "reranker", "digest")
    result = await engine.search("劳动", top_k=2, retrieval_mode="bm25")

    assert len(result) == 2
    assert result[0]["retrieval_scores"]["rerank"] == pytest.approx(0.9)
    assert result[0]["rerank_applied"] is True
    assert result[0]["ranking_version"] == "reranker@digest"
    assert result[0]["rerank_prompt_tokens"] == 40


@pytest.mark.asyncio
async def test_reranker_failure_falls_back_to_rrf_not_no_match(tmp_path, monkeypatch):
    source = tmp_path / "law.json"
    source.write_text('{"测试法第一条":"劳动合同解除"}', encoding="utf-8")
    config = settings(source, tmp_path / "indexes")
    config.rag_bm25_min_score = -1
    config.rag_rerank_candidate_count = 12
    monkeypatch.setattr(engine_module, "get_settings", lambda: config)
    engine = engine_module.LawSearchEngine()

    class FailedReranker:
        async def rerank(self, _query, _candidates):
            raise RerankerUnavailable("offline")

        def status(self):
            return {
                "status": "degraded", "message": "offline", "provider": "ollama",
                "model": "reranker", "model_digest": "", "ranking_version": "",
            }

        async def close(self):
            return None

    engine.reranker = FailedReranker()
    result = await engine.search("劳动", retrieval_mode="bm25")

    assert result
    assert result[0]["rerank_applied"] is False
    assert "rerank" not in result[0]["retrieval_scores"]
    assert result[0]["ranking_version"] == "rrf-v1"


@pytest.mark.asyncio
async def test_reranker_threshold_can_return_normal_no_match(tmp_path, monkeypatch):
    source = tmp_path / "law.json"
    source.write_text('{"测试法第一条":"劳动合同解除"}', encoding="utf-8")
    config = settings(source, tmp_path / "indexes")
    config.rag_bm25_min_score = -1
    config.rag_rerank_candidate_count = 12
    config.rag_rerank_min_score = 0.8
    monkeypatch.setattr(engine_module, "get_settings", lambda: config)
    engine = engine_module.LawSearchEngine()

    class LowScoreReranker:
        async def rerank(self, _query, candidates):
            item = candidates[0]
            return RerankBatch([
                RerankResult(item.index, item.chunk_id, 0.2, item.rrf_score)
            ], 3.0, 10)

        def status(self):
            return {
                "status": "ready", "message": "ready", "provider": "ollama",
                "model": "reranker", "model_digest": "digest", "ranking_version": "reranker@digest",
            }

        async def close(self):
            return None

    engine.reranker = LowScoreReranker()
    engine.reranker_descriptor = RerankerDescriptor("ollama", "reranker", "digest")
    assert await engine.search("劳动", retrieval_mode="bm25") == []


def test_exact_article_lookup_uses_map_and_returns_all_chunks(tmp_path, monkeypatch):
    source = tmp_path / "law.json"
    source.write_text('{"中华人民共和国测试法第一条":"' + "甲" * 1200 + '"}', encoding="utf-8")
    monkeypatch.setattr(engine_module, "get_settings", lambda: settings(source, tmp_path / "indexes"))
    engine = engine_module.LawSearchEngine()

    result = engine.get("测试法", "第一条")

    assert result["law_name"] == "中华人民共和国测试法"
    assert len(result["chunks"]) == 2
    assert all(item["chunk_id"] for item in result["chunks"])
