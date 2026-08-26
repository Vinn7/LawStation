import json
from types import SimpleNamespace

import httpx
import pytest

from mcp_servers.law_rag.reranker import (
    OllamaReranker,
    RerankCandidate,
    RerankerUnavailable,
    TEIReranker,
    create_reranker,
    extract_yes_no_score,
    parse_tei_ranks,
)


def settings():
    return SimpleNamespace(
        ollama_base_url="http://127.0.0.1:11434",
        rag_rerank_model="dengcao/Qwen3-Reranker-0.6B:latest",
        rag_rerank_concurrency=2,
        rag_rerank_request_timeout_seconds=30,
        rag_rerank_top_logprobs=20,
        rag_rerank_keep_alive="30m",
        rag_rerank_stage_timeout_seconds=20,
        rag_rerank_retry_seconds=60,
    )


def response(score_yes: float, score_no: float, prompt_tokens: int = 10):
    return {
        "response": "yes",
        "prompt_eval_count": prompt_tokens,
        "logprobs": [{
            "token": " yes",
            "logprob": score_yes,
            "top_logprobs": [
                {"token": " yes", "logprob": score_yes},
                {"token": "NO", "logprob": score_no},
            ],
        }],
    }


def tei_settings():
    return SimpleNamespace(
        rag_rerank_provider="tei",
        rag_rerank_model="BAAI/bge-reranker-v2-m3",
        rag_rerank_model_revision="pinned-bge-sha",
        rag_rerank_model_cache="",
        rag_rerank_base_url="http://127.0.0.1:8081",
        rag_rerank_request_timeout_seconds=30,
        rag_rerank_stage_timeout_seconds=20,
        rag_rerank_retry_seconds=60,
        rag_rerank_enabled=True,
    )


def test_extract_yes_no_score_normalizes_case_and_whitespace():
    score = extract_yes_no_score(response(2.0, 0.0))
    assert score == pytest.approx(0.880797, rel=1e-5)


def test_extract_yes_no_score_rejects_incomplete_distribution():
    with pytest.raises(ValueError, match="yes/no"):
        extract_yes_no_score({
            "logprobs": [{
                "token": "yes",
                "logprob": 0.0,
                "top_logprobs": [{"token": "yes", "logprob": 0.0}],
            }]
        })


@pytest.mark.asyncio
async def test_ollama_reranker_uses_generate_logprobs_and_stable_scores(monkeypatch):
    requests = []

    def handler(request: httpx.Request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{
                "name": "dengcao/Qwen3-Reranker-0.6B:latest",
                "digest": "digest-rerank",
            }]})
        body = json.loads(request.content)
        requests.append(body)
        positive = "海商法" not in body["prompt"]
        return httpx.Response(200, json=response(4 if positive else -2, -2 if positive else 4))

    reranker = OllamaReranker(settings())
    monkeypatch.setattr("mcp_servers.law_rag.reranker.get_ollama_runtime_info", lambda: None)
    reranker.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await reranker.prepare()
    batch = await reranker.rerank("未签劳动合同", [
        RerankCandidate(1, "irrelevant", "海商法 船舶碰撞", 0.04),
        RerankCandidate(2, "relevant", "劳动合同法 未签书面劳动合同", 0.03),
    ])

    assert [item.chunk_id for item in batch.results] == ["relevant", "irrelevant"]
    assert batch.prompt_tokens == 20
    assert all(item["raw"] is True and item["stream"] is False for item in requests)
    assert all(item["think"] is False and item["logprobs"] is True for item in requests)
    assert all(item["options"]["num_predict"] == 1 for item in requests)
    await reranker.close()


@pytest.mark.asyncio
async def test_invalid_candidate_response_degrades_entire_rerank_batch():
    def handler(request: httpx.Request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{
                "name": "dengcao/Qwen3-Reranker-0.6B:latest",
                "digest": "digest-rerank",
            }]})
        return httpx.Response(200, json={"logprobs": []})

    reranker = OllamaReranker(settings())
    reranker.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await reranker.prepare()
    with pytest.raises(RerankerUnavailable):
        await reranker.rerank("查询", [RerankCandidate(1, "chunk", "法条", 0.1)])
    assert reranker.status()["status"] == "cooldown"
    with pytest.raises(RerankerUnavailable, match="冷却期"):
        await reranker.rerank("查询", [RerankCandidate(1, "chunk", "法条", 0.1)])
    await reranker.close()


def test_parse_tei_ranks_requires_complete_unique_normalized_scores():
    assert parse_tei_ranks(
        [{"index": 1, "score": 0.2}, {"index": 0, "score": 0.9}], 2
    ) == [{"index": 1, "score": 0.2}, {"index": 0, "score": 0.9}]
    with pytest.raises(ValueError, match="数量不完整"):
        parse_tei_ranks([{"index": 0, "score": 0.9}], 2)
    with pytest.raises(ValueError, match="重复或越界"):
        parse_tei_ranks(
            [{"index": 0, "score": 0.9}, {"index": 0, "score": 0.8}], 2
        )
    with pytest.raises(ValueError, match="归一化分数"):
        parse_tei_ranks([{"index": 0, "score": 1.1}], 1)


@pytest.mark.asyncio
async def test_tei_reranker_batches_candidates_and_restores_chunk_ids(monkeypatch):
    requests = []

    def handler(request: httpx.Request):
        if request.url.path == "/health":
            return httpx.Response(200, text="ok")
        if request.url.path == "/info":
            return httpx.Response(200, json={
                "model_id": "BAAI/bge-reranker-v2-m3",
                "model_sha": None,
                "model_type": {"reranker": {}},
            })
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(
            200,
            headers={"x-compute-tokens": "24"},
            json=[{"index": 1, "score": 0.95}, {"index": 0, "score": 0.05}],
        )

    reranker = TEIReranker(tei_settings())
    monkeypatch.setattr(
        "mcp_servers.law_rag.reranker.get_tei_reranker_runtime_info", lambda: None
    )
    reranker.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    descriptor = await reranker.prepare()
    batch = await reranker.rerank("未签劳动合同", [
        RerankCandidate(8, "irrelevant", "海商法 船舶碰撞", 0.04),
        RerankCandidate(3, "relevant", "劳动合同法 未签书面合同", 0.03),
    ])
    assert descriptor.ranking_version == "BAAI/bge-reranker-v2-m3@pinned-bge-sha"
    assert [item.chunk_id for item in batch.results] == ["relevant", "irrelevant"]
    assert [item.index for item in batch.results] == [3, 8]
    assert batch.prompt_tokens == 24
    assert len(requests) == 1
    assert requests[0] == {
        "query": "未签劳动合同",
        "texts": ["海商法 船舶碰撞", "劳动合同法 未签书面合同"],
        "truncate": True,
        "raw_scores": False,
        "return_text": False,
    }
    await reranker.close()


@pytest.mark.asyncio
async def test_tei_partial_response_degrades_entire_batch(monkeypatch):
    def handler(request: httpx.Request):
        if request.url.path == "/health":
            return httpx.Response(200, text="ok")
        if request.url.path == "/info":
            return httpx.Response(200, json={
                "model_id": "BAAI/bge-reranker-v2-m3",
                "model_sha": "bge-sha",
                "model_type": {"reranker": {}},
            })
        return httpx.Response(200, json=[{"index": 0, "score": 0.5}])

    reranker = TEIReranker(tei_settings())
    monkeypatch.setattr(
        "mcp_servers.law_rag.reranker.get_tei_reranker_runtime_info", lambda: None
    )
    reranker.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await reranker.prepare()
    candidates = [
        RerankCandidate(0, "one", "法条一", 0.1),
        RerankCandidate(1, "two", "法条二", 0.09),
    ]
    with pytest.raises(RerankerUnavailable, match="调用失败"):
        await reranker.rerank("查询", candidates)
    assert reranker.status()["status"] == "cooldown"
    await reranker.close()


def test_create_reranker_defaults_to_tei():
    assert isinstance(create_reranker(tei_settings()), TEIReranker)
