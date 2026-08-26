from types import SimpleNamespace

import pytest

from backend.app.core.resource_budget import MonthlyResourceBudget, ResourceBudgetExceeded
from backend.app.evaluation.profiles import case_hash, load_cases, select_cases, to_examples
from backend.app.evaluation.targets import RetrievalTarget
from scripts.run_langsmith_eval import (
    _formal_eval_cache_disabled,
    _local_evaluate,
    _run_learn,
)


def test_monthly_budget_reserve_and_settle(tmp_path):
    ledger = MonthlyResourceBudget(tmp_path / "budget.json")
    ledger.reserve("evaluation_traces", 10, 12)
    with pytest.raises(ResourceBudgetExceeded):
        ledger.reserve("evaluation_traces", 3, 12)
    ledger.settle("evaluation_traces", 10, 4)
    assert ledger.snapshot()["evaluation_traces"] == 4


def test_multi_resource_reservation_is_atomic(tmp_path):
    ledger = MonthlyResourceBudget(tmp_path / "budget.json")
    with pytest.raises(ResourceBudgetExceeded):
        ledger.reserve_many(
            {"evaluation_traces": 3, "judge_calls": 3},
            {"evaluation_traces": 10, "judge_calls": 2},
        )
    state = ledger.snapshot()
    assert state["evaluation_traces"] == 0
    assert state["judge_calls"] == 0


def test_stratified_selection_is_deterministic():
    cases = load_cases("lawstation-agent-v3")
    categories = ["casual", "clarification", "matched", "no_match", "tool_error", "memory"]
    first = select_cases(cases, limit=6, categories=categories, seed=42)
    second = select_cases(cases, limit=6, categories=categories, seed=42)
    assert [case_hash(item) for item in first] == [case_hash(item) for item in second]
    assert {item["metadata"]["category"] for item in first} == set(categories)


def test_learn_profile_has_no_external_resource_usage(tmp_path):
    args = SimpleNamespace(
        dataset="lawstation-agent-v3",
        sample_seed=42,
        sample_categories=[],
        max_examples=6,
        output=str(tmp_path / "learn.json"),
    )
    result = _run_learn(args)
    assert result["sample_size"] == 6
    assert result["resource_usage"]["planned_traces"] == 0
    assert result["resource_usage"]["agent_model_calls"] == 0
    assert result["resource_usage"]["judge_calls"] == 0
    assert all(metric["mean"] == 1 for metric in result["metrics"].values())


@pytest.mark.asyncio
async def test_local_examples_can_run_without_uploading(tmp_path):
    cases = select_cases(load_cases("lawstation-agent-v3"), limit=1, categories=["casual"], seed=42)

    async def target(inputs):
        return {"final_answer": inputs["question"]}

    rows, cache_hits = await _local_evaluate(
        target,
        to_examples("lawstation-agent-v3", cases),
        [],
        repetitions=1,
        cache_dir=tmp_path / "cache",
        cache_metadata={},
    )
    assert len(rows) == 1
    assert cache_hits == 0


@pytest.mark.asyncio
async def test_formal_local_run_can_disable_test_cache(tmp_path):
    cases = select_cases(
        load_cases("lawstation-agent-v3"), limit=1, categories=["casual"], seed=42
    )
    calls = 0

    async def target(inputs):
        nonlocal calls
        calls += 1
        return {"final_answer": inputs["question"]}

    examples = to_examples("lawstation-agent-v3", cases)
    await _local_evaluate(
        target,
        examples,
        [],
        repetitions=2,
        cache_dir=tmp_path / "cache",
        cache_metadata={},
        use_cache=False,
    )
    assert calls == 2
    assert not (tmp_path / "cache").exists()


def test_formal_cloud_evaluation_disables_and_restores_vcr_cache(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TEST_CACHE", "/tmp/lawstation-test-cache")
    with _formal_eval_cache_disabled():
        assert "LANGSMITH_TEST_CACHE" not in __import__("os").environ
    assert __import__("os").environ["LANGSMITH_TEST_CACHE"] == "/tmp/lawstation-test-cache"


@pytest.mark.asyncio
async def test_retrieval_target_records_runtime_reranker_version():
    class Engine:
        async def search(self, *_args, **_kwargs):
            return [{
                "chunk_id": "chunk-1",
                "rerank_applied": True,
                "rerank_duration_ms": 12.5,
                "ranking_version": "BAAI/bge-reranker-v2-m3@sha-123",
            }]

        def status(self):
            return {
                "dense_enabled": True,
                "reranker_status": "ready",
                "reranker_provider": "tei",
                "reranker_model": "BAAI/bge-reranker-v2-m3",
                "reranker_model_digest": "sha-123",
            }

        async def close(self):
            return None

    result = await RetrievalTarget(Engine(), "hybrid")({"question": "测试"})

    assert result["reranker_provider"] == "tei"
    assert result["reranker_model_digest"] == "sha-123"
    assert result["ranking_version"] == "BAAI/bge-reranker-v2-m3@sha-123"
