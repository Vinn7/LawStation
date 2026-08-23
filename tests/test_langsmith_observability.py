from types import SimpleNamespace

from backend.app.core.config import Settings
from backend.app.evaluation.evaluators import (
    citation_grounding,
    exact_article_hit,
    loop_limit,
    no_match_safety,
    route_correctness,
)
from backend.app.observability.langsmith import LangSmithObservability, _safe_payload


def test_sensitive_trace_fields_and_reasoning_are_removed():
    value = {
        "messages": [{"content": "电话13812345678"}],
        "reasoning_content": "hidden chain of thought",
        "authorization": "Bearer secret-value",
        "nested": {"database_url": "sqlite:///secret.db"},
    }
    cleaned = _safe_payload(value, capture_content=True)
    assert "13812345678" not in str(cleaned)
    assert cleaned["reasoning_content"] == "<redacted>"
    assert cleaned["authorization"] == "<redacted>"
    assert cleaned["nested"]["database_url"] == "<redacted>"
    assert _safe_payload(value, capture_content=False) == {}


def test_sampling_and_identity_hashing_are_stable():
    settings = Settings(
        _env_file=None,
        langsmith_enabled=False,
        langsmith_id_hash_secret="test-secret",
    )
    observability = LangSmithObservability(settings)
    assert observability._stable_hash("user-a") == observability._stable_hash("user-a")
    assert observability._stable_hash("user-a") != observability._stable_hash("user-b")
    assert observability._sampled("request-1", 1.0)
    assert not observability._sampled("request-1", 0.0)


def test_production_trace_budget_is_fail_open(tmp_path):
    settings = Settings(
        _env_file=None,
        langsmith_enabled=False,
        eval_resource_budget_path=str(tmp_path / "budget.json"),
        langsmith_monthly_production_trace_budget=1,
    )
    observability = LangSmithObservability(settings)
    assert observability._production_slot()
    assert not observability._production_slot()


def test_deterministic_evaluators_cover_evidence_boundaries():
    run = SimpleNamespace(outputs={
        "final_answer": "当前法规库未检索到可引用法条。仅作一般分析。",
        "case_analysis": {"next_action": "research"},
        "evidence_packet": {"retrieval_status": "no_match", "evidence_items": []},
        "citations": [],
        "tool_call_count": 1,
        "model_call_count": 4,
        "retry_count": 0,
        "revision_count": 0,
    })
    example = SimpleNamespace(outputs={"expected_route": "research"}, inputs={})
    assert route_correctness(run, example)["score"] == 1
    assert citation_grounding(run, example)["score"] == 1
    assert no_match_safety(run, example)["score"] == 1
    assert loop_limit(run, example)["score"] == 1


def test_retrieval_evaluator_uses_real_chunk_ids():
    run = SimpleNamespace(outputs={
        "retrieval_status": "matched",
        "retrieval_results": [
            {"document_id": "doc-1", "chunk_id": "chunk-1"},
            {"document_id": "doc-2", "chunk_id": "chunk-2"},
        ],
    })
    example = SimpleNamespace(
        outputs={"expected_chunk_ids": ["chunk-2"]}, inputs={}
    )
    assert exact_article_hit(run, example)["score"] == 1
