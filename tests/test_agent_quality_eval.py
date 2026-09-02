import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from backend.app.core.config import Settings
from backend.app.evaluation.agent_quality import (
    AgentQualityJudge,
    answer_quality_safety_evaluators,
    factual_fidelity_evaluators,
    reviewer_effectiveness_evaluators,
)
from backend.app.evaluation.profiles import load_cases
from backend.app.evaluation.targets import reviewer_target
from scripts import run_agent_quality_eval


def test_agent_quality_datasets_are_frozen_at_expected_sizes_and_use_real_chunks():
    assert len(load_cases("lawstation-factual-fidelity-v1")) == 10
    assert len(load_cases("lawstation-reviewer-effectiveness-v1")) == 12
    answer = load_cases("lawstation-answer-quality-v1")
    assert len(answer) == 8
    law_data = json.loads(Path("data/knowledge/law/law.json").read_text("utf-8"))
    known_content = set(law_data.values())
    documents = [
        item
        for case in answer
        for item in case["inputs"].get("fixture_documents", [])
    ]
    assert documents
    assert all(item["content"] in known_content for item in documents)
    assert all(item["chunk_id"] and item["document_id"] for item in documents)


def test_factual_metrics_detect_latest_fact_and_old_fact_leak():
    example = SimpleNamespace(outputs={
        "expected_current_facts": [{"key": "amount", "value": "五万元"}],
        "forbidden_old_facts": [{"key": "amount", "value": "三万元"}],
        "forbidden_invented_facts": [],
    })
    good = SimpleNamespace(outputs={"final_answer": "欠款为五万元。"})
    bad = SimpleNamespace(outputs={"final_answer": "欠款为三万元。"})
    good_scores = {item["key"]: item["score"] for item in factual_fidelity_evaluators(good, example)}
    bad_scores = {item["key"]: item["score"] for item in factual_fidelity_evaluators(bad, example)}
    assert good_scores["latest_fact_priority_rate"] == 1
    assert good_scores["old_fact_leak_rate"] == 0
    assert bad_scores["latest_fact_priority_rate"] == 0
    assert bad_scores["old_fact_leak_rate"] == 1


def test_reviewer_metrics_distinguish_detection_escape_and_safe_acceptance():
    unsafe_example = SimpleNamespace(outputs={
        "expected_unsafe": True,
        "expected_action": "revise_draft",
        "forbidden_terms": ["一定胜诉"],
    })
    unsafe_run = SimpleNamespace(outputs={
        "initial_review_result": {"approved": False, "next_action": "revise_draft"},
        "revision_performed": True,
        "final_answer": "现有信息不足以作出确定结论。",
        "review_mode": "llm",
    })
    scores = {item["key"]: item["score"] for item in reviewer_effectiveness_evaluators(unsafe_run, unsafe_example)}
    assert scores["reviewer_detection_recall"] == 1
    assert scores["unsafe_draft_escape_rate"] == 0
    assert scores["revision_success_rate"] == 1


def test_answer_quality_gate_cannot_be_hidden_by_style_scores():
    run = SimpleNamespace(outputs={
        "final_answer": "《虚构法》第九十九条明确规定。",
        "evidence_packet": {"retrieval_status": "no_match", "evidence_items": []},
        "citations": [],
        "counsel_claims": [],
    })
    example = SimpleNamespace(outputs={"forbidden_terms": []})
    scores = {item["key"]: item["score"] for item in answer_quality_safety_evaluators(run, example)}
    assert scores["no_match_safety"] == 0
    assert scores["quality_gate_passed"] == 0


def test_agent_quality_judge_uses_one_json_call_and_returns_all_dimensions(monkeypatch):
    calls = []

    class Bound:
        async def ainvoke(self, messages):
            calls.append(messages)
            return AIMessage(content=json.dumps({
                "facts": [{"fact_key": "amount", "status": "preserved", "comment": ""}],
                "factual_fidelity": 5,
                "fabrication_control": 5,
                "uncertainty_preservation": 4,
                "context_consistency": 5,
                "comment": "ok",
            }))

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def bind(self, **kwargs):
            assert kwargs == {"response_format": {"type": "json_object"}}
            return Bound()

    monkeypatch.setattr("backend.app.evaluation.agent_quality.ChatOpenAI", FakeChatOpenAI)
    judge = AgentQualityJudge(
        "factual-fidelity",
        Settings(_env_file=None, deepseek_api_key="test-key"),
    )
    result = asyncio.run(judge(
        SimpleNamespace(outputs={"final_answer": "五万元"}),
        SimpleNamespace(inputs={"question": "金额？"}, outputs={}),
    ))
    assert len(calls) == 1
    assert {item["key"] for item in result} == {
        "judge_factual_fidelity", "judge_fabrication_control",
        "judge_uncertainty_preservation", "judge_context_consistency",
    }


def test_reviewer_judge_derives_error_category_recall_without_second_call(monkeypatch):
    calls = []

    class Bound:
        async def ainvoke(self, messages):
            calls.append(messages)
            return AIMessage(content=json.dumps({
                "detected_error_labels": ["fabricated_citation"],
                "missed_error_labels": ["overclaiming"],
                "false_positive_labels": [],
                "instruction_specificity": 4,
                "decision_quality": 4,
                "comment": "partial",
            }))

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def bind(self, **kwargs):
            return Bound()

    monkeypatch.setattr("backend.app.evaluation.agent_quality.ChatOpenAI", FakeChatOpenAI)
    judge = AgentQualityJudge(
        "reviewer-effectiveness",
        Settings(_env_file=None, deepseek_api_key="test-key"),
    )
    result = asyncio.run(judge(
        SimpleNamespace(outputs={"final_answer": ""}),
        SimpleNamespace(
            inputs={},
            outputs={"injected_error_labels": ["fabricated_citation", "overclaiming"]},
        ),
    ))
    scores = {item["key"]: item["score"] for item in result}
    assert len(calls) == 1
    assert scores["error_category_recall"] == 0.5


def test_runner_requires_explicit_upload_confirmation(monkeypatch):
    monkeypatch.setattr(
        run_agent_quality_eval,
        "get_settings",
        lambda: Settings(_env_file=None, deepseek_api_key="test"),
    )
    args = SimpleNamespace(suite="all", upload_results=False, confirm_upload=True)
    with pytest.raises(SystemExit, match="--upload-results --confirm-upload"):
        asyncio.run(run_agent_quality_eval.execute(args))


def test_reviewer_target_executes_production_fast_path_without_other_agents(monkeypatch):
    class Provider:
        def __init__(self, _settings):
            pass

        def get_chat_model(self):
            return FakeMessagesListChatModel(responses=[AIMessage(content="{}")])

    monkeypatch.setattr("backend.app.evaluation.targets.LLMProvider", Provider)
    case = load_cases("lawstation-reviewer-effectiveness-v1")[-1]
    target = reviewer_target(Settings(_env_file=None, deepseek_api_key="test"))
    output = asyncio.run(target(case["inputs"]))
    assert output["review_mode"] == "deterministic"
    assert output["model_call_count"] == 0
    assert output["initial_review_result"]["approved"] is True
    assert output["final_answer"]
