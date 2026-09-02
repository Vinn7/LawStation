"""Agent-quality evaluators and one-call structured LLM judges.

These evaluators intentionally sit outside the production graph.  They consume the
safe ``evaluation_output`` produced by AgentRuntime and never write conversations,
memories, AgentRuns, or checkpoints.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from backend.app.core.config import Settings, get_settings

JUDGE_PROMPT_VERSION = "agent-quality-judge-v1"
JUDGE_SCHEMA_VERSION = "agent-quality-schema-v1"


def _outputs(run: Any) -> dict[str, Any]:
    value = getattr(run, "outputs", None)
    if value is None and isinstance(run, dict):
        value = run.get("outputs")
    return value if isinstance(value, dict) else {}


def _reference(example: Any) -> dict[str, Any]:
    value = getattr(example, "outputs", None)
    if value is None and isinstance(example, dict):
        value = example.get("outputs")
    return value if isinstance(value, dict) else {}


def _feedback(key: str, score: float | bool | None, comment: str = "") -> dict[str, Any]:
    return {"key": key, "score": None if score is None else float(score), "comment": comment}


def factual_fidelity_evaluators(run: Any, example: Any) -> list[dict[str, Any]]:
    answer = str(_outputs(run).get("final_answer") or "")
    reference = _reference(example)
    current = [str(item.get("value", "")) for item in reference.get("expected_current_facts", [])]
    old = [str(item.get("value", "")) for item in reference.get("forbidden_old_facts", [])]
    forbidden = [str(item) for item in reference.get("forbidden_invented_facts", [])]
    current_hits = sum(bool(value and value in answer) for value in current)
    old_hits = sum(bool(value and value in answer) for value in old)
    forbidden_hits = sum(bool(value and value in answer) for value in forbidden)
    current_rate = current_hits / len(current) if current else 1.0
    old_leak = old_hits / len(old) if old else 0.0
    forbidden_rate = forbidden_hits / len(forbidden) if forbidden else 0.0
    return [
        _feedback("current_fact_presence_rate", current_rate),
        _feedback("old_fact_leak_rate", old_leak),
        _feedback("latest_fact_priority_rate", current_rate == 1 and old_leak == 0),
        _feedback("entity_value_accuracy", current_rate == 1 and old_leak == 0),
        _feedback("forbidden_fact_violation_rate", forbidden_rate),
    ]


def reviewer_effectiveness_evaluators(run: Any, example: Any) -> list[dict[str, Any]]:
    output = _outputs(run)
    reference = _reference(example)
    unsafe = bool(reference.get("expected_unsafe"))
    expected_action = str(reference.get("expected_action") or "finalize")
    review = output.get("initial_review_result") or {}
    action = str(review.get("next_action") or "finalize")
    detected = bool(not review.get("approved", False) and action != "finalize")
    final_answer = str(output.get("final_answer") or "")
    forbidden = [str(item) for item in reference.get("forbidden_terms", [])]
    escaped = unsafe and any(item and item in final_answer for item in forbidden)
    revision_required = expected_action == "revise_draft"
    revision_performed = bool(output.get("revision_performed"))
    revision_succeeded = (not escaped) if revision_required and revision_performed else None
    return [
        _feedback("reviewer_detection_recall", detected if unsafe else None),
        _feedback("reviewer_detection_precision", unsafe if detected else None),
        _feedback("reviewer_false_rejection_rate", detected if not unsafe else None),
        _feedback("reviewer_action_accuracy", action == expected_action),
        _feedback("unsafe_draft_escape_rate", escaped if unsafe else None),
        _feedback("revision_success_rate", revision_succeeded),
        _feedback(
            "fast_path_safety_rate",
            not escaped if output.get("review_mode") == "deterministic" else None,
        ),
    ]


def answer_quality_safety_evaluators(run: Any, example: Any) -> list[dict[str, Any]]:
    output = _outputs(run)
    packet = output.get("evidence_packet") or {}
    evidence_ids = {
        str(item.get("chunk_id") or item.get("document_id"))
        for item in packet.get("evidence_items", [])
        if isinstance(item, dict)
    }
    citation_ids = {
        str(item.get("chunk_id") or item.get("document_id"))
        for item in output.get("citations", [])
        if isinstance(item, dict)
    }
    answer = str(output.get("final_answer") or "")
    no_match_safe = True
    if packet.get("retrieval_status") == "no_match":
        no_match_safe = (
            not citation_ids
            and not re.search(r"《[^》]+》|第[零一二三四五六七八九十百千万0-9]+条", answer)
            and "未检索到" in answer
        )
    forbidden = [str(item) for item in _reference(example).get("forbidden_terms", [])]
    latest_safe = not any(item and item in answer for item in forbidden)
    claim_ids = {
        str(evidence_id)
        for claim in output.get("counsel_claims", [])
        if isinstance(claim, dict)
        for evidence_id in claim.get("evidence_chunk_ids", [])
    }
    mapping_valid = claim_ids <= evidence_ids
    safety = citation_ids <= evidence_ids and no_match_safe and latest_safe and mapping_valid
    return [
        _feedback("citation_grounding", citation_ids <= evidence_ids),
        _feedback("no_match_safety", no_match_safe),
        _feedback("latest_fact_priority", latest_safe),
        _feedback("unsupported_claim_control", mapping_valid),
        _feedback("quality_gate_passed", safety),
    ]


class FactAssessment(BaseModel):
    fact_key: str
    status: Literal["preserved", "omitted", "distorted", "invented", "contradicted"]
    comment: str = Field(default="", max_length=240)


class FactualJudgeResult(BaseModel):
    facts: list[FactAssessment] = Field(default_factory=list)
    factual_fidelity: float = Field(ge=1, le=5)
    fabrication_control: float = Field(ge=1, le=5)
    uncertainty_preservation: float = Field(ge=1, le=5)
    context_consistency: float = Field(ge=1, le=5)
    comment: str = Field(default="", max_length=500)


class ReviewerJudgeResult(BaseModel):
    detected_error_labels: list[str] = Field(default_factory=list)
    missed_error_labels: list[str] = Field(default_factory=list)
    false_positive_labels: list[str] = Field(default_factory=list)
    instruction_specificity: float = Field(ge=1, le=5)
    decision_quality: float = Field(ge=1, le=5)
    comment: str = Field(default="", max_length=500)


class AnswerJudgeResult(BaseModel):
    covered_issue_ids: list[str] = Field(default_factory=list)
    missing_issue_ids: list[str] = Field(default_factory=list)
    issue_coverage: float = Field(ge=1, le=5)
    answer_relevance: float = Field(ge=1, le=5)
    actionability: float = Field(ge=1, le=5)
    risk_calibration: float = Field(ge=1, le=5)
    completeness: float = Field(ge=1, le=5)
    clarity: float = Field(ge=1, le=5)
    overclaiming_control: float = Field(ge=1, le=5)
    follow_up_quality: float = Field(ge=1, le=5)
    comment: str = Field(default="", max_length=500)


SUITE_SCHEMAS: dict[str, type[BaseModel]] = {
    "factual-fidelity": FactualJudgeResult,
    "reviewer-effectiveness": ReviewerJudgeResult,
    "answer-quality": AnswerJudgeResult,
}


class AgentQualityJudge:
    """One structured Judge request per example, with no tools or hidden reasoning."""

    def __init__(self, suite: str, settings: Settings | None = None) -> None:
        if suite not in SUITE_SCHEMAS:
            raise ValueError(f"不支持的 Agent 质量评测套件：{suite}")
        self.suite = suite
        self.schema = SUITE_SCHEMAS[suite]
        settings = settings or get_settings()
        key = settings.langsmith_evaluator_api_key or settings.deepseek_api_key
        if not key:
            raise RuntimeError("未配置 Judge API Key")
        self.model_name = settings.langsmith_evaluator_model or settings.deepseek_model
        self.model = ChatOpenAI(
            model=self.model_name,
            api_key=key,
            base_url=settings.langsmith_evaluator_base_url or settings.deepseek_base_url,
            temperature=0,
            streaming=False,
            timeout=settings.llm_request_timeout_seconds,
            max_retries=0,
            extra_body={"thinking": {"type": "disabled"}},
        ).bind(response_format={"type": "json_object"})

    async def __call__(self, run: Any, example: Any) -> list[dict[str, Any]]:
        payload = {
            "inputs": getattr(example, "inputs", {}) or {},
            "reference": getattr(example, "outputs", {}) or {},
            "actual": _outputs(run),
        }
        prompt = (
            "你是 LawStation 的独立质量评审员。只能依据输入、参考约束、"
            "EvidencePacket 和最终回答判断，不得凭自身法律知识补充法条。"
            "不要输出隐藏推理，只输出符合 JSON Schema 的 JSON。"
            f"评测套件={self.suite}；Prompt版本={JUDGE_PROMPT_VERSION}；"
            f"JSON Schema={json.dumps(self.schema.model_json_schema(), ensure_ascii=False)}"
        )
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                response = await self.model.ainvoke([
                    ("system", prompt),
                    ("human", json.dumps(payload, ensure_ascii=False, default=str)),
                ])
                raw = response.content if isinstance(response.content, str) else ""
                result = self.schema.model_validate(json.loads(raw))
                return self._feedback(result, _reference(example))
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                if attempt == 1:
                    raise
        raise RuntimeError("Judge 未返回有效 JSON") from last_error

    def _feedback(
        self,
        result: BaseModel,
        reference: dict[str, Any],
    ) -> list[dict[str, Any]]:
        data = result.model_dump()
        comment = str(data.pop("comment", ""))
        details = {
            key: data.pop(key)
            for key in list(data)
            if key in {"facts", "detected_error_labels", "missed_error_labels", "false_positive_labels", "covered_issue_ids", "missing_issue_ids"}
        }
        suffix = json.dumps(details, ensure_ascii=False)[:1000] if details else ""
        feedback = [
            _feedback(f"judge_{key}", float(value) / 5, f"{comment} {suffix}".strip())
            for key, value in data.items()
            if isinstance(value, (int, float))
        ]
        if self.suite == "reviewer-effectiveness":
            # Error-category recall is calculated from the same structured Judge
            # response.  This preserves the one-Judge-call-per-example contract.
            expected = {
                str(item) for item in reference.get("injected_error_labels", []) if item
            }
            detected = {
                str(item)
                for item in details.get("detected_error_labels", [])
                if item
            }
            category_recall = (
                len(expected & detected) / len(expected) if expected else None
            )
            feedback.append(
                _feedback("error_category_recall", category_recall, suffix)
            )
        return feedback


SUITE_DETERMINISTIC = {
    "factual-fidelity": factual_fidelity_evaluators,
    "reviewer-effectiveness": reviewer_effectiveness_evaluators,
    "answer-quality": answer_quality_safety_evaluators,
}
