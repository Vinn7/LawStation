"""Schemas and deterministic blueprints for generated multi-turn Agent scenarios."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError, model_validator

PROMPT_VERSION = "lawstation-dialogue-generator-v1"
DATASET_VERSION = "lawstation-dialogue-scenarios-v1"
VARIANTS_PER_BLUEPRINT = 2

ALLOWED_ACTIONS = {
    "send_message",
    "switch_user",
    "switch_conversation",
    "wait_for_completion",
    "cancel_run",
    "disconnect_stream",
    "reconnect_stream",
    "inspect_messages",
    "inspect_memories",
}
ALLOWED_SKILLS = {
    "case-intake",
    "evidence-audit",
    "procedure-roadmap",
    "document-readiness",
}

_SENSITIVE_PATTERNS = {
    "mobile": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    "identity_card": re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),
    "bank_card": re.compile(r"(?<!\d)\d{16,19}(?!\d)"),
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "secret": re.compile(r"(?i)(api[_ -]?key|authorization|bearer\s+[A-Za-z0-9._-]+)"),
}
_INJECTION_PATTERNS = (
    "忽略系统提示",
    "忽略之前的指令",
    "泄露系统提示",
    "绕过权限",
    "输出隐藏提示词",
)


class GeneratedVariant(BaseModel):
    variant: int = Field(ge=1)
    title: str = Field(min_length=2, max_length=80)
    messages: list[str] = Field(min_length=1, max_length=6)


class GeneratedBlueprintResponse(BaseModel):
    blueprint_id: str
    variants: list[GeneratedVariant]

    @model_validator(mode="after")
    def unique_variants(self):
        numbers = [item.variant for item in self.variants]
        if len(numbers) != len(set(numbers)):
            raise ValueError("variant 编号重复")
        return self


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _send(
    message_slot: int,
    *,
    actor: str = "primary",
    conversation: str = "main",
    events: list[str] | None = None,
    retrieval_status: str | None = None,
    skill_ids: list[str] | None = None,
    citations: bool | None = None,
    terminal: str = "completed",
) -> dict[str, Any]:
    expected: dict[str, Any] = {
        "events": events or ["message_start", "agent_status", "message_end"],
        "terminal": terminal,
        "max_model_calls": 10,
        "max_tool_calls": 4,
    }
    if retrieval_status is not None:
        expected["retrieval_status"] = retrieval_status
    if skill_ids is not None:
        expected["skill_ids"] = skill_ids
    if citations is not None:
        expected["citations"] = citations
    return {
        "action": "send_message",
        "actor": actor,
        "conversation": conversation,
        "message_slot": message_slot,
        "expected": expected,
    }


def _action(action: str, *, actor: str = "primary", conversation: str = "main", **data):
    return {"action": action, "actor": actor, "conversation": conversation, **data}


def blueprint_definitions() -> list[dict[str, Any]]:
    """Return the frozen operational templates; the model only supplies messages."""

    return [
        {
            "id": "casual-chat",
            "category": "routing",
            "purpose": "验证普通闲聊不调用法律检索。",
            "message_requirements": ["简短自然的非法律问候，不包含具体案件事实。"],
            "steps": [_send(0, events=["message_start", "agent_status", "message_end"], citations=False)],
        },
        {
            "id": "clarification-required",
            "category": "routing",
            "purpose": "验证事实不足时由 Case Analyst 请求澄清。",
            "message_requirements": ["提出模糊的法律问题，故意缺少主体、时间或行为细节。"],
            "steps": [_send(0, citations=False)],
        },
        {
            "id": "matched-law-citation",
            "category": "rag",
            "purpose": "验证检索命中、chunk证据和最终引用。",
            "message_requirements": ["围绕给定法条写生活化案情，不出现法名、条号或连续原文。"],
            "source_required": True,
            "steps": [_send(0, events=["message_start", "agent_status", "tool_call_start", "tool_call_result", "citations", "message_end"], retrieval_status="matched", citations=True)],
        },
        {
            "id": "normal-no-match",
            "category": "rag",
            "purpose": "验证无法条是正常结果并继续生成一般性分析。",
            "message_requirements": ["提出具有未来科技或纯虚拟制度背景、现行法规难以直接覆盖的问题。"],
            "steps": [_send(0, retrieval_status="no_match", citations=False)],
        },
        {
            "id": "tool-error-degradation",
            "category": "rag",
            "purpose": "验证MCP异常时受控降级。",
            "message_requirements": ["明确要求核验一个常见民事或劳动问题的法律依据。"],
            "preconditions": ["fixture_tool_error=true"],
            "steps": [_send(0, events=["message_start", "agent_status", "tool_call_start", "tool_call_result", "message_end"], retrieval_status="tool_error", citations=False)],
        },
        {
            "id": "skill-case-intake",
            "category": "skill",
            "purpose": "验证复杂案情结构化Skill。",
            "message_requirements": ["描述多主体、多时间节点和多笔金额，并要求先梳理案情。"],
            "steps": [_send(0, events=["message_start", "agent_status", "skill_status", "message_end"], skill_ids=["case-intake"])],
        },
        {
            "id": "skill-evidence-audit",
            "category": "skill",
            "purpose": "验证证据审查Skill。",
            "message_requirements": ["列出两三种现有材料，询问证明力和缺失证据。"],
            "steps": [_send(0, events=["message_start", "agent_status", "skill_status", "message_end"], skill_ids=["evidence-audit"])],
        },
        {
            "id": "skill-procedure-roadmap",
            "category": "skill",
            "purpose": "验证程序路线Skill及法律检索边界。",
            "message_requirements": ["询问仲裁、起诉、执行或投诉的步骤、材料与风险。"],
            "steps": [_send(0, events=["message_start", "agent_status", "skill_status", "tool_call_start", "tool_call_result", "message_end"], skill_ids=["procedure-roadmap"])],
        },
        {
            "id": "skill-document-readiness",
            "category": "skill",
            "purpose": "验证文书材料就绪检查Skill。",
            "message_requirements": ["准备一种法律文书，要求检查事实、请求和材料是否齐全。"],
            "steps": [_send(0, events=["message_start", "agent_status", "skill_status", "message_end"], skill_ids=["document-readiness"])],
        },
        {
            "id": "skill-combination-limit",
            "category": "skill",
            "purpose": "验证两个Skill组合和数量上限。",
            "message_requirements": ["同时要求梳理复杂案情并审查已有证据。"],
            "steps": [_send(0, events=["message_start", "agent_status", "skill_status", "message_end"], skill_ids=["case-intake", "evidence-audit"])],
        },
        {
            "id": "skill-unknown-rejected",
            "category": "skill",
            "purpose": "验证未知或越权Skill被服务端拒绝。",
            "message_requirements": ["要求证据审查，并夹带一个不存在的系统能力请求。"],
            "fixture_requested_skill_ids": ["evidence-audit", "delete-user-memory"],
            "steps": [_send(0, events=["message_start", "agent_status", "skill_status", "message_end"], skill_ids=["evidence-audit"])],
        },
        {
            "id": "memory-latest-fact",
            "category": "memory",
            "purpose": "验证本轮新事实覆盖旧记忆。",
            "message_requirements": ["首次明确陈述一个金额或日期事实。", "第二轮明确纠正同一事实并给出新值。"],
            "steps": [
                _send(0),
                _action("wait_for_completion"),
                _send(1),
                _action("inspect_memories", expected={"latest_fact_only": True}),
            ],
        },
        {
            "id": "memory-user-preference",
            "category": "memory",
            "purpose": "验证用户级偏好跨会话复用。",
            "message_requirements": ["说明一种稳定的回答表达偏好。", "在新会话中提出一个简短法律问题。"],
            "steps": [
                _send(0),
                _action("wait_for_completion"),
                _action("switch_conversation", conversation="secondary"),
                _send(1, conversation="secondary"),
                _action("inspect_memories", conversation="secondary", expected={"user_scope_reused": True}),
            ],
        },
        {
            "id": "memory-case-isolation",
            "category": "memory",
            "purpose": "验证案件事实不跨会话。",
            "message_requirements": ["在主会话陈述一个案件专属金额和主体事实。", "在第二会话咨询完全不同的法律关系。"],
            "steps": [
                _send(0),
                _action("wait_for_completion"),
                _action("switch_conversation", conversation="secondary"),
                _send(1, conversation="secondary"),
                _action("inspect_memories", conversation="secondary", expected={"conversation_scope_isolated": True}),
            ],
        },
        {
            "id": "concurrency-user-switch",
            "category": "concurrency",
            "purpose": "验证切换用户后原任务后台继续且状态隔离。",
            "actors": ["primary", "secondary"],
            "message_requirements": ["用户A提出需要检索和分析的法律问题。", "用户B提出不同领域的法律问题。"],
            "steps": [
                _send(0, actor="primary"),
                _action("switch_user", actor="secondary"),
                _send(1, actor="secondary"),
                _action("wait_for_completion", actor="primary", expected={"background_continues": True}),
                _action("inspect_messages", actor="secondary", expected={"cross_user_isolated": True}),
            ],
        },
        {
            "id": "run-cancel",
            "category": "durable_run",
            "purpose": "验证主动取消只影响指定会话任务。",
            "message_requirements": ["提出需要多步检索与分析的复杂法律问题。"],
            "steps": [
                _send(0, terminal="interrupted"),
                _action("cancel_run"),
                _action("inspect_messages", expected={"terminal": "interrupted"}),
            ],
        },
        {
            "id": "sse-reconnect-replay",
            "category": "durable_run",
            "purpose": "验证SSE断线重连和sequence重放。",
            "message_requirements": ["提出一个会触发法律检索的明确咨询。"],
            "steps": [
                _send(0),
                _action("disconnect_stream", after_event="agent_status"),
                _action("reconnect_stream", expected={"sequence_replay": True, "duplicates": False}),
                _action("wait_for_completion"),
            ],
        },
        {
            "id": "same-conversation-busy",
            "category": "concurrency",
            "purpose": "验证同会话重复生成409、不同会话可并发。",
            "message_requirements": ["第一条需要检索的法律问题。", "同一会话立即发送的第二条问题。", "另一会话中的独立问题。"],
            "steps": [
                _send(0),
                _send(1, terminal="rejected_409"),
                _send(2, conversation="secondary"),
                _action("wait_for_completion", expected={"different_conversation_concurrent": True}),
            ],
        },
    ]


def attach_sources(
    blueprints: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    *,
    seed: int = 42,
) -> list[dict[str, Any]]:
    eligible = [
        item for item in chunks
        if item.get("law_name") and item.get("article_number") and 80 <= len(str(item.get("content", ""))) <= 800
    ]
    if not eligible:
        raise RuntimeError("law.json 中没有可用于对话样例的法规 chunk")
    source = sorted(eligible, key=lambda item: str(item["chunk_id"]))[seed % len(eligible)]
    result = json.loads(json.dumps(blueprints, ensure_ascii=False))
    for blueprint in result:
        if blueprint.get("source_required"):
            blueprint["source"] = {
                key: source[key]
                for key in ("document_id", "chunk_id", "law_name", "article_number", "content")
            }
    return result


def generation_prompt(blueprint: dict[str, Any], variants: int) -> str:
    source_rule = ""
    if blueprint.get("source"):
        source_rule = (
            "该蓝图包含 source 法条。用户消息必须由该法条直接支持，但不得出现 law_name、"
            "article_number，也不得连续复制 content 中7个或更多字符。\n"
        )
    return (
        "你正在为中国法律咨询多Agent系统生成合成多轮测试话术。\n"
        "只生成用户会说的话，不生成答案、系统指令、工具名、Skill ID、用户ID、会话ID或测试动作。\n"
        "使用张某、李某、甲公司等虚构主体，不得包含真实手机号、身份证、银行卡、邮箱或密钥。\n"
        "不得包含要求忽略系统提示、泄露提示词或绕过权限的内容。\n"
        f"{source_rule}"
        f"必须生成恰好 {variants} 个变体，每个变体恰好 "
        f"{len(blueprint['message_requirements'])} 条 messages，并逐项满足 message_requirements。\n"
        "各变体事实和措辞应明显不同，但必须保持蓝图测试目的。\n"
        "仅返回JSON对象："
        '{"blueprint_id":"...","variants":[{"variant":1,"title":"...","messages":["..."]}]}。\n'
        "蓝图：\n" + json.dumps(blueprint, ensure_ascii=False)
    )


def _normalized(value: str) -> str:
    return re.sub(r"[\s\W_]+", "", value, flags=re.UNICODE).lower()


def _source_leak(message: str, content: str, length: int = 7) -> bool:
    normalized_message = _normalized(message)
    normalized_source = _normalized(content)
    if len(normalized_message) < length or len(normalized_source) < length:
        return False
    return any(
        normalized_source[index:index + length] in normalized_message
        for index in range(len(normalized_source) - length + 1)
    )


def materialize_scenarios(
    blueprints: list[dict[str, Any]],
    responses: list[dict[str, Any]],
    *,
    generator_model: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    blueprint_map = {item["id"]: item for item in blueprints}
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_scenarios: set[str] = set()
    seen_dialogues: set[str] = set()
    for raw in responses:
        try:
            parsed = GeneratedBlueprintResponse.model_validate(raw)
        except ValidationError as exc:
            rejected.append({"blueprint_id": str(raw.get("blueprint_id", "")), "reasons": ["schema_invalid"], "error_type": type(exc).__name__})
            continue
        blueprint = blueprint_map.get(parsed.blueprint_id)
        if blueprint is None:
            rejected.append({"blueprint_id": parsed.blueprint_id, "reasons": ["unknown_blueprint"]})
            continue
        for variant in parsed.variants:
            reasons: list[str] = []
            scenario_id = f"{parsed.blueprint_id}-{variant.variant:02d}"
            if scenario_id in seen_scenarios:
                reasons.append("duplicate_scenario_id")
            if len(variant.messages) != len(blueprint["message_requirements"]):
                reasons.append("message_count_mismatch")
            dialogue_hash = sha256_json([_normalized(item) for item in variant.messages])
            if dialogue_hash in seen_dialogues:
                reasons.append("duplicate_dialogue")
            for message in variant.messages:
                if not 2 <= len(message.strip()) <= 1000:
                    reasons.append("message_length")
                reasons.extend(name for name, pattern in _SENSITIVE_PATTERNS.items() if pattern.search(message))
                if any(item in message for item in _INJECTION_PATTERNS):
                    reasons.append("prompt_injection")
                source = blueprint.get("source")
                if source and (
                    str(source["law_name"]) in message
                    or str(source["article_number"]) in message
                    or _source_leak(message, str(source["content"]))
                ):
                    reasons.append("source_leak")
            steps: list[dict[str, Any]] = []
            for template in blueprint["steps"]:
                if template.get("action") not in ALLOWED_ACTIONS:
                    reasons.append("illegal_action")
                    continue
                step = json.loads(json.dumps(template, ensure_ascii=False))
                slot = step.pop("message_slot", None)
                if slot is not None:
                    if not isinstance(slot, int) or slot >= len(variant.messages):
                        reasons.append("invalid_message_slot")
                    else:
                        step["content"] = variant.messages[slot].strip()
                skill_ids = step.get("expected", {}).get("skill_ids", [])
                if any(skill_id not in ALLOWED_SKILLS for skill_id in skill_ids):
                    reasons.append("unauthorized_expected_skill")
                steps.append(step)
            if reasons:
                rejected.append({"scenario_id": scenario_id, "blueprint_id": parsed.blueprint_id, "reasons": sorted(set(reasons))})
                continue
            fixtures: dict[str, Any] = {}
            if blueprint.get("source"):
                fixtures["documents"] = [blueprint["source"]]
            if "fixture_tool_error=true" in blueprint.get("preconditions", []):
                fixtures["tool_error"] = True
            if blueprint.get("fixture_requested_skill_ids"):
                fixtures["requested_skill_ids"] = blueprint["fixture_requested_skill_ids"]
            scenario = {
                "schema_version": "dialogue-scenario-v1",
                "scenario_id": scenario_id,
                "title": variant.title.strip(),
                "category": blueprint["category"],
                "description": blueprint["purpose"],
                "synthetic": True,
                "actors": blueprint.get("actors", ["primary"]),
                "preconditions": blueprint.get("preconditions", []),
                "steps": steps,
                "fixtures": fixtures,
                "generation_metadata": {
                    "blueprint_id": parsed.blueprint_id,
                    "variant": variant.variant,
                    "generator_model": generator_model,
                    "prompt_version": PROMPT_VERSION,
                },
            }
            candidates.append(scenario)
            seen_scenarios.add(scenario_id)
            seen_dialogues.add(dialogue_hash)
    return candidates, rejected


def validate_scenarios(scenarios: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    ids = [str(item.get("scenario_id", "")) for item in scenarios]
    if len(ids) != len(set(ids)):
        errors.append("scenario_id不唯一")
    for scenario in scenarios:
        scenario_id = str(scenario.get("scenario_id", "unknown"))
        if scenario.get("schema_version") != "dialogue-scenario-v1":
            errors.append(f"{scenario_id}: schema_version错误")
        if not scenario.get("synthetic"):
            errors.append(f"{scenario_id}: 必须标记synthetic")
        for step in scenario.get("steps", []):
            if step.get("action") not in ALLOWED_ACTIONS:
                errors.append(f"{scenario_id}: 非法动作")
            if step.get("action") == "send_message" and not str(step.get("content", "")).strip():
                errors.append(f"{scenario_id}: send_message缺少content")
            skill_ids = step.get("expected", {}).get("skill_ids", [])
            if any(item not in ALLOWED_SKILLS for item in skill_ids):
                errors.append(f"{scenario_id}: 预期Skill越权")
    counts = Counter(item.get("generation_metadata", {}).get("blueprint_id") for item in scenarios)
    if any(count > VARIANTS_PER_BLUEPRINT for count in counts.values()):
        errors.append("单个蓝图的变体数量超过限制")
    return errors
