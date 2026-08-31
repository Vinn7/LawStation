import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import HumanMessage

from backend.app.agent.runtime import AgentRuntime
from backend.app.agent.skills import SkillRegistry
from backend.app.agent.state import AgentInvocationContext, AgentInvocationIdentity
from backend.app.core.config import Settings
from backend.app.evaluation.evaluators import (
    skill_policy_compliance,
    skill_selection_precision,
    skill_selection_recall,
)


def settings(tmp_path, **changes):
    values = {
        "deepseek_api_key": "test-key",
        "agent_skill_root": str(tmp_path),
        "agent_skills_enabled": True,
        "agent_skill_strict_validation": True,
        "agent_max_active_skills": 2,
    }
    values.update(changes)
    return Settings(_env_file=None, **values)


def write_skill(
    root,
    name,
    *,
    agents=("case_analyst",),
    tools=(),
    schema="CaseIntakeResult",
    body="只返回结构化结果。",
):
    path = root / name
    path.mkdir(parents=True)
    agent_lines = "\n".join(f"  - {item}" for item in agents)
    tool_lines = "[]" if not tools else "\n" + "\n".join(f"  - {item}" for item in tools)
    (path / "SKILL.md").write_text(
        f"""---
name: {name}
version: 1.0.0
description: {name} test skill
allowed_agents:
{agent_lines}
allowed_tools: {tool_lines}
output_schema: {schema}
---
{body}
""",
        encoding="utf-8",
    )


def test_registry_loads_summaries_and_progressively_discloses_body(tmp_path):
    write_skill(tmp_path, "case-intake", body="TRUSTED_FULL_INSTRUCTION")

    registry = SkillRegistry(settings(tmp_path))

    assert registry.status().status == "ready"
    assert registry.status().skill_ids == ("case-intake",)
    assert "TRUSTED_FULL_INSTRUCTION" not in registry.catalog_prompt()
    assert "TRUSTED_FULL_INSTRUCTION" in registry.prompt_for(
        ["case-intake"], "case_analyst"
    )


def test_registry_rejects_unknown_ids_limits_composition_and_filters_roles(tmp_path):
    write_skill(tmp_path, "case-intake")
    write_skill(
        tmp_path,
        "evidence-audit",
        agents=("legal_counsel", "reviewer"),
        schema="EvidenceAuditResult",
    )
    write_skill(
        tmp_path,
        "procedure-roadmap",
        agents=("legal_researcher", "legal_counsel"),
        tools=("search_laws", "get_law_article"),
        schema="ProcedureRoadmapResult",
    )
    registry = SkillRegistry(settings(tmp_path, agent_max_active_skills=2))

    selected = registry.resolve(
        ["unknown", "procedure-roadmap", "evidence-audit", "case-intake"]
    )

    assert [item.summary.skill_id for item in selected] == [
        "case-intake",
        "evidence-audit",
    ]
    assert registry.resolve(["case-intake", "evidence-audit"], "reviewer")[0].summary.skill_id == "evidence-audit"


def test_registry_validates_output_schema_and_drops_unselected_output(tmp_path):
    write_skill(
        tmp_path,
        "evidence-audit",
        agents=("legal_counsel",),
        schema="EvidenceAuditResult",
    )
    registry = SkillRegistry(settings(tmp_path))

    outputs = registry.validate_outputs(
        {
            "evidence-audit": {"evidence_strength": "mixed", "missing_evidence": ["合同"]},
            "forged-skill": {"value": "ignored"},
        },
        ["evidence-audit"],
        "legal_counsel",
    )

    assert set(outputs) == {"evidence-audit"}
    assert outputs["evidence-audit"]["evidence_strength"] == "mixed"
    assert outputs["evidence-audit"]["missing_evidence"] == ["合同"]


def test_registry_strict_mode_rejects_unauthorized_tool(tmp_path):
    write_skill(tmp_path, "case-intake", tools=("delete_memory",))

    with pytest.raises(ValueError, match="未授权工具"):
        SkillRegistry(settings(tmp_path))


def test_disabled_registry_does_not_require_skill_files(tmp_path):
    registry = SkillRegistry(
        settings(tmp_path / "missing", agent_skills_enabled=False)
    )

    assert registry.status().status == "disabled"
    assert registry.resolve(["case-intake"]) == []
    assert "requested_skill_ids必须返回空数组" in registry.catalog_prompt()


@pytest.mark.asyncio
async def test_model_suggested_skill_is_validated_executed_and_exposed(tmp_path):
    write_skill(tmp_path, "case-intake")
    skill_registry = SkillRegistry(settings(tmp_path))
    analysis = {
        "request_type": "casual_chat",
        "case_summary": "复杂事实整理",
        "next_action": "direct_answer",
        "direct_answer": "已完成初步整理。",
        "requested_skill_ids": ["case-intake", "forged-skill"],
    }
    intake = {
        "parties": ["用户", "对方"],
        "legal_relationships": ["合同关系"],
        "key_facts": ["用户陈述存在争议"],
        "timeline": [],
        "amounts": [],
        "goals": ["解决争议"],
        "conflicts": [],
        "missing_information": ["合同文本"],
    }
    model = GenericFakeChatModel(
        messages=iter([
            json.dumps(analysis, ensure_ascii=False),
            json.dumps(intake, ensure_ascii=False),
        ])
    )
    mcp_registry = SimpleNamespace(
        get_tools=AsyncMock(return_value=[]),
        status=lambda: SimpleNamespace(version=0),
        close=AsyncMock(),
    )
    runtime = AgentRuntime(
        mcp_registry,
        SimpleNamespace(get_chat_model=lambda: model),
        settings(tmp_path),
        skill_registry=skill_registry,
    )
    context = AgentInvocationContext(
        AgentInvocationIdentity("request", "tenant", "user", "conversation")
    )

    events = [
        event
        async for event in runtime.stream(
            context, [HumanMessage(content="请帮我整理复杂案情")], ""
        )
    ]

    skill_events = [item["data"] for item in events if item["event"] == "skill_status"]
    assert [item["status"] for item in skill_events] == ["selected", "running", "completed"]
    assert context.evaluation_output["active_skills"][0]["skill_id"] == "case-intake"
    assert "forged-skill" not in {
        item["skill_id"] for item in context.evaluation_output["active_skills"]
    }
    assert context.evaluation_output["skill_outputs"]["case-intake"]["parties"] == [
        "用户",
        "对方",
    ]
    assert events[-1] == {"event": "agent_final", "data": "已完成初步整理。"}


@pytest.mark.asyncio
async def test_shared_runtime_keeps_concurrent_skill_state_request_scoped(tmp_path):
    write_skill(
        tmp_path,
        "evidence-audit",
        agents=("legal_counsel", "reviewer"),
        schema="EvidenceAuditResult",
    )
    registry = SkillRegistry(settings(tmp_path))
    model = GenericFakeChatModel(messages=iter(()))
    mcp_registry = SimpleNamespace(
        get_tools=AsyncMock(return_value=[]),
        status=lambda: SimpleNamespace(version=0),
        close=AsyncMock(),
    )
    runtime = AgentRuntime(
        mcp_registry,
        SimpleNamespace(get_chat_model=lambda: model),
        settings(tmp_path),
        skill_registry=registry,
    )
    context_a = AgentInvocationContext(
        AgentInvocationIdentity("request-a", "tenant", "user-a", "conversation-a"),
        evaluation_case_analysis={
            "request_type": "casual_chat",
            "next_action": "direct_answer",
            "direct_answer": "A",
            "requested_skill_ids": ["evidence-audit"],
        },
    )
    context_b = AgentInvocationContext(
        AgentInvocationIdentity("request-b", "tenant", "user-b", "conversation-b"),
        evaluation_case_analysis={
            "request_type": "casual_chat",
            "next_action": "direct_answer",
            "direct_answer": "B",
            "requested_skill_ids": [],
        },
    )

    async def consume(context, content):
        return [
            item
            async for item in runtime.stream(context, [HumanMessage(content=content)], "")
        ]

    await asyncio.gather(consume(context_a, "A"), consume(context_b, "B"))

    assert [item["skill_id"] for item in context_a.active_skills] == ["evidence-audit"]
    assert context_b.active_skills == []
    assert context_a.evaluation_output["active_skills"] != context_b.evaluation_output["active_skills"]


def test_skill_evaluators_measure_selection_and_reject_forged_outputs():
    example = SimpleNamespace(
        outputs={"expected_skill_ids": ["case-intake", "evidence-audit"]}
    )
    run = SimpleNamespace(
        outputs={
            "active_skills": [
                {"skill_id": "case-intake"},
                {"skill_id": "procedure-roadmap"},
            ],
            "skill_outputs": {"case-intake": {"parties": []}},
            "tool_trajectory": [],
        }
    )

    assert skill_selection_precision(run, example)["score"] == 0.5
    assert skill_selection_recall(run, example)["score"] == 0.5
    assert skill_policy_compliance(run, example)["score"] == 1.0

    run.outputs["skill_outputs"]["forged-skill"] = {"unsafe": True}
    assert skill_policy_compliance(run, example)["score"] == 0.0


def test_skill_routing_dataset_has_balanced_frozen_cases():
    dataset_path = Path(__file__).parents[1] / "evals/datasets/lawstation-skills-v1.jsonl"
    rows = [json.loads(line) for line in dataset_path.read_text("utf-8").splitlines()]

    assert len(rows) == 24
    categories = {row["metadata"]["category"] for row in rows}
    assert categories == {
        "skill_positive",
        "skill_negative",
        "skill_combination",
        "skill_unauthorized",
        "skill_disabled_baseline",
    }
    for skill_id in (
        "case-intake",
        "evidence-audit",
        "procedure-roadmap",
        "document-readiness",
    ):
        assert sum(row["metadata"]["skill_id"] == skill_id for row in rows) == 6
