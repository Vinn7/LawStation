import asyncio
import json
import logging
import re
import time
from typing import Any, TypeVar

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from pydantic import BaseModel, ValidationError
from sqlalchemy import or_, select

from backend.app.agent.middleware import InvocationModelLimitMiddleware, ToolAuditMiddleware
from backend.app.agent.registry import MCPToolRegistry
from backend.app.agent.schemas import (
    AgentError,
    CaseAnalysis,
    Citation,
    CounselDraft,
    EvidenceItem,
    EvidencePacket,
    EvidenceSelectionResult,
    ResearchTask,
    ReviewResult,
    UnresolvedIssue,
)
from backend.app.agent.skills import CaseIntakeResult, SkillRegistry
from backend.app.agent.state import AgentInvocationContext, LegalConsultationState
from backend.app.core.config import Settings
from backend.app.core.logging import audit, summary
from backend.app.db.models import UserMemory
from backend.app.db.session import SessionLocal

SchemaT = TypeVar("SchemaT", bound=BaseModel)

ANALYST_PROMPT = """你是法律咨询的案情分析与调度 Agent。只做问题分类、事实整理、争议点拆分和研究规划。
当前用户最新消息与历史对话、摘要或 memory_context 冲突时，必须采用当前用户最新明确陈述的事实，
不得让历史记忆覆盖本轮修正。
不要编造法条，也不要输出内部推理过程。返回严格 JSON，字段必须符合以下结构：
request_type(casual_chat|legal_consultation|insufficient_information), case_summary, jurisdiction,
legal_domain, key_facts[], missing_facts[], legal_issues[], research_tasks[{issue_id,query,purpose}],
risk_level(low|medium|high), next_action(direct_answer|ask_clarification|research), direct_answer,
clarification_questions[], current_fact_overrides[{canonical_key,new_value,old_value,
replaced_memory_id,confidence}]。只有当前消息明确修正历史记忆时才填写override。普通闲聊填写
direct_answer；关键事实不足时给出简洁澄清问题。根据服务端提供的Skill目录填写
requested_skill_ids[]；只选择与本轮任务直接相关的ID，不得生成目录外ID。"""

RESEARCH_PROMPT = """你是法律研究 Agent，也是唯一可以调用法律检索工具的角色。针对每个 research task，
    先使用 search_laws 获取候选；需要确认具体条号时使用 get_law_article。所有法规必须来自工具真实返回结果。
    最终只返回严格 JSON：research_tasks[], evidence_items[{document_id,chunk_id,supports_issue_ids,
    verification_status}], accepted_chunk_ids[], rejected_candidates[{chunk_id,reason}],
    unresolved_issues[{issue_id,description}], conflicts[], research_summary。
    evidence_items 必须使用工具结果中真实存在的 chunk_id；document_id 仅表示原始法条，不能代替 chunk_id。
    找不到依据时返回空 evidence_items，
并在 unresolved_issues 说明，这属于正常检索结果，不得凭常识补造法条。"""

COUNSEL_PROMPT = """你是面向用户的法律顾问 Agent。依据案情分析和 EvidencePacket 形成法律意见。
当前用户最新消息与历史记忆冲突时，以最新消息和案情分析中的修正事实为准，不得沿用旧事实。
retrieval_status=matched 时仅引用证据包中的具体法律名称与条号。
retrieval_status=no_match 时仍要提供有帮助的一般性、条件化分析和行动建议，但 confidence 必须为 low，
不得输出具体法律名称、司法解释名称或条号，不得声称已经完成法规核验，并必须明确说明当前法规库
未检索到可引用法条。tool_unavailable/tool_error 时应明确说明检索服务状态，不得冒充 no_match。
区分已知事实、条件性推论、法律依据和行动建议。
    返回严格 JSON：answer, claims[{claim,evidence_chunk_ids}], confidence(low|medium|high),
limitations[], follow_up_questions[], skill_outputs{}。只为服务端提供的 active_skills 输出对应
skill_outputs；answer 使用清晰 Markdown，包含结论、依据、分析、风险和建议。"""

REVIEW_PROMPT = """你是 Case Analyst 的复核阶段。检查草稿是否覆盖争议点、是否存在无证据法条、
结论与证据是否一致、是否把推测写成事实、是否自相矛盾，以及是否错误采用了与用户最新消息
冲突的历史记忆；如有则要求 revise_draft。retrieval_status=no_match 时，不得仅因
没有法条而要求重新检索；应检查回答是否采用低置信度条件化表达、是否披露未检索到可引用法条、
是否避免具体法律名称和条号。存在越界时选择 revise_draft，不选择 research_again。
只返回严格 JSON：approved,
unsupported_claims[], missing_issue_ids[], citation_errors[], contradictions[], revision_instruction,
next_action(finalize|research_again|revise_draft)。只有明确证据缺口才 research_again；表达或论证问题选 revise_draft。"""

EVIDENCE_SELECTOR_PROMPT = """你是受限证据选择器，不允许调用任何工具。输入只包含本轮已检索到的
候选法条和争议点。逐个候选决定接受或拒绝：只有直接支持争议点的候选才能进入 accepted_chunk_ids；
其余必须进入 rejected_candidates 并给出简短原因。不得生成输入中不存在的 chunk_id。仅输出 JSON。"""

SKILL_STATUS_MESSAGES = {
    "case-intake": "正在结构化案情",
    "evidence-audit": "正在审查证据准备情况",
    "procedure-roadmap": "正在整理程序路线",
    "document-readiness": "正在检查文书材料完整性",
}


def _message_text(message: BaseMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    parts: list[str] = []
    for item in message.content if isinstance(message.content, list) else []:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
            parts.append(str(item.get("text", "")))
    return "".join(parts)


def _extract_json(text: str) -> Any:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1) if fenced else text
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end < start:
        raise ValueError("模型未返回 JSON 对象")
    return json.loads(candidate[start : end + 1])


def _json_values(value: Any) -> list[Any]:
    """Expand MCP content blocks and JSON strings without trusting model metadata."""
    values = [value]
    if isinstance(value, str):
        candidate = value.strip()
        if candidate.startswith(("{", "[")):
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                return values
            values.extend(_json_values(parsed))
    elif isinstance(value, dict):
        for nested in value.values():
            values.extend(_json_values(nested))
    elif isinstance(value, list):
        for nested in value:
            values.extend(_json_values(nested))
    return values


def _tool_documents(message: ToolMessage) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for value in _json_values(message.content):
        if isinstance(value, dict) and value.get("document_id"):
            documents.append(value)
    return documents


def _authoritative_evidence(packet: EvidencePacket, candidates: list[dict[str, Any]]) -> list[EvidenceItem]:
    by_chunk: dict[str, dict[str, Any]] = {}
    by_document: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        document_id = str(candidate.get("document_id", ""))
        chunk_id = str(candidate.get("chunk_id", ""))
        if chunk_id:
            by_chunk.setdefault(chunk_id, candidate)
        if document_id:
            by_document.setdefault(document_id, []).append(candidate)
    accepted: list[EvidenceItem] = []
    seen: set[str] = set()
    for selected in packet.evidence_items:
        source = by_chunk.get(selected.chunk_id) if selected.chunk_id else None
        if source is None and not selected.chunk_id:
            legacy_matches = by_document.get(selected.document_id, [])
            if len(legacy_matches) == 1:
                source = legacy_matches[0]
        if source is None:
            continue
        source_chunk_id = str(source.get("chunk_id") or source.get("document_id") or "")
        if not source_chunk_id or source_chunk_id in seen:
            continue
        seen.add(source_chunk_id)
        accepted.append(EvidenceItem(
            document_id=str(source.get("document_id", "")),
            chunk_id=source_chunk_id,
            law_name=str(source.get("law_name", "")),
            article_number=str(source.get("article_number", "")),
            content=str(source.get("content", "")),
            supports_issue_ids=[str(item) for item in selected.supports_issue_ids],
            retrieval_sources=[str(item) for item in source.get("retrieval_sources", [])],
            verification_status=selected.verification_status,
            data_version=str(source.get("data_version", "")),
        ))
    return accepted


def _no_match_disclosure_present(answer: str) -> bool:
    return "未检索到" in answer and ("可引用法条" in answer or "可直接引用" in answer)


def _no_match_violations(answer: str) -> list[str]:
    violations = []
    if re.search(r"《[^》]+》|第[零一二三四五六七八九十百千万0-9]+条", answer):
        violations.append("无法条模式包含未经检索核验的法律名称或条号")
    if not _no_match_disclosure_present(answer):
        violations.append("无法条模式缺少法规库未检索到可引用法条的说明")
    return violations


def _no_match_safe_answer(state: LegalConsultationState) -> str:
    analysis = state.get("case_analysis")
    summary_text = analysis.case_summary if analysis and analysis.case_summary else "你目前描述的情况"
    return (
        "## 初步判断\n\n"
        f"根据现有信息，关于“{summary_text}”，目前只能作一般性、条件化分析。最终处理方式仍会受到"
        "事实经过、时间节点、双方约定和现有证据的影响。\n\n"
        "## 建议重点确认\n\n"
        "- 梳理关键时间、双方约定以及已经进行的沟通。\n"
        "- 保存合同、付款记录、聊天记录、通知和其他能够证明事实的材料。\n"
        "- 在采取诉讼、仲裁或其他正式措施前，结合完整材料咨询专业律师。\n\n"
        "## 检索说明\n\n"
        "本轮法规检索正常完成，但当前法规库中未检索到可引用法条。以上属于一般性分析，"
        "不构成已经过法规核验的确定性法律结论。"
    )


def _validate_fact_overrides(items, context: AgentInvocationContext):
    """Treat model-provided memory IDs as untrusted ownership hints."""
    requested_ids = {item.replaced_memory_id for item in items if item.replaced_memory_id}
    valid_ids: set[str] = set()
    if requested_ids:
        identity = context.identity
        with SessionLocal() as db:
            valid_ids = set(db.scalars(select(UserMemory.id).where(
                UserMemory.id.in_(requested_ids),
                UserMemory.tenant_id == identity.tenant_id,
                UserMemory.user_id == identity.user_id,
                UserMemory.status == "active",
                or_(
                    UserMemory.scope == "user",
                    UserMemory.conversation_id == identity.conversation_id,
                ),
            )))
    accepted = [
        item for item in items
        if not item.replaced_memory_id or item.replaced_memory_id in valid_ids
    ]
    rejected_count = len(items) - len(accepted)
    if rejected_count:
        audit(
            "agent.fact_override.rejected",
            level=logging.WARNING,
            status="rejected",
            rejected_count=rejected_count,
            **context.audit_fields,
        )
    return accepted


def _fact_boundary_errors(state: LegalConsultationState, answer: str) -> list[str]:
    """Reject deterministic reuse of an explicitly superseded scalar fact."""
    errors: list[str] = []
    for override in state.get("current_fact_overrides", []):
        old_value = str(override.get("old_value") or "").strip()
        new_value = str(override.get("new_value") or "").strip()
        if old_value and old_value != new_value and old_value in answer:
            errors.append(
                f"回答仍使用已被本轮消息替换的事实 {override.get('canonical_key', '')}"
            )
    return errors


def _citation_errors(draft: CounselDraft | None, packet: EvidencePacket | None) -> list[str]:
    if draft is None:
        return ["缺少法律意见草稿"]
    evidence = packet.evidence_items if packet else []
    known_chunk_ids = {item.chunk_id for item in evidence if item.chunk_id}
    document_counts: dict[str, int] = {}
    for item in evidence:
        document_counts[item.document_id] = document_counts.get(item.document_id, 0) + 1
    legacy_document_ids = {
        document_id for document_id, count in document_counts.items() if count == 1
    }
    errors = [
        f"论证引用了未知证据 {evidence_id}"
        for claim in draft.claims
        for evidence_id in [*claim.evidence_chunk_ids, *claim.evidence_document_ids]
        if evidence_id not in known_chunk_ids and evidence_id not in legacy_document_ids
    ]
    for law_name, article_number in re.findall(
        r"《([^》]+)》\s*(第[零一二三四五六七八九十百千万0-9]+条)", draft.answer
    ):
        supported = any(
            article_number == item.article_number
            and (law_name in item.law_name or item.law_name in law_name)
            for item in evidence
        )
        if not supported:
            errors.append(f"《{law_name}》{article_number} 不在本轮证据包中")
    return errors


def _payload(state: LegalConsultationState) -> dict[str, Any]:
    return {
        "conversation": [
            {"role": message.type, "content": _message_text(message)}
            for message in state["messages"][-12:]
        ],
        "memory_context": state["memory_context"],
        "case_analysis": state["case_analysis"].model_dump() if state["case_analysis"] else None,
        "evidence_packet": state["evidence_packet"].model_dump() if state["evidence_packet"] else None,
        "counsel_draft": state["counsel_draft"].model_dump() if state["counsel_draft"] else None,
        "review_result": state["review_result"].model_dump() if state["review_result"] else None,
        "current_fact_overrides": state.get("current_fact_overrides", []),
        "active_skills": state.get("active_skills", []),
        "skill_outputs": state.get("skill_outputs", {}),
    }


class LegalConsultationGraph:
    def __init__(
        self,
        model: Any,
        tools: list[BaseTool],
        registry: MCPToolRegistry,
        skill_registry: SkillRegistry,
        settings: Settings,
        checkpointer: Any = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.registry = registry
        self.skill_registry = skill_registry
        self.settings = settings
        self.checkpointer = checkpointer
        self.research_agent = create_agent(
            model=model,
            tools=tools,
            context_schema=AgentInvocationContext,
            middleware=[
                InvocationModelLimitMiddleware(settings),
                ToolCallLimitMiddleware(run_limit=min(settings.agent_max_tool_calls, 2), exit_behavior="continue"),
                ModelCallLimitMiddleware(run_limit=settings.agent_max_model_calls, exit_behavior="end"),
                ToolAuditMiddleware(registry, settings),
            ],
            name="legal-research-agent",
        ) if tools else None
        self.compiled = self._compile()

    def _compile(self):
        graph = StateGraph(LegalConsultationState, context_schema=AgentInvocationContext)
        graph.add_node("case_analyst", self.case_analyst)
        graph.add_node("legal_researcher", self.legal_researcher)
        graph.add_node("legal_counsel", self.legal_counsel)
        graph.add_node("review_gate", self.review_gate)
        graph.add_node("reviewer", self.reviewer)
        graph.add_node("finalize", self.finalize)
        graph.add_edge(START, "case_analyst")
        graph.add_conditional_edges("case_analyst", self.after_analysis, {"finish": "finalize", "research": "legal_researcher"})
        graph.add_edge("legal_researcher", "legal_counsel")
        graph.add_edge("legal_counsel", "review_gate")
        graph.add_conditional_edges(
            "review_gate", self.after_review_gate, {"review": "reviewer", "finish": "finalize"}
        )
        graph.add_conditional_edges(
            "reviewer",
            self.after_review,
            {"finish": "finalize", "research": "legal_researcher", "revise": "legal_counsel"},
        )
        graph.add_edge("finalize", END)
        return graph.compile(
            name="lawstation-three-agent-graph",
            checkpointer=self.checkpointer,
        )

    async def _invoke_json(self, runtime: Runtime[AgentInvocationContext], agent_name: str, prompt: str, payload: dict[str, Any], schema: type[SchemaT]) -> SchemaT:
        context = runtime.context
        if context.metrics.model_call_count >= self.settings.agent_max_model_calls:
            raise RuntimeError("本轮模型调用次数已达到上限")
        context.metrics.model_call_count += 1
        started = time.perf_counter()
        audit("agent.node.started", status="started", agent=agent_name, **context.audit_fields)
        try:
            response = await self.model.ainvoke([
                SystemMessage(content=prompt),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False, default=str)),
            ])
            result = schema.model_validate(_extract_json(_message_text(response)))
            audit("agent.node.completed", status="success", agent=agent_name, duration_ms=int((time.perf_counter() - started) * 1000), **context.audit_fields)
            return result
        except Exception as exc:  # Central model boundary audits and re-raises provider errors.
            audit("agent.node.failed", level=logging.ERROR, status="failed", agent=agent_name, error_type=type(exc).__name__, error=summary(str(exc)), duration_ms=int((time.perf_counter() - started) * 1000), **context.audit_fields)
            raise

    @staticmethod
    def _active_skill_ids(state: LegalConsultationState) -> list[str]:
        return [str(item.get("skill_id")) for item in state.get("active_skills", [])]

    def _skill_event(
        self,
        runtime: Runtime[AgentInvocationContext],
        skill_id: str,
        status: str,
        message: str | None = None,
    ) -> None:
        runtime.stream_writer({
            "event": "skill_status",
            "data": {
                "skill_id": skill_id,
                "status": status,
                "message": message or SKILL_STATUS_MESSAGES.get(skill_id, "正在执行领域能力"),
            },
        })

    async def _activate_skills(
        self,
        state: LegalConsultationState,
        runtime: Runtime[AgentInvocationContext],
        analysis: CaseAnalysis,
    ) -> tuple[list[dict[str, str]], dict[str, dict[str, Any]]]:
        resolved = self.skill_registry.resolve(
            analysis.requested_skill_ids,
            audit_fields=runtime.context.audit_fields,
        )
        active = self.skill_registry.public_skills(resolved)
        runtime.context.active_skills = active
        outputs = dict(state.get("skill_outputs", {}))
        for item in resolved:
            skill_id = item.summary.skill_id
            audit(
                "skill.selection.accepted",
                status="selected",
                skill_id=skill_id,
                skill_version=item.summary.version,
                content_digest=item.summary.content_digest,
                **runtime.context.audit_fields,
            )
            self._skill_event(runtime, skill_id, "selected")
        if any(item.summary.skill_id == "case-intake" for item in resolved):
            self._skill_event(runtime, "case-intake", "running")
            audit(
                "skill.execution.started",
                status="started",
                skill_id="case-intake",
                **runtime.context.audit_fields,
            )
            try:
                result = await self._invoke_json(
                    runtime,
                    "skill_case_intake",
                    self.skill_registry.prompt_for(["case-intake"], "case_analyst"),
                    _payload(state) | {"case_analysis": analysis.model_dump()},
                    CaseIntakeResult,
                )
                outputs["case-intake"] = result.model_dump()
                self._skill_event(runtime, "case-intake", "completed", "案情结构化完成")
                audit(
                    "skill.execution.completed",
                    status="success",
                    skill_id="case-intake",
                    **runtime.context.audit_fields,
                )
            except Exception as exc:  # noqa: BLE001 - optional skill is fail-open
                outputs.pop("case-intake", None)
                self._skill_event(runtime, "case-intake", "failed", "案情结构化未完成，继续基础分析")
                audit(
                    "skill.execution.failed",
                    level=logging.WARNING,
                    status="failed",
                    skill_id="case-intake",
                    error_type=type(exc).__name__,
                    error=summary(str(exc)),
                    **runtime.context.audit_fields,
                )
        return active, outputs

    async def case_analyst(self, state: LegalConsultationState, runtime: Runtime[AgentInvocationContext]) -> dict[str, Any]:
        runtime.stream_writer({"event": "agent_status", "data": {"agent": "case_analyst", "status": "analyzing", "message": "正在分析案情"}})
        if runtime.context.evaluation_case_analysis is not None:
            analysis = CaseAnalysis.model_validate(runtime.context.evaluation_case_analysis)
            validated_overrides = await asyncio.to_thread(
                _validate_fact_overrides, analysis.current_fact_overrides, runtime.context
            )
            analysis = analysis.model_copy(update={"current_fact_overrides": validated_overrides})
            active_skills, skill_outputs = await self._activate_skills(
                state, runtime, analysis
            )
            update: dict[str, Any] = {
                "case_analysis": analysis,
                "current_fact_overrides": [
                    item.model_dump() for item in analysis.current_fact_overrides
                ],
                "active_skills": active_skills,
                "skill_outputs": skill_outputs,
                "model_call_count": runtime.context.metrics.model_call_count,
            }
            if analysis.next_action == "direct_answer":
                update["final_answer"] = analysis.direct_answer
            elif analysis.next_action == "ask_clarification":
                questions = analysis.clarification_questions or analysis.missing_facts
                update["final_answer"] = "为了更准确地分析，请补充以下信息：\n\n" + "\n".join(
                    f"- {item}" for item in questions
                )
            return update
        try:
            analyst_prompt = f"{ANALYST_PROMPT}\n\n{self.skill_registry.catalog_prompt()}"
            analysis = await self._invoke_json(
                runtime, "case_analyst", analyst_prompt, _payload(state), CaseAnalysis
            )
        except (ValueError, ValidationError, RuntimeError) as exc:
            question = _message_text(state["messages"][-1]) if state["messages"] else ""
            analysis = CaseAnalysis(request_type="legal_consultation", case_summary=question, legal_issues=[question], research_tasks=[ResearchTask(issue_id="issue-1", query=question, purpose="核验法律依据")], next_action="research")
            return {
                "case_analysis": analysis,
                "errors": [*state["errors"], AgentError(agent="case_analyst", message=str(exc))],
                "model_call_count": runtime.context.metrics.model_call_count,
            }
        validated_overrides = await asyncio.to_thread(
            _validate_fact_overrides, analysis.current_fact_overrides, runtime.context
        )
        analysis = analysis.model_copy(update={"current_fact_overrides": validated_overrides})
        active_skills, skill_outputs = await self._activate_skills(state, runtime, analysis)
        update: dict[str, Any] = {
            "case_analysis": analysis,
            "current_fact_overrides": [
                item.model_dump() for item in analysis.current_fact_overrides
            ],
            "active_skills": active_skills,
            "skill_outputs": skill_outputs,
            "model_call_count": runtime.context.metrics.model_call_count,
        }
        if analysis.next_action == "direct_answer":
            update["final_answer"] = analysis.direct_answer or "您好，请告诉我需要咨询的法律问题。"
        elif analysis.next_action == "ask_clarification":
            questions = analysis.clarification_questions or analysis.missing_facts
            update["final_answer"] = "为了更准确地分析，请补充以下信息：\n\n" + "\n".join(f"- {item}" for item in questions)
        return update

    def after_analysis(self, state: LegalConsultationState) -> str:
        analysis = state["case_analysis"]
        return "research" if analysis and analysis.next_action == "research" else "finish"

    async def legal_researcher(self, state: LegalConsultationState, runtime: Runtime[AgentInvocationContext]) -> dict[str, Any]:
        runtime.stream_writer({"event": "agent_status", "data": {"agent": "legal_researcher", "status": "researching", "message": "正在检索法律依据"}})
        analysis = state["case_analysis"]
        if self.research_agent is None:
            issues = [UnresolvedIssue(issue_id=f"issue-{index + 1}", description=item) for index, item in enumerate(analysis.legal_issues if analysis else [])]
            return {"evidence_packet": EvidencePacket(retrieval_status="tool_unavailable", research_tasks=analysis.research_tasks if analysis else [], unresolved_issues=issues, research_summary="法律检索工具当前不可用。")}
        context = runtime.context
        started = time.perf_counter()
        audit("agent.node.started", status="started", agent="legal_researcher", **context.audit_fields)
        research_payload = _payload(state)
        if state["review_result"] and state["review_result"].revision_instruction:
            research_payload["supplemental_instruction"] = state["review_result"].revision_instruction
        try:
            final_message: BaseMessage | None = None
            candidates: list[dict[str, Any]] = []
            seen_tool_messages: set[str] = set()
            successful_tool_result = False
            failed_tool_result = False
            skill_prompt = self.skill_registry.prompt_for(
                self._active_skill_ids(state), "legal_researcher"
            )
            async for part in self.research_agent.astream(
                {"messages": [SystemMessage(content="\n\n".join(item for item in (RESEARCH_PROMPT, skill_prompt) if item)), HumanMessage(content=json.dumps(research_payload, ensure_ascii=False, default=str))]},
                context=context,
                stream_mode=["updates", "custom"],
                version="v2",
            ):
                if part.get("type") == "custom" and isinstance(part.get("data"), dict):
                    runtime.stream_writer(part["data"])
                elif part.get("type") == "updates" and isinstance(part.get("data"), dict):
                    for update in part["data"].values():
                        messages = update.get("messages", []) if isinstance(update, dict) else []
                        if messages:
                            final_message = messages[-1]
                        for message in messages:
                            if not isinstance(message, ToolMessage):
                                continue
                            message_id = str(message.tool_call_id)
                            if message_id in seen_tool_messages:
                                continue
                            seen_tool_messages.add(message_id)
                            if message.status == "error":
                                failed_tool_result = True
                            else:
                                successful_tool_result = True
                                candidates.extend(_tool_documents(message))
            if final_message is None:
                raise RuntimeError("法律研究 Agent 未返回结果")
            raw_packet = EvidencePacket.model_validate(_extract_json(_message_text(final_message)))
            if not raw_packet.evidence_items and raw_packet.accepted_chunk_ids:
                selected_ids = set(raw_packet.accepted_chunk_ids)
                raw_packet = raw_packet.model_copy(update={
                    "evidence_items": [
                        EvidenceItem(
                            document_id=str(item.get("document_id", "")),
                            chunk_id=str(item.get("chunk_id", "")),
                            supports_issue_ids=[task.issue_id for task in analysis.research_tasks],
                        )
                        for item in candidates
                        if str(item.get("chunk_id", "")) in selected_ids
                    ]
                })
            accepted = _authoritative_evidence(raw_packet, candidates)
            if candidates and not accepted and not raw_packet.rejected_candidates:
                selection = await self._invoke_json(
                    runtime,
                    "evidence_selector",
                    EVIDENCE_SELECTOR_PROMPT,
                    {
                        "legal_issues": analysis.legal_issues if analysis else [],
                        "candidates": [
                            {
                                "chunk_id": item.get("chunk_id"),
                                "law_name": item.get("law_name"),
                                "article_number": item.get("article_number"),
                                "content": item.get("content"),
                            }
                            for item in candidates
                        ],
                    },
                    EvidenceSelectionResult,
                )
                valid_ids = {str(item.get("chunk_id", "")) for item in candidates}
                accepted_ids = {
                    item for item in selection.accepted_chunk_ids if item in valid_ids
                }
                valid_rejections = [
                    item for item in selection.rejected_candidates if item.chunk_id in valid_ids
                ]
                selected_packet = raw_packet.model_copy(update={
                    "evidence_items": [
                        EvidenceItem(
                            document_id=str(item.get("document_id", "")),
                            chunk_id=str(item.get("chunk_id", "")),
                            supports_issue_ids=[task.issue_id for task in analysis.research_tasks],
                        )
                        for item in candidates
                        if str(item.get("chunk_id", "")) in accepted_ids
                    ],
                    "accepted_chunk_ids": list(accepted_ids),
                    "rejected_candidates": valid_rejections,
                })
                accepted = _authoritative_evidence(selected_packet, candidates)
                raw_packet = selected_packet
            if accepted:
                retrieval_status = "matched"
            elif failed_tool_result and not successful_tool_result:
                retrieval_status = "tool_error"
            else:
                retrieval_status = "no_match"
            unresolved = raw_packet.unresolved_issues
            if retrieval_status == "no_match" and not unresolved:
                unresolved = [UnresolvedIssue(issue_id="issue-1", description="当前法规库未检索到可直接引用的依据")]
            packet = raw_packet.model_copy(update={
                "retrieval_status": retrieval_status,
                "candidate_status": "matched" if candidates else "no_match",
                "evidence_status": (
                    "accepted"
                    if accepted
                    else "error"
                    if failed_tool_result and not successful_tool_result
                    else "rejected"
                ),
                "evidence_items": accepted,
                "accepted_chunk_ids": [item.chunk_id for item in accepted],
                "unresolved_issues": unresolved,
                "research_summary": raw_packet.research_summary or (
                    "检索正常完成，但未找到可引用法条。" if retrieval_status == "no_match" else raw_packet.research_summary
                ),
            })
            unique_candidates = {str(item.get("chunk_id") or item.get("document_id")) for item in candidates}
            audit(
                "agent.node.completed",
                status="success",
                agent="legal_researcher",
                duration_ms=int((time.perf_counter() - started) * 1000),
                retrieval_status=retrieval_status,
                candidate_count=len(unique_candidates),
                accepted_evidence_count=len(accepted),
                rejected_candidate_count=max(0, len(unique_candidates) - len(accepted)),
                evidence_count=len(accepted),
                no_match_reason=("候选法条未被研究 Agent 认定为可引用依据" if retrieval_status == "no_match" else ""),
                remaining_tool_calls=max(0, self.settings.agent_max_tool_calls - context.metrics.tool_call_count),
                remaining_model_calls=max(0, self.settings.agent_max_model_calls - context.metrics.model_call_count),
                **context.audit_fields,
            )
            return {
                "evidence_packet": packet,
                "retry_count": state["retry_count"] + (1 if state["review_result"] else 0),
                "model_call_count": context.metrics.model_call_count,
                "tool_call_count": context.metrics.tool_call_count,
                "tool_trajectory": list(context.metrics.tool_trajectory),
            }
        except Exception as exc:  # noqa: BLE001 - research failures become controlled evidence state
            audit("agent.node.failed", level=logging.ERROR, status="failed", agent="legal_researcher", error_type=type(exc).__name__, error=summary(str(exc)), **context.audit_fields)
            issues = [UnresolvedIssue(issue_id=f"issue-{index + 1}", description=item) for index, item in enumerate(analysis.legal_issues if analysis else [])]
            packet = EvidencePacket(retrieval_status="tool_error", research_tasks=analysis.research_tasks if analysis else [], unresolved_issues=issues, research_summary="法律检索执行失败，未能完成法规核验。")
            return {
                "evidence_packet": packet,
                "errors": [*state["errors"], AgentError(agent="legal_researcher", message=str(exc))],
                "retry_count": state["retry_count"] + (1 if state["review_result"] else 0),
                "model_call_count": runtime.context.metrics.model_call_count,
                "tool_call_count": runtime.context.metrics.tool_call_count,
                "tool_trajectory": list(runtime.context.metrics.tool_trajectory),
            }

    async def legal_counsel(self, state: LegalConsultationState, runtime: Runtime[AgentInvocationContext]) -> dict[str, Any]:
        evidence = state["evidence_packet"]
        no_match = bool(evidence and evidence.retrieval_status == "no_match")
        message = "未检索到可引用法条，将基于案情形成一般性分析" if no_match else "正在形成法律意见"
        runtime.stream_writer({"event": "agent_status", "data": {"agent": "legal_counsel", "status": "drafting", "message": message}})
        payload = _payload(state)
        payload["answer_mode"] = "general_analysis_without_citations" if no_match else "evidence_based"
        if state["review_result"] and state["review_result"].revision_instruction:
            payload["revision_instruction"] = state["review_result"].revision_instruction
        active_ids = self._active_skill_ids(state)
        counsel_skills = self.skill_registry.resolve(active_ids, "legal_counsel")
        for item in counsel_skills:
            self._skill_event(runtime, item.summary.skill_id, "running")
            audit(
                "skill.execution.started",
                status="started",
                skill_id=item.summary.skill_id,
                **runtime.context.audit_fields,
            )
        try:
            skill_prompt = self.skill_registry.prompt_for(active_ids, "legal_counsel")
            draft = await self._invoke_json(
                runtime,
                "legal_counsel",
                "\n\n".join(item for item in (COUNSEL_PROMPT, skill_prompt) if item),
                payload,
                CounselDraft,
            )
            try:
                validated_outputs = self.skill_registry.validate_outputs(
                    draft.skill_outputs, active_ids, "legal_counsel"
                )
            except ValidationError as exc:
                validated_outputs = {}
                audit(
                    "skill.execution.failed",
                    level=logging.WARNING,
                    status="failed",
                    skill_id="counsel_skill_outputs",
                    error_type=type(exc).__name__,
                    error=summary(str(exc)),
                    **runtime.context.audit_fields,
                )
            draft = draft.model_copy(update={"skill_outputs": validated_outputs})
            merged_outputs = {**state.get("skill_outputs", {}), **validated_outputs}
            for item in counsel_skills:
                skill_id = item.summary.skill_id
                completed = skill_id in validated_outputs
                self._skill_event(
                    runtime,
                    skill_id,
                    "completed" if completed else "failed",
                    "领域分析完成" if completed else "领域分析未完成，继续基础回答",
                )
                audit(
                    "skill.execution.completed" if completed else "skill.execution.failed",
                    level=logging.INFO if completed else logging.WARNING,
                    status="success" if completed else "failed",
                    skill_id=skill_id,
                    **runtime.context.audit_fields,
                )
            if no_match:
                disclosure = "本轮法规检索正常完成，但当前法规库中未检索到可引用法条。以上属于一般性分析，不构成已经过法规核验的确定性法律结论。"
                answer = draft.answer if _no_match_disclosure_present(draft.answer) else f"{draft.answer.rstrip()}\n\n## 检索说明\n\n{disclosure}"
                draft = draft.model_copy(update={"answer": answer, "confidence": "low"})
            return {
                "counsel_draft": draft,
                "skill_outputs": merged_outputs,
                "revision_count": state["revision_count"] + (1 if state["review_result"] else 0),
                "model_call_count": runtime.context.metrics.model_call_count,
            }
        except Exception as exc:  # noqa: BLE001 - counsel failures use a safe user-facing fallback
            for item in counsel_skills:
                self._skill_event(
                    runtime,
                    item.summary.skill_id,
                    "failed",
                    "领域分析未完成，继续安全兜底回答",
                )
            fallback_answer = _no_match_safe_answer(state) if no_match else (evidence.research_summary if evidence else "暂时无法形成完整法律意见。") + "\n\n当前回答生成失败，建议稍后重试或咨询专业律师。"
            fallback = CounselDraft(answer=fallback_answer, confidence="low", limitations=["回答生成或法规核验未完整完成"])
            return {
                "counsel_draft": fallback,
                "errors": [*state["errors"], AgentError(agent="legal_counsel", message=str(exc))],
                "model_call_count": runtime.context.metrics.model_call_count,
            }

    async def reviewer(self, state: LegalConsultationState, runtime: Runtime[AgentInvocationContext]) -> dict[str, Any]:
        runtime.stream_writer({"event": "agent_status", "data": {"agent": "case_analyst", "status": "reviewing", "message": "正在核验回答"}})
        try:
            skill_prompt = self.skill_registry.prompt_for(
                self._active_skill_ids(state), "reviewer"
            )
            review = await self._invoke_json(
                runtime,
                "case_analyst_reviewer",
                "\n\n".join(item for item in (REVIEW_PROMPT, skill_prompt) if item),
                _payload(state),
                ReviewResult,
            )
        except Exception as exc:  # noqa: BLE001 - review failures finalize with explicit limitations
            review = ReviewResult(approved=False, revision_instruction="自动复核未完成，最终回答应保留风险提示。", next_action="finalize")
            return {
                "review_result": review,
                "errors": [*state["errors"], AgentError(agent="case_analyst_reviewer", message=str(exc))],
                "model_call_count": runtime.context.metrics.model_call_count,
            }
        deterministic_errors = _citation_errors(state["counsel_draft"], state["evidence_packet"])
        packet = state["evidence_packet"]
        if packet and packet.retrieval_status == "no_match":
            deterministic_errors.extend(_no_match_violations(state["counsel_draft"].answer if state["counsel_draft"] else ""))
        deterministic_errors.extend(
            _fact_boundary_errors(
                state,
                state["counsel_draft"].answer if state["counsel_draft"] else "",
            )
        )
        if deterministic_errors:
            review.approved = False
            review.citation_errors = list(dict.fromkeys([*review.citation_errors, *deterministic_errors]))
            review.next_action = "revise_draft"
            instruction = "删除或改写所有不在 EvidencePacket 中的法条和证据引用。"
            review.revision_instruction = " ".join(
                item for item in (review.revision_instruction, instruction) if item
            )
        elif packet and packet.retrieval_status == "no_match":
            if review.unsupported_claims or review.contradictions:
                review.approved = False
                review.next_action = "revise_draft"
            else:
                review.approved = True
                review.next_action = "finalize"
        return {
            "review_result": review,
            "model_call_count": runtime.context.metrics.model_call_count,
        }

    async def review_gate(
        self, state: LegalConsultationState, runtime: Runtime[AgentInvocationContext]
    ) -> dict[str, Any]:
        analysis = state["case_analysis"]
        packet = state["evidence_packet"]
        draft = state["counsel_draft"]
        skip_reason = ""
        if self.settings.agent_review_mode == "always-llm":
            skip_reason = "评测配置要求始终执行模型复核"
        elif state["review_result"] is not None:
            skip_reason = "修订后的草稿必须再次复核"
        elif not analysis or analysis.risk_level != "low":
            skip_reason = "中高风险问题必须进行模型复核"
        elif not packet or packet.retrieval_status != "no_match":
            skip_reason = "存在法规依据或检索异常，必须进行模型复核"
        elif not draft or draft.confidence != "low":
            skip_reason = "无法条回答的置信度边界未满足"
        elif _no_match_violations(draft.answer):
            skip_reason = "无法条回答未通过确定性边界校验"
        elif any(claim.evidence_chunk_ids or claim.evidence_document_ids for claim in draft.claims):
            skip_reason = "无法条回答包含证据引用"

        if skip_reason:
            runtime.context.metrics.review_mode = "llm"
            audit(
                "agent.review.selected",
                status="selected",
                review_mode="llm",
                review_skip_reason=skip_reason,
                **runtime.context.audit_fields,
            )
            return {}
        audit(
            "agent.review.skipped",
            status="success",
            review_mode="deterministic",
            review_skip_reason="低风险无法条回答已通过确定性边界校验",
            **runtime.context.audit_fields,
        )
        runtime.context.metrics.review_mode = "deterministic"
        return {
            "review_result": ReviewResult(approved=True, next_action="finalize")
        }

    @staticmethod
    def after_review_gate(state: LegalConsultationState) -> str:
        review = state["review_result"]
        return "finish" if review and review.approved else "review"

    def after_review(self, state: LegalConsultationState) -> str:
        review = state["review_result"]
        packet = state["evidence_packet"]
        if not review or review.approved or review.next_action == "finalize":
            return "finish"
        if packet and packet.retrieval_status in {"no_match", "tool_unavailable", "tool_error"}:
            return "revise" if review.next_action == "revise_draft" and state["revision_count"] < 1 else "finish"
        if review.next_action == "research_again" and state["retry_count"] < 1:
            return "research"
        if review.next_action == "revise_draft" and state["revision_count"] < 1:
            return "revise"
        return "finish"

    async def finalize(self, state: LegalConsultationState, runtime: Runtime[AgentInvocationContext]) -> dict[str, Any]:
        answer = state["final_answer"] or (state["counsel_draft"].answer if state["counsel_draft"] else "")
        packet = state["evidence_packet"]
        review = state["review_result"]
        if packet and packet.retrieval_status == "no_match":
            violations = _no_match_violations(answer)
            if not review or not review.approved or violations:
                answer = _no_match_safe_answer(state)
        elif review and not review.approved and (
            review.unsupported_claims
            or review.missing_issue_ids
            or review.citation_errors
            or review.contradictions
        ):
            evidence_lines = [
                f"- 《{item.law_name}》{item.article_number}：{item.content[:180]}"
                for item in (packet.evidence_items if packet else [])
            ]
            answer = (
                "本轮自动复核未通过，因此未输出可能缺乏依据的确定性法律结论。\n\n"
                "### 已核验到的材料\n"
                + ("\n".join(evidence_lines) if evidence_lines else "- 暂未取得足够的法规依据。")
                + "\n\n建议补充案情后重新咨询，或由专业律师结合完整材料审查。"
            )
        citations: list[Citation] = []
        seen_documents: set[str] = set()
        referenced_ids = {
            evidence_id
            for claim in (state["counsel_draft"].claims if state["counsel_draft"] else [])
            for evidence_id in [*claim.evidence_chunk_ids, *claim.evidence_document_ids]
        }
        document_counts: dict[str, int] = {}
        for item in packet.evidence_items if packet else []:
            document_counts[item.document_id] = document_counts.get(item.document_id, 0) + 1
        for item in packet.evidence_items if packet and packet.retrieval_status == "matched" else []:
            legacy_match = item.document_id in referenced_ids and document_counts[item.document_id] == 1
            if item.chunk_id not in referenced_ids and not legacy_match:
                continue
            evidence_key = item.chunk_id or item.document_id
            if evidence_key in seen_documents:
                continue
            seen_documents.add(evidence_key)
            citations.append(Citation(document_id=item.document_id, chunk_id=item.chunk_id, law_name=item.law_name, article_number=item.article_number, quoted_excerpt=item.content[:240], data_version=item.data_version))
        runtime.stream_writer({"event": "agent_status", "data": {"agent": "case_analyst", "status": "completed", "message": "分析已完成"}})
        return {
            "final_answer": answer,
            "citations": citations,
            "model_call_count": runtime.context.metrics.model_call_count,
            "tool_call_count": runtime.context.metrics.tool_call_count,
            "tool_trajectory": list(runtime.context.metrics.tool_trajectory),
        }
