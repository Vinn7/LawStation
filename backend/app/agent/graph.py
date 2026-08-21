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

from backend.app.agent.middleware import InvocationModelLimitMiddleware, ToolAuditMiddleware
from backend.app.agent.registry import MCPToolRegistry
from backend.app.agent.schemas import (
    AgentError,
    CaseAnalysis,
    Citation,
    CounselDraft,
    EvidenceItem,
    EvidencePacket,
    ResearchTask,
    ReviewResult,
    UnresolvedIssue,
)
from backend.app.agent.state import AgentInvocationContext, LegalConsultationState
from backend.app.core.config import Settings
from backend.app.core.logging import audit, summary

SchemaT = TypeVar("SchemaT", bound=BaseModel)

ANALYST_PROMPT = """你是法律咨询的案情分析与调度 Agent。只做问题分类、事实整理、争议点拆分和研究规划。
当前用户最新消息与历史对话、摘要或 memory_context 冲突时，必须采用当前用户最新明确陈述的事实，
不得让历史记忆覆盖本轮修正。
不要编造法条，也不要输出内部推理过程。返回严格 JSON，字段必须符合以下结构：
request_type(casual_chat|legal_consultation|insufficient_information), case_summary, jurisdiction,
legal_domain, key_facts[], missing_facts[], legal_issues[], research_tasks[{issue_id,query,purpose}],
risk_level(low|medium|high), next_action(direct_answer|ask_clarification|research), direct_answer,
clarification_questions[]。普通闲聊填写 direct_answer；关键事实不足时给出简洁澄清问题。"""

RESEARCH_PROMPT = """你是法律研究 Agent，也是唯一可以调用法律检索工具的角色。针对每个 research task，
先使用 search_laws 获取候选；需要确认具体条号时使用 get_law_article。所有法规必须来自工具真实返回结果。
最终只返回严格 JSON：research_tasks[], evidence_items[{document_id,supports_issue_ids,
verification_status}], unresolved_issues[{issue_id,description}], conflicts[], research_summary。
evidence_items 只能选择工具结果中真实存在的 document_id；找不到依据时返回空 evidence_items，
并在 unresolved_issues 说明，这属于正常检索结果，不得凭常识补造法条。"""

COUNSEL_PROMPT = """你是面向用户的法律顾问 Agent。依据案情分析和 EvidencePacket 形成法律意见。
当前用户最新消息与历史记忆冲突时，以最新消息和案情分析中的修正事实为准，不得沿用旧事实。
retrieval_status=matched 时仅引用证据包中的具体法律名称与条号。
retrieval_status=no_match 时仍要提供有帮助的一般性、条件化分析和行动建议，但 confidence 必须为 low，
不得输出具体法律名称、司法解释名称或条号，不得声称已经完成法规核验，并必须明确说明当前法规库
未检索到可引用法条。tool_unavailable/tool_error 时应明确说明检索服务状态，不得冒充 no_match。
区分已知事实、条件性推论、法律依据和行动建议。
返回严格 JSON：answer, claims[{claim,evidence_document_ids}], confidence(low|medium|high),
limitations[], follow_up_questions[]。answer 使用清晰 Markdown，包含结论、依据、分析、风险和建议。"""

REVIEW_PROMPT = """你是 Case Analyst 的复核阶段。检查草稿是否覆盖争议点、是否存在无证据法条、
结论与证据是否一致、是否把推测写成事实、是否自相矛盾，以及是否错误采用了与用户最新消息
冲突的历史记忆；如有则要求 revise_draft。retrieval_status=no_match 时，不得仅因
没有法条而要求重新检索；应检查回答是否采用低置信度条件化表达、是否披露未检索到可引用法条、
是否避免具体法律名称和条号。存在越界时选择 revise_draft，不选择 research_again。
只返回严格 JSON：approved,
unsupported_claims[], missing_issue_ids[], citation_errors[], contradictions[], revision_instruction,
next_action(finalize|research_again|revise_draft)。只有明确证据缺口才 research_again；表达或论证问题选 revise_draft。"""


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
    by_document: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        by_document.setdefault(str(candidate.get("document_id", "")), candidate)
    accepted: list[EvidenceItem] = []
    seen: set[str] = set()
    for selected in packet.evidence_items:
        source = by_document.get(selected.document_id)
        if source is None or selected.document_id in seen:
            continue
        seen.add(selected.document_id)
        accepted.append(EvidenceItem(
            document_id=selected.document_id,
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


def _citation_errors(draft: CounselDraft | None, packet: EvidencePacket | None) -> list[str]:
    if draft is None:
        return ["缺少法律意见草稿"]
    evidence = packet.evidence_items if packet else []
    known_ids = {item.document_id for item in evidence}
    errors = [
        f"论证引用了未知证据 {document_id}"
        for claim in draft.claims
        for document_id in claim.evidence_document_ids
        if document_id not in known_ids
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
    }


class LegalConsultationGraph:
    def __init__(self, model: Any, tools: list[BaseTool], registry: MCPToolRegistry, settings: Settings) -> None:
        self.model = model
        self.tools = tools
        self.registry = registry
        self.settings = settings
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
        graph.add_node("reviewer", self.reviewer)
        graph.add_node("finalize", self.finalize)
        graph.add_edge(START, "case_analyst")
        graph.add_conditional_edges("case_analyst", self.after_analysis, {"finish": "finalize", "research": "legal_researcher"})
        graph.add_edge("legal_researcher", "legal_counsel")
        graph.add_edge("legal_counsel", "reviewer")
        graph.add_conditional_edges(
            "reviewer",
            self.after_review,
            {"finish": "finalize", "research": "legal_researcher", "revise": "legal_counsel"},
        )
        graph.add_edge("finalize", END)
        return graph.compile(name="lawstation-three-agent-graph")

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
        except Exception as exc:
            audit("agent.node.failed", level=logging.ERROR, status="failed", agent=agent_name, error_type=type(exc).__name__, error=summary(str(exc)), duration_ms=int((time.perf_counter() - started) * 1000), **context.audit_fields)
            raise

    async def case_analyst(self, state: LegalConsultationState, runtime: Runtime[AgentInvocationContext]) -> dict[str, Any]:
        runtime.stream_writer({"event": "agent_status", "data": {"agent": "case_analyst", "status": "analyzing", "message": "正在分析案情"}})
        try:
            analysis = await self._invoke_json(runtime, "case_analyst", ANALYST_PROMPT, _payload(state), CaseAnalysis)
        except (ValueError, ValidationError, RuntimeError) as exc:
            question = _message_text(state["messages"][-1]) if state["messages"] else ""
            analysis = CaseAnalysis(request_type="legal_consultation", case_summary=question, legal_issues=[question], research_tasks=[ResearchTask(issue_id="issue-1", query=question, purpose="核验法律依据")], next_action="research")
            return {"case_analysis": analysis, "errors": [*state["errors"], AgentError(agent="case_analyst", message=str(exc))]}
        update: dict[str, Any] = {"case_analysis": analysis}
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
            async for part in self.research_agent.astream(
                {"messages": [SystemMessage(content=RESEARCH_PROMPT), HumanMessage(content=json.dumps(research_payload, ensure_ascii=False, default=str))]},
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
            accepted = _authoritative_evidence(raw_packet, candidates)
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
                "evidence_items": accepted,
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
            return {"evidence_packet": packet, "retry_count": state["retry_count"] + (1 if state["review_result"] else 0)}
        except Exception as exc:
            audit("agent.node.failed", level=logging.ERROR, status="failed", agent="legal_researcher", error_type=type(exc).__name__, error=summary(str(exc)), **context.audit_fields)
            issues = [UnresolvedIssue(issue_id=f"issue-{index + 1}", description=item) for index, item in enumerate(analysis.legal_issues if analysis else [])]
            packet = EvidencePacket(retrieval_status="tool_error", research_tasks=analysis.research_tasks if analysis else [], unresolved_issues=issues, research_summary="法律检索执行失败，未能完成法规核验。")
            return {"evidence_packet": packet, "errors": [*state["errors"], AgentError(agent="legal_researcher", message=str(exc))], "retry_count": state["retry_count"] + (1 if state["review_result"] else 0)}

    async def legal_counsel(self, state: LegalConsultationState, runtime: Runtime[AgentInvocationContext]) -> dict[str, Any]:
        evidence = state["evidence_packet"]
        no_match = bool(evidence and evidence.retrieval_status == "no_match")
        message = "未检索到可引用法条，将基于案情形成一般性分析" if no_match else "正在形成法律意见"
        runtime.stream_writer({"event": "agent_status", "data": {"agent": "legal_counsel", "status": "drafting", "message": message}})
        payload = _payload(state)
        payload["answer_mode"] = "general_analysis_without_citations" if no_match else "evidence_based"
        if state["review_result"] and state["review_result"].revision_instruction:
            payload["revision_instruction"] = state["review_result"].revision_instruction
        try:
            draft = await self._invoke_json(runtime, "legal_counsel", COUNSEL_PROMPT, payload, CounselDraft)
            if no_match:
                disclosure = "本轮法规检索正常完成，但当前法规库中未检索到可引用法条。以上属于一般性分析，不构成已经过法规核验的确定性法律结论。"
                answer = draft.answer if _no_match_disclosure_present(draft.answer) else f"{draft.answer.rstrip()}\n\n## 检索说明\n\n{disclosure}"
                draft = draft.model_copy(update={"answer": answer, "confidence": "low"})
            return {"counsel_draft": draft, "revision_count": state["revision_count"] + (1 if state["review_result"] else 0)}
        except Exception as exc:
            fallback_answer = _no_match_safe_answer(state) if no_match else (evidence.research_summary if evidence else "暂时无法形成完整法律意见。") + "\n\n当前回答生成失败，建议稍后重试或咨询专业律师。"
            fallback = CounselDraft(answer=fallback_answer, confidence="low", limitations=["回答生成或法规核验未完整完成"])
            return {"counsel_draft": fallback, "errors": [*state["errors"], AgentError(agent="legal_counsel", message=str(exc))]}

    async def reviewer(self, state: LegalConsultationState, runtime: Runtime[AgentInvocationContext]) -> dict[str, Any]:
        runtime.stream_writer({"event": "agent_status", "data": {"agent": "case_analyst", "status": "reviewing", "message": "正在核验回答"}})
        try:
            review = await self._invoke_json(runtime, "case_analyst_reviewer", REVIEW_PROMPT, _payload(state), ReviewResult)
        except Exception as exc:
            review = ReviewResult(approved=False, revision_instruction="自动复核未完成，最终回答应保留风险提示。", next_action="finalize")
            return {"review_result": review, "errors": [*state["errors"], AgentError(agent="case_analyst_reviewer", message=str(exc))]}
        deterministic_errors = _citation_errors(state["counsel_draft"], state["evidence_packet"])
        packet = state["evidence_packet"]
        if packet and packet.retrieval_status == "no_match":
            deterministic_errors.extend(_no_match_violations(state["counsel_draft"].answer if state["counsel_draft"] else ""))
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
        return {"review_result": review}

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
        for item in packet.evidence_items if packet and packet.retrieval_status == "matched" else []:
            if item.document_id in seen_documents:
                continue
            seen_documents.add(item.document_id)
            citations.append(Citation(document_id=item.document_id, law_name=item.law_name, article_number=item.article_number, quoted_excerpt=item.content[:240], data_version=item.data_version))
        runtime.stream_writer({"event": "agent_status", "data": {"agent": "case_analyst", "status": "completed", "message": "分析已完成"}})
        return {"final_answer": answer, "citations": citations}
