"""LawStation 法律咨询的 LangGraph 编排与节点实现。

本模块包含两层容易混淆的“Agent”抽象：

1. 外层 ``StateGraph`` 负责节点顺序、条件路由、状态合并和 Checkpoint；它本身
   不是大模型，也不会在 ``compile()`` 时发起模型请求。
2. ``legal_researcher`` 节点内部使用 LangChain ``create_agent``。该 Agent 才会
   让 DeepSeek 自主产生 tool_calls、经 MCP 执行工具、接收 ToolMessage，并继续
   调用模型形成研究结论。

主路径为 ``START -> case_analyst -> [finalize | legal_researcher] ->
legal_counsel -> review_gate -> [finalize | reviewer]``。Reviewer 最多把状态送回
Research 或 Counsel 各一次，最后由 ``finalize`` 执行确定性证据边界校验。

``LegalConsultationState`` 是节点间传递且可被 LangGraph Checkpoint 持久化的数据；
``AgentInvocationContext`` 则保存当前请求的身份、调用计数和审计对象，不属于跨节点
业务状态，也不得成为跨轮会话记忆。
"""

import asyncio
import json
import logging
import re
import time
from typing import Any, TypeVar

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware

# LangChain 的 Message 是模型上下文协议：SystemMessage 放系统约束，HumanMessage
# 放本轮结构化输入，ToolMessage 是工具执行后反馈给模型的结果；BaseMessage 用于
# 接收这些消息的共同父类型。ToolMessage 不等于最终用户可见回答。
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage

# MCP Adapter 发现的远程工具会被包装为 BaseTool，供 create_agent 统一调用。
from langchain_core.tools import BaseTool

# StateGraph/START/END 描述节点拓扑；Runtime 为节点提供请求级 context 和
# stream_writer，使节点能发送安全状态事件而无需接触 API/SSE 实现。
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

# Pydantic 不负责“相信模型”，而是在模型返回 JSON 后验证字段、枚举和嵌套结构。
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

# case_analyst 使用；不绑定工具。输出由 CaseAnalysis 校验并写入
# state.case_analysis，after_analysis 再根据 next_action 决定结束还是研究。
# Analyst 只能建议 requested_skill_ids，SkillRegistry 才拥有最终启用权。
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

# legal_researcher 内部的 LangChain Agent 使用；这是唯一绑定 MCP BaseTool 的角色。
# 最终 JSON 由 EvidencePacket 校验，随后代码再用真实 ToolMessage 回填权威元数据。
RESEARCH_PROMPT = """你是法律研究 Agent，也是唯一可以调用法律检索工具的角色。针对每个 research task，
    先使用 search_laws 获取候选；需要确认具体条号时使用 get_law_article。所有法规必须来自工具真实返回结果。
    最终只返回严格 JSON：research_tasks[], evidence_items[{document_id,chunk_id,supports_issue_ids,
    verification_status}], accepted_chunk_ids[], rejected_candidates[{chunk_id,reason}],
    unresolved_issues[{issue_id,description}], conflicts[], research_summary。
    evidence_items 必须使用工具结果中真实存在的 chunk_id；document_id 仅表示原始法条，不能代替 chunk_id。
    找不到依据时返回空 evidence_items，
并在 unresolved_issues 说明，这属于正常检索结果，不得凭常识补造法条。"""

# legal_counsel 使用；不调用工具。CounselDraft 只能引用 state.evidence_packet 中的
# chunk，生成的仍是待复核草稿，而不是已经写入 messages 表的最终回答。
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

# reviewer 使用；不调用工具。ReviewResult 决定 finalize、补充研究或修改草稿。
# 正常 no_match 只能修改越界表达，不能仅因证据为空重新检索。
REVIEW_PROMPT = """你是 Case Analyst 的复核阶段。检查草稿是否覆盖争议点、是否存在无证据法条、
结论与证据是否一致、是否把推测写成事实、是否自相矛盾，以及是否错误采用了与用户最新消息
冲突的历史记忆；如有则要求 revise_draft。retrieval_status=no_match 时，不得仅因
没有法条而要求重新检索；应检查回答是否采用低置信度条件化表达、是否披露未检索到可引用法条、
是否避免具体法律名称和条号。存在越界时选择 revise_draft，不选择 research_again。
只返回严格 JSON：approved,
unsupported_claims[], missing_issue_ids[], citation_errors[], contradictions[], revision_instruction,
next_action(finalize|research_again|revise_draft)。只有明确证据缺口才 research_again；表达或论证问题选 revise_draft。"""

# 当工具已有候选、Research 模型却既未接受也未拒绝时调用。它是一次无工具的
# 受限结构化调用，只能在已有 chunk_id 中选择，结果由 EvidenceSelectionResult 校验。
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
    """把 LangChain 的字符串或 Content Block 消息统一为可解析文本。

    OpenAI-compatible Provider 可能把 ``content`` 返回为纯字符串，也可能返回
    text/output_text 内容块。这里仅规范传输格式，不对文本真实性作任何判断。
    """
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
    """从模型文本中提取 JSON 对象，再交给调用方的 Pydantic Schema 校验。

    兼容模型用 Markdown 代码块包裹 JSON 的情况。成功 ``json.loads`` 只代表语法
    正确，字段合法性、证据 ID 和所有权仍必须由后续 Schema/服务端校验。
    """
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1) if fenced else text
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end < start:
        raise ValueError("模型未返回 JSON 对象")
    return json.loads(candidate[start : end + 1])


def _json_values(value: Any) -> list[Any]:
    """递归展开 MCP Content Block、JSON 字符串和嵌套容器。

    MCP Adapter 的结果常呈现为 ``ToolMessage -> content blocks -> JSON string ->
    object/list``。递归只解决封装层级问题，不会把其中任意对象自动认定为证据。
    """
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
    """从一个真实 ToolMessage 中收集带 document_id 的候选法规对象。"""
    documents: list[dict[str, Any]] = []
    for value in _json_values(message.content):
        if isinstance(value, dict) and value.get("document_id"):
            documents.append(value)
    return documents


def _authoritative_evidence(packet: EvidencePacket, candidates: list[dict[str, Any]]) -> list[EvidenceItem]:
    """将模型的证据选择映射回本轮 MCP 返回的权威候选。

    ``document_id`` 标识原始法条，``chunk_id`` 标识实际命中的切分片段。模型只能
    选择真实 chunk_id；正文、法名和条号全部从候选回填。为兼容旧输出，只有某个
    document_id 在本轮恰好对应一个 chunk 时才允许省略 chunk_id。伪造、歧义和
    重复 ID 都会被静默丢弃，不能进入 EvidencePacket。
    """
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
    """确定性检查 no_match 回答是否向用户披露未找到可引用法条。"""
    return "未检索到" in answer and ("可引用法条" in answer or "可直接引用" in answer)


def _no_match_violations(answer: str) -> list[str]:
    """用代码而非另一次 LLM 调用拦截无法条模式下的具体法名/条号幻觉。"""
    violations = []
    if re.search(r"《[^》]+》|第[零一二三四五六七八九十百千万0-9]+条", answer):
        violations.append("无法条模式包含未经检索核验的法律名称或条号")
    if not _no_match_disclosure_present(answer):
        violations.append("无法条模式缺少法规库未检索到可引用法条的说明")
    return violations


def _no_match_safe_answer(state: LegalConsultationState) -> str:
    """生成不会引用具体法规的保底回答；no_match 是成功空结果而非系统异常。"""
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
    """把模型给出的记忆 ID 当作不可信提示，并在同一 SQL 中验证所有权。

    只有当前 tenant/user 下的 active 记忆可被替换；会话级记忆还必须属于当前
    conversation。这样即使模型生成了其他 ID，也无法越权修改或推断记录。
    """
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
    """确定性拒绝回答继续使用本轮已经明确替换的旧金额、日期或其他标量事实。"""
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
    """验证草稿声明和文中的法条都能映射回本轮 EvidencePacket。"""
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
    """构造节点可见的有限上下文，避免直接暴露完整 Checkpoint 或服务端对象。"""
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
    """编译并持有可复用的三 Agent LangGraph。

    类实例可以由多个请求共享；每次 ``compiled.astream`` 接收的 State 与
    ``AgentInvocationContext`` 仍相互隔离。只有 Research 节点需要 LangChain
    create_agent，因为只有它允许模型自主选择 MCP 工具。
    """

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
        # LangChain create_agent 返回一个可 astream 的模型—工具循环：model 是共享
        # DeepSeek ChatOpenAI，tools 是 MCP Adapter 包装的 BaseTool，context_schema
        # 让中间件取得本轮身份和计数。中间件负责真实调用上限、超时与安全审计。
        # 构造阶段不会请求模型；没有工具时保留 None，由节点走 tool_unavailable。
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
        """声明节点拓扑并编译，不在此阶段执行任何模型或工具。"""

        # LegalConsultationState 会在节点返回 dict 时被 LangGraph 合并并写入
        # Checkpoint；context_schema 只提供运行期依赖，不成为可持久化 Graph State。
        graph = StateGraph(LegalConsultationState, context_schema=AgentInvocationContext)
        # Analyst：模型结构化分析，写 case_analysis/事实覆盖/Skill 选择。
        graph.add_node("case_analyst", self.case_analyst)
        # Research：内部 LangChain Agent 可调用 MCP，写 evidence_packet。
        graph.add_node("legal_researcher", self.legal_researcher)
        # Counsel：无工具模型调用，基于证据包写 counsel_draft。
        graph.add_node("legal_counsel", self.legal_counsel)
        # Gate：纯代码快速检查，决定是否需要额外 LLM Reviewer。
        graph.add_node("review_gate", self.review_gate)
        # Reviewer：结构化复核，可能要求一次补检索或一次改稿。
        graph.add_node("reviewer", self.reviewer)
        # Finalize：纯代码安全出口，生成 final_answer 与 citations。
        graph.add_node("finalize", self.finalize)
        graph.add_edge(START, "case_analyst")
        # conditional_edges 先调用 after_analysis，再用返回的字符串查右侧映射。
        graph.add_conditional_edges("case_analyst", self.after_analysis, {"finish": "finalize", "research": "legal_researcher"})
        # 普通 edge 无条件执行后继节点。
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
        # compile 生成可复用 CompiledStateGraph。Checkpointer 在 super-step 边界保存
        # State，使重启后能从最近完成节点继续；它不会保存每个 token 或请求级对象。
        return graph.compile(
            name="lawstation-three-agent-graph",
            checkpointer=self.checkpointer,
        )

    async def _invoke_json(self, runtime: Runtime[AgentInvocationContext], agent_name: str, prompt: str, payload: dict[str, Any], schema: type[SchemaT]) -> SchemaT:
        """执行一次不带工具的 LangChain 模型调用并返回 Pydantic 对象。

        顺序为：检查/增加模型额度 -> 组装 System/Human Message -> ChatOpenAI
        ``ainvoke`` -> 提取 AIMessage 文本 -> JSON 解析 -> Schema 校验。``ainvoke``
        返回完整消息而非 SSE token；Provider/解析错误向上抛给节点决定降级策略。
        """
        context = runtime.context
        if context.metrics.model_call_count >= self.settings.agent_max_model_calls:
            raise RuntimeError("本轮模型调用次数已达到上限")
        context.metrics.model_call_count += 1
        started = time.perf_counter()
        audit("agent.node.started", status="started", agent=agent_name, **context.audit_fields)
        try:
            # ChatOpenAI 是 LangChain 对 DeepSeek OpenAI-compatible API 的包装；这里
            # 没有 bind_tools，因此模型只能返回文本，不能在该调用中执行 MCP。
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
        """通过 LangGraph custom stream 写入可公开状态，不暴露 Skill Prompt。"""

        # stream_writer 的字典随后由 AgentRuntime 转换为持久化 AgentRunEvent/SSE。
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
        """将 Analyst 建议的 Skill ID 解析为受信任、请求级的可选能力。

        这里不是向量或关键词查询：Analyst 已基于摘要目录给出 ID，Registry 再执行
        白名单、角色和数量校验。完整 SKILL.md 仅在选中后由 prompt_for 注入。
        """

        # 步骤 1：读取 Analyst 的 Skill 选择建议。
        # requested_skill_ids 来自 CaseAnalysis 的结构化模型输出，因此只能视为“不可信
        # 建议”；模型不能凭一个字符串直接取得本地 Skill 指令或额外工具权限。
        # 步骤 2：交给应用级 SkillRegistry 做确定性裁决。
        # resolve() 不调用模型，也不进行向量/关键词搜索；它只接受启动时已经扫描并校验
        # 的 Skill ID，按注册表固定顺序去重，并执行未知 ID、角色和最大数量限制。这里
        # 没有指定 agent_role，是因为本阶段先确定“本轮全局激活集合”；具体节点稍后在
        # prompt_for()/validate_outputs() 中还会再次按照 case_analyst/legal_counsel 等角色过滤。
        resolved = self.skill_registry.resolve(
            analysis.requested_skill_ids,
            audit_fields=runtime.context.audit_fields,
        )

        # 步骤 3：把内部 ResolvedSkill 转换成可以进入运行状态的公开元数据。
        # public_skills() 只保留 ID、版本、描述和内容摘要，不包含 SKILL.md 完整指令。
        # 这样 LangGraph Checkpoint、LangSmith metadata 和后续节点可以识别 Skill 版本，
        # 又不会让内部 Prompt 因状态持久化或事件输出而泄露。
        active = self.skill_registry.public_skills(resolved)

        # 步骤 4：将本轮激活结果写入请求级 Invocation Context。
        # runtime.context 属于当前 AgentRun，不是共享 Graph 的实例字段；AgentRuntime 会用
        # 这里的数据补充 Trace 和最终运行元数据。不同用户并发调用时各自持有独立 context。
        runtime.context.active_skills = active

        # 步骤 5：继承当前 Graph State 中已经存在的 Skill 输出。
        # 使用 dict() 创建浅拷贝，避免直接原地修改传入 State；节点返回 outputs 后，
        # LangGraph 才会把它合并进 state.skill_outputs 并在节点边界写入 Checkpoint。
        outputs = dict(state.get("skill_outputs", {}))

        # 步骤 6：逐个记录服务端最终接受的 Skill。
        # 这里只发送选择状态，不执行 Skill，也不把完整 Prompt 写入审计或 SSE。
        for item in resolved:
            # ResolvedSkill.summary 是启动时经过 Frontmatter、版本、权限和路径校验的摘要。
            skill_id = item.summary.skill_id

            # 本地 JSONL 审计记录 ID、版本和 digest，便于重现本轮使用了哪个 Skill 版本；
            # audit_fields 携带 request/tenant/user/conversation 关联信息，但不记录 Skill 正文。
            audit(
                "skill.selection.accepted",
                status="selected",
                skill_id=skill_id,
                skill_version=item.summary.version,
                content_digest=item.summary.content_digest,
                **runtime.context.audit_fields,
            )

            # custom stream 事件随后由 AgentRuntime/AgentRunManager 转换为可重放事件和 SSE。
            # 前端只能看到安全的 skill_id、status 和中文提示，不能读取 Skill 内部指令。
            self._skill_event(runtime, skill_id, "selected")

        # 步骤 7：判断是否需要立即执行 Analyst 专属的 case-intake Skill。
        # 其他 Skill（如 evidence-audit、procedure-roadmap、document-readiness）主要作为
        # 后续节点的约束或结构化输出协议，不在这里统一执行，以免绕过角色边界。
        if any(item.summary.skill_id == "case-intake" for item in resolved):
            # 步骤 7.1：先向事件流声明 case-intake 已进入运行状态。
            self._skill_event(runtime, "case-intake", "running")

            # 审计开始事件与完成/失败事件配对，用于统计 Skill 调用次数和定位失败阶段。
            audit(
                "skill.execution.started",
                status="started",
                skill_id="case-intake",
                **runtime.context.audit_fields,
            )
            try:
                # 步骤 7.2：按 case_analyst 角色加载 case-intake 的完整可信指令。
                # prompt_for() 会再次调用 Registry.resolve(..., agent_role="case_analyst")，
                # 未授权角色无法获得 Skill 正文。这一步体现 Progressive Disclosure：Analyst
                # 首次选择时只看到目录摘要，真正选中并通过权限校验后才加载完整 SKILL.md。
                # 步骤 7.3：组装本次 Skill 模型调用的结构化输入。
                # _payload(state) 包含当前问题、记忆快照和已有 Graph 结果；新的
                # case_analysis 由当前节点显式覆盖进去，保证 Skill 读取的是本轮最新分析。
                # 步骤 7.4：通过统一的无工具模型边界执行 Skill。
                # _invoke_json() 会检查模型调用额度、调用共享 ChatOpenAI.ainvoke()、提取
                # AIMessage JSON，并用 CaseIntakeResult 做 Pydantic 校验。它没有 bind_tools，
                # 所以 case-intake 不能调用 MCP，也不会进入 Research Agent 的工具循环。
                result = await self._invoke_json(
                    runtime,
                    "skill_case_intake",
                    self.skill_registry.prompt_for(["case-intake"], "case_analyst"),
                    _payload(state) | {"case_analysis": analysis.model_dump()},
                    CaseIntakeResult,
                )

                # 步骤 7.5：只有模型响应通过 JSON 与 Schema 校验后才写入输出集合。
                # model_dump() 将 Pydantic 对象转换为可由 LangGraph Checkpoint 序列化的字典；
                # 原始 AIMessage、模型内部字段和未经校验的文本都不会进入 Graph State。
                outputs["case-intake"] = result.model_dump()

                # 步骤 7.6：分别发送用户可见的安全完成状态和本地审计完成事件。
                self._skill_event(runtime, "case-intake", "completed", "案情结构化完成")
                audit(
                    "skill.execution.completed",
                    status="success",
                    skill_id="case-intake",
                    **runtime.context.audit_fields,
                )
            except Exception as exc:  # noqa: BLE001 - optional skill is fail-open
                # 步骤 7.7：可选 Skill 失败采用 fail-open。
                # 无论失败发生在模型请求、JSON 解析还是 Schema 校验，都先删除可能遗留的
                # case-intake 输出，避免半成品被后续 Counsel 或 Checkpoint 当成可信结果。
                outputs.pop("case-intake", None)

                # 告知前端该增强能力未完成，但基础三 Agent 链路会继续执行；这里不会把
                # Skill 异常升级成整轮咨询失败，也不会触发 MCP 工具重试。
                self._skill_event(runtime, "case-intake", "failed", "案情结构化未完成，继续基础分析")

                # 审计只记录异常类型和经过 summary() 截断/清洗的摘要，不上传完整模型响应、
                # Skill Prompt 或用户敏感正文。若属于越权 Skill/工具，Registry 会更早拒绝，
                # 不会进入这个普通执行失败分支。
                audit(
                    "skill.execution.failed",
                    level=logging.WARNING,
                    status="failed",
                    skill_id="case-intake",
                    error_type=type(exc).__name__,
                    error=summary(str(exc)),
                    **runtime.context.audit_fields,
                )

        # 步骤 8：把“激活的安全元数据”和“已验证的 Skill 输出”返回给 case_analyst。
        # 调用方会将二者写入 LegalConsultationState；启用了但无需立即执行的 Skill 可以只有
        # active 元数据而没有 outputs，这是正常状态，后续节点会按各自角色加载和使用它们。
        return active, outputs

    async def case_analyst(self, state: LegalConsultationState, runtime: Runtime[AgentInvocationContext]) -> dict[str, Any]:
        """分类请求、整理事实、提出研究任务，并建议运行时 Skill。

        输入：本轮 messages、memory_context 以及尚为空的中间状态。
        输出：case_analysis、有效事实覆盖、active_skills/skill_outputs；闲聊或需要
        澄清时还会直接写 final_answer。返回字典由 LangGraph 合并回 State。
        """

        # 步骤 1：先发布节点状态。stream_writer 产生 LangGraph custom event，随后
        # AgentRuntime 会补充 request/user/conversation 快照并交给 AgentRunEvent；
        # 它不会把 Prompt、内部推理或完整 Graph State 暴露给订阅者。
        runtime.stream_writer({"event": "agent_status", "data": {"agent": "case_analyst", "status": "analyzing", "message": "正在分析案情"}})

        # 步骤 2A（仅离线评测）：evaluation_case_analysis 是测试提供的固定分析结果，
        # 用于把“路由/后续节点测试”与真实 Analyst 模型波动分离。生产请求默认为
        # None；即使走固定输入，仍必须执行事实所有权和 Skill 白名单校验。
        if runtime.context.evaluation_case_analysis is not None:
            # 步骤 2A-1：先用 CaseAnalysis 校验测试对象，保证字段与真实模型输出一致。
            analysis = CaseAnalysis.model_validate(runtime.context.evaluation_case_analysis)
            # 步骤 2A-2：模型/Fixture 提供的 replaced_memory_id 不能直接信任。
            # to_thread 把同步 SQLAlchemy 查询放到工作线程，避免阻塞事件循环。
            validated_overrides = await asyncio.to_thread(
                _validate_fact_overrides, analysis.current_fact_overrides, runtime.context
            )
            # 步骤 2A-3：只保留通过 tenant/user/conversation 所有权校验的修正事实。
            analysis = analysis.model_copy(update={"current_fact_overrides": validated_overrides})
            # 步骤 2A-4：解析 Analyst 建议的 Skill，并按需执行 case-intake。
            active_skills, skill_outputs = await self._activate_skills(
                state, runtime, analysis
            )
            # 步骤 2A-5：构造节点增量。LangGraph 会把这些键合并进现有 State；
            # model_call_count 同步写回是为了 Checkpoint 恢复时不重置调用额度。
            update: dict[str, Any] = {
                "case_analysis": analysis,
                "current_fact_overrides": [
                    item.model_dump() for item in analysis.current_fact_overrides
                ],
                "active_skills": active_skills,
                "skill_outputs": skill_outputs,
                "model_call_count": runtime.context.metrics.model_call_count,
            }
            # 步骤 2A-6：无需研究的两种请求直接准备最终正文。after_analysis 会把
            # Graph 路由到 finalize，而不是进入 MCP Research。
            if analysis.next_action == "direct_answer":
                update["final_answer"] = analysis.direct_answer
            elif analysis.next_action == "ask_clarification":
                questions = analysis.clarification_questions or analysis.missing_facts
                update["final_answer"] = "为了更准确地分析，请补充以下信息：\n\n" + "\n".join(
                    f"- {item}" for item in questions
                )
            return update

        # 步骤 2B（生产路径）：调用 Analyst 模型并得到结构化 CaseAnalysis。
        try:
            # 步骤 2B-1：Progressive Disclosure 只给 Analyst Skill 名称和描述目录，
            # 尚未把所有完整 Skill 指令塞入 Prompt，减少上下文并避免无关能力干扰。
            analyst_prompt = f"{ANALYST_PROMPT}\n\n{self.skill_registry.catalog_prompt()}"
            # 步骤 2B-2：_invoke_json 使用 ChatOpenAI.ainvoke，返回完整 AIMessage，
            # 再经 JSON 提取和 CaseAnalysis Pydantic 校验；该调用不绑定任何工具。
            analysis = await self._invoke_json(
                runtime, "case_analyst", analyst_prompt, _payload(state), CaseAnalysis
            )
        except (ValueError, ValidationError, RuntimeError) as exc:
            # 步骤 2B-失败：模型输出无法解析、Schema 不合法或调用额度耗尽时，采用
            # 保守研究路线，避免把法律问题误当闲聊直接回答。错误写入 State 供审计，
            # 但不会把异常堆栈或内部细节直接返回用户。
            question = _message_text(state["messages"][-1]) if state["messages"] else ""
            analysis = CaseAnalysis(request_type="legal_consultation", case_summary=question, legal_issues=[question], research_tasks=[ResearchTask(issue_id="issue-1", query=question, purpose="核验法律依据")], next_action="research")
            return {
                "case_analysis": analysis,
                "errors": [*state["errors"], AgentError(agent="case_analyst", message=str(exc))],
                "model_call_count": runtime.context.metrics.model_call_count,
            }

        # 步骤 3：对生产模型给出的 Fact Override 执行与评测路径相同的所有权校验。
        validated_overrides = await asyncio.to_thread(
            _validate_fact_overrides, analysis.current_fact_overrides, runtime.context
        )
        analysis = analysis.model_copy(update={"current_fact_overrides": validated_overrides})

        # 步骤 4：服务端裁决 requested_skill_ids。未知、越权或超限 Skill 会被拒绝；
        # 合法 Skill 的安全元数据和结构化输出才进入本轮 State。
        active_skills, skill_outputs = await self._activate_skills(state, runtime, analysis)

        # 步骤 5：构造统一的 State 增量，供下一节点及 Checkpoint 使用。
        update: dict[str, Any] = {
            "case_analysis": analysis,
            "current_fact_overrides": [
                item.model_dump() for item in analysis.current_fact_overrides
            ],
            "active_skills": active_skills,
            "skill_outputs": skill_outputs,
            "model_call_count": runtime.context.metrics.model_call_count,
        }

        # 步骤 6：闲聊/澄清提前写 final_answer；法律咨询不写正文，等待 Research。
        if analysis.next_action == "direct_answer":
            update["final_answer"] = analysis.direct_answer or "您好，请告诉我需要咨询的法律问题。"
        elif analysis.next_action == "ask_clarification":
            questions = analysis.clarification_questions or analysis.missing_facts
            update["final_answer"] = "为了更准确地分析，请补充以下信息：\n\n" + "\n".join(f"- {item}" for item in questions)

        # 步骤 7：返回的 dict 不是 HTTP 响应；LangGraph 会先合并 State，然后调用
        # after_analysis 选择 legal_researcher 或 finalize。
        return update

    def after_analysis(self, state: LegalConsultationState) -> str:
        """LangGraph 条件路由：只有 next_action=research 才进入 MCP 研究节点。"""
        # 步骤 1：读取 Analyst 已写入的结构化结果，不再次调用模型。
        analysis = state["case_analysis"]
        # 步骤 2：返回值必须匹配 _compile() 中 conditional_edges 的映射 key。
        return "research" if analysis and analysis.next_action == "research" else "finish"

    async def legal_researcher(self, state: LegalConsultationState, runtime: Runtime[AgentInvocationContext]) -> dict[str, Any]:
        """运行唯一允许调用 MCP 的 LangChain Agent，并生成可验证 EvidencePacket。

        输入：CaseAnalysis 中的争议点/研究任务，以及补检索时的 ReviewResult。
        输出：EvidencePacket、实际模型/工具计数和工具轨迹。工具返回的候选必须再次
        映射为权威 chunk，模型不能自行提供法条正文或证据元数据。
        """

        # 步骤 1：发布“研究中”状态。此事件只描述阶段，不包含查询参数或法规正文。
        runtime.stream_writer({"event": "agent_status", "data": {"agent": "legal_researcher", "status": "researching", "message": "正在检索法律依据"}})

        # 步骤 2：读取 Analyst 产物；正常路由下它一定存在，但类型仍允许兜底。
        analysis = state["case_analysis"]

        # 步骤 3：处理“从未发现到 MCP 工具”的降级。Runtime 仍可编译无工具 Graph，
        # 这里明确返回 tool_unavailable，而不是把能力故障误写成正常 no_match。
        if self.research_agent is None:
            issues = [UnresolvedIssue(issue_id=f"issue-{index + 1}", description=item) for index, item in enumerate(analysis.legal_issues if analysis else [])]
            return {"evidence_packet": EvidencePacket(retrieval_status="tool_unavailable", research_tasks=analysis.research_tasks if analysis else [], unresolved_issues=issues, research_summary="法律检索工具当前不可用。")}

        # 步骤 4：取得当前 invocation 的身份、计数和 stream writer 上下文，并开始
        # 节点耗时审计。context 属于本轮请求，不保存到共享 research_agent 实例。
        context = runtime.context
        started = time.perf_counter()
        audit("agent.node.started", status="started", agent="legal_researcher", **context.audit_fields)

        # 步骤 5：把 Graph State 转成模型可消费的普通字典。若 Reviewer 要求补检索，
        # 只追加明确的 supplemental_instruction，不允许模型自由读取整个 Checkpoint。
        research_payload = _payload(state)
        if state["review_result"] and state["review_result"].revision_instruction:
            research_payload["supplemental_instruction"] = state["review_result"].revision_instruction
        try:
            # 步骤 6：初始化本次 LangChain 子 Agent 流的聚合变量。
            # final_message 保存最后一条 AIMessage；candidates 只收集成功 ToolMessage；
            # seen_tool_messages 防止 updates 重复；两个布尔值用于区分空结果与工具故障。
            final_message: BaseMessage | None = None
            candidates: list[dict[str, Any]] = []
            seen_tool_messages: set[str] = set()
            successful_tool_result = False
            failed_tool_result = False
            # 步骤 7：只加载允许 legal_researcher 使用的已激活 Skill 指令。没有合法
            # Skill 时返回空字符串，不影响基础检索 Prompt。
            skill_prompt = self.skill_registry.prompt_for(
                self._active_skill_ids(state), "legal_researcher"
            )

            # 步骤 8：启动 LangChain Agent 内部模型—工具循环。传入 SystemMessage
            # 约束研究角色，HumanMessage 携带结构化任务；context 供 Middleware 读取
            # 身份/计数。updates 携带模型/工具 Message，custom 携带 Middleware 写出
            # 的安全状态；完整工具参数、法规正文和推理不会直接转发给客户端。
            async for part in self.research_agent.astream(
                {"messages": [SystemMessage(content="\n\n".join(item for item in (RESEARCH_PROMPT, skill_prompt) if item)), HumanMessage(content=json.dumps(research_payload, ensure_ascii=False, default=str))]},
                context=context,
                stream_mode=["updates", "custom"],
                version="v2",
            ):
                # 步骤 8A：custom event 已由 ToolAuditMiddleware 做过安全裁剪，可继续
                # 写入外层 Graph stream；它通常表示工具开始、成功、失败或超时。
                if part.get("type") == "custom" and isinstance(part.get("data"), dict):
                    runtime.stream_writer(part["data"])

                # 步骤 8B：updates 是 LangChain 子 Agent 的内部状态增量。只读取其中
                # Message，不把完整 update 或内部 Agent State 暴露给外层调用者。
                elif part.get("type") == "updates" and isinstance(part.get("data"), dict):
                    for update in part["data"].values():
                        messages = update.get("messages", []) if isinstance(update, dict) else []
                        # 最新消息可能是模型消息或 ToolMessage；循环完成后最后一条
                        # AIMessage 应包含 RESEARCH_PROMPT 要求的 EvidencePacket JSON。
                        if messages:
                            final_message = messages[-1]
                        for message in messages:
                            # 普通 AI/Human/System Message 不含工具结果，直接跳过。
                            if not isinstance(message, ToolMessage):
                                continue
                            # state update 可能重复包含旧 ToolMessage，按 tool_call_id
                            # 去重，避免同一次 MCP 结果被重复收集和审计。
                            message_id = str(message.tool_call_id)
                            if message_id in seen_tool_messages:
                                continue
                            seen_tool_messages.add(message_id)
                            # 错误 ToolMessage 不解析为法规候选，但会参与最终 tool_error
                            # 判断；成功 ToolMessage 才递归提取 document/chunk 元数据。
                            if message.status == "error":
                                failed_tool_result = True
                            else:
                                successful_tool_result = True
                                candidates.extend(_tool_documents(message))

            # 步骤 9：没有任何最终消息说明 LangChain 子 Agent 未正常收敛，转入统一
            # exception 分支生成 tool_error EvidencePacket。
            if final_message is None:
                raise RuntimeError("法律研究 Agent 未返回结果")

            # 步骤 10：最后一条 AIMessage 是模型整理的研究 JSON。先验证结构，再使用真实
            # ToolMessage 候选回填元数据；绝不直接信任模型生成的正文或证据 ID。
            raw_packet = EvidencePacket.model_validate(_extract_json(_message_text(final_message)))

            # 步骤 11：兼容模型只返回 accepted_chunk_ids 的情况。先用候选构造最小
            # EvidenceItem 选择声明，后续仍由 _authoritative_evidence 回填权威字段。
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
            # 步骤 12：执行第一次权威映射。伪造 ID、歧义 document_id 和不在本轮
            # ToolMessage 中的证据都会被删除。
            accepted = _authoritative_evidence(raw_packet, candidates)
            if candidates and not accepted and not raw_packet.rejected_candidates:
                # 步骤 13（按需）：已有候选但模型既未接受也未拒绝时，追加一次无工具
                # 受限选择。Selector 只能选择已有 chunk，不会重新搜索或产生新证据。
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
                # 步骤 13A：即使 Selector 通过 Schema，也要用候选 ID 集合再次裁剪，
                # 防止模型输出本轮不存在的 chunk_id。
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
                # 步骤 13B：把 Selector 结果再次映射为带真实正文/条号的证据对象。
                accepted = _authoritative_evidence(selected_packet, candidates)
                raw_packet = selected_packet

            # 步骤 14：状态由真实工具结果和权威证据共同决定：有采纳证据为 matched；仅有
            # 失败 ToolMessage 为 tool_error；其余 evidence_items=[] 是正常 no_match。
            if accepted:
                retrieval_status = "matched"
            elif failed_tool_result and not successful_tool_result:
                retrieval_status = "tool_error"
            else:
                retrieval_status = "no_match"
            # 步骤 15：保证 no_match 至少携带一个可解释的未解决问题，供 Counsel
            # 说明证据限制；这不是异常，也不触发第二次相同查询。
            unresolved = raw_packet.unresolved_issues
            if retrieval_status == "no_match" and not unresolved:
                unresolved = [UnresolvedIssue(issue_id="issue-1", description="当前法规库未检索到可直接引用的依据")]
            # 步骤 16：归一化最终 EvidencePacket。candidate_status 描述是否召回候选，
            # evidence_status 描述候选是否被接受，retrieval_status 描述业务终态。
            # EvidencePacket 是 Counsel/Reviewer 唯一允许引用的法规事实源。空结果
            # 可以正常继续回答，不会仅因候选未被采纳而触发异常或重复检索。
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
            # 步骤 17：按 chunk/document ID 去重后记录节点指标，不写完整法规正文。
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
            # 步骤 18：返回 State 增量。若本次由 Reviewer 回流，retry_count 增加；
            # 调用计数和轨迹同步入 State，确保 Checkpoint 恢复后不归零。
            return {
                "evidence_packet": packet,
                "retry_count": state["retry_count"] + (1 if state["review_result"] else 0),
                "model_call_count": context.metrics.model_call_count,
                "tool_call_count": context.metrics.tool_call_count,
                "tool_trajectory": list(context.metrics.tool_trajectory),
            }
        except Exception as exc:  # noqa: BLE001 - research failures become controlled evidence state
            # 失败步骤：研究节点内部异常被收敛成 tool_error，使后续 Counsel 能明确披露未完成
            # 法规核验；这样工具故障不会让整条咨询任务永久卡住。
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
        """基于案情、记忆快照和 EvidencePacket 生成待复核法律意见草稿。

        输入：CaseAnalysis、EvidencePacket、当前有效 Skill 输出和 Reviewer 修改要求。
        输出：CounselDraft、经 Schema 验证的 Skill 输出及 revision_count。该节点无
        MCP Tool 权限，草稿也不会在此处直接持久化或输出给用户。
        """

        # 步骤 1：读取 Research 证据状态，并把 no_match 单独建模。no_match 表示工具
        # 正常完成但无可引用证据，不等同于 tool_error/tool_unavailable。
        evidence = state["evidence_packet"]
        no_match = bool(evidence and evidence.retrieval_status == "no_match")

        # 步骤 2：根据证据状态发布用户可理解的阶段信息；此事件不会携带草稿正文。
        message = "未检索到可引用法条，将基于案情形成一般性分析" if no_match else "正在形成法律意见"
        runtime.stream_writer({"event": "agent_status", "data": {"agent": "legal_counsel", "status": "drafting", "message": message}})

        # 步骤 3：构造有限模型输入，并显式告诉 Counsel 当前是“基于证据回答”还是
        # “无法条的一般性分析”，避免模型仅根据空数组自行猜测模式。
        payload = _payload(state)
        payload["answer_mode"] = "general_analysis_without_citations" if no_match else "evidence_based"

        # 步骤 4：若 Reviewer 要求改稿，只注入简短 revision_instruction；不把
        # Reviewer 隐藏推理或完整 Checkpoint 暴露给 Counsel。
        if state["review_result"] and state["review_result"].revision_instruction:
            payload["revision_instruction"] = state["review_result"].revision_instruction

        # 步骤 5：取得本轮已由服务端激活的 Skill ID。
        active_ids = self._active_skill_ids(state)

        # 步骤 6：选中的 Skill 还要按 legal_counsel 角色二次过滤；未授权 Skill 的完整
        # 指令不会进入本节点 Prompt。
        counsel_skills = self.skill_registry.resolve(active_ids, "legal_counsel")

        # 步骤 7：对每个合法 Counsel Skill 发布开始事件和审计记录。这里还没有
        # 单独调用模型；这些 Skill 将作为同一次 Counsel Prompt 的附加领域约束。
        for item in counsel_skills:
            self._skill_event(runtime, item.summary.skill_id, "running")
            audit(
                "skill.execution.started",
                status="started",
                skill_id=item.summary.skill_id,
                **runtime.context.audit_fields,
            )
        try:
            # 步骤 8：此时才按需加载完整 Skill 正文，实现 Progressive Disclosure。
            skill_prompt = self.skill_registry.prompt_for(active_ids, "legal_counsel")

            # 步骤 9：_invoke_json 是无工具的完整消息调用。输入含分析、证据、有限记忆和
            # Fact Override；输出 CounselDraft 此时尚未写入业务 messages 表。
            draft = await self._invoke_json(
                runtime,
                "legal_counsel",
                "\n\n".join(item for item in (COUNSEL_PROMPT, skill_prompt) if item),
                payload,
                CounselDraft,
            )
            try:
                # 步骤 10：模型输出的 Skill 结果仍须满足“已激活 + 角色允许 + Schema 合法”，
                # 否则不会进入 Graph State 或最终回答。
                validated_outputs = self.skill_registry.validate_outputs(
                    draft.skill_outputs, active_ids, "legal_counsel"
                )
            except ValidationError as exc:
                # 步骤 10-失败：某个 Skill 输出结构错误时整批丢弃该组可选输出并记录
                # 安全摘要，但保留基础 CounselDraft，体现 Skill fail-open。
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
            # 步骤 11：用验证后的结果替换模型原始 skill_outputs，再与此前例如
            # case-intake 的结果合并；未验证输出不会进入后续 Reviewer/Finalize。
            draft = draft.model_copy(update={"skill_outputs": validated_outputs})
            merged_outputs = {**state.get("skill_outputs", {}), **validated_outputs}

            # 步骤 12：逐个 Skill 发布成功/失败状态。是否完成以对应 Schema 输出是否
            # 存在为准，不以“模型调用没有抛异常”代替业务完成。
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
            # 步骤 13：no_match 模式强制补充披露并把 confidence 压到 low。即使模型
            # 遗漏了说明，也由代码追加；具体法名/条号仍会在 Reviewer/Finalize 检查。
            if no_match:
                disclosure = "本轮法规检索正常完成，但当前法规库中未检索到可引用法条。以上属于一般性分析，不构成已经过法规核验的确定性法律结论。"
                answer = draft.answer if _no_match_disclosure_present(draft.answer) else f"{draft.answer.rstrip()}\n\n## 检索说明\n\n{disclosure}"
                draft = draft.model_copy(update={"answer": answer, "confidence": "low"})
            # 步骤 14：返回草稿 State 增量。只有由 Reviewer 回流才增加 revision_count，
            # 该计数会随 Checkpoint 保存，确保服务重启后不能再次免费改稿。
            return {
                "counsel_draft": draft,
                "skill_outputs": merged_outputs,
                "revision_count": state["revision_count"] + (1 if state["review_result"] else 0),
                "model_call_count": runtime.context.metrics.model_call_count,
            }
        except Exception as exc:  # noqa: BLE001 - counsel failures use a safe user-facing fallback
            # 失败步骤 1：Skill/Counsel 调用、解析或 Schema 校验异常时，先把本节点
            # 所有可选 Skill 标为失败，避免 UI 长时间停留在 running。
            # Skill/Counsel 失败采用安全草稿继续到复核；最终消息仍只能由 Finalize
            # 和 AgentRunManager 的幂等持久化阶段产生。
            for item in counsel_skills:
                self._skill_event(
                    runtime,
                    item.summary.skill_id,
                    "failed",
                    "领域分析未完成，继续安全兜底回答",
                )
            # 失败步骤 2：no_match 使用确定性安全模板；其他状态仅输出研究摘要和
            # 明确限制，不把异常内容或未经复核的半成品草稿返回用户。
            fallback_answer = _no_match_safe_answer(state) if no_match else (evidence.research_summary if evidence else "暂时无法形成完整法律意见。") + "\n\n当前回答生成失败，建议稍后重试或咨询专业律师。"
            fallback = CounselDraft(answer=fallback_answer, confidence="low", limitations=["回答生成或法规核验未完整完成"])
            # 失败步骤 3：把安全草稿和 AgentError 写回 State，让 Graph 仍进入
            # review_gate；错误用于审计，不直接作为最终正文。
            return {
                "counsel_draft": fallback,
                "errors": [*state["errors"], AgentError(agent="legal_counsel", message=str(exc))],
                "model_call_count": runtime.context.metrics.model_call_count,
            }

    async def reviewer(self, state: LegalConsultationState, runtime: Runtime[AgentInvocationContext]) -> dict[str, Any]:
        """执行 Case Analyst 的 LLM 复核阶段，而不是另一个工具型服务。

        ReviewResult 可要求 finalize、research_again 或 revise_draft；路由函数会用
        retry_count/revision_count 强制限制回流次数。
        """

        # 步骤 1：发布复核状态。Reviewer 仍属于 Case Analyst 的复核阶段，因此前端
        # agent 名称沿用 case_analyst，而状态明确为 reviewing。
        runtime.stream_writer({"event": "agent_status", "data": {"agent": "case_analyst", "status": "reviewing", "message": "正在核验回答"}})
        try:
            # 步骤 2：按 reviewer 角色加载本轮合法 Skill 指令；没有授权 Skill 时
            # prompt_for 返回空字符串，基础复核规则仍然执行。
            skill_prompt = self.skill_registry.prompt_for(
                self._active_skill_ids(state), "reviewer"
            )
            # 步骤 3：执行一次无工具结构化模型调用。输入包含草稿、证据包、分析和
            # 修订计数，输出必须满足 ReviewResult Schema。
            review = await self._invoke_json(
                runtime,
                "case_analyst_reviewer",
                "\n\n".join(item for item in (REVIEW_PROMPT, skill_prompt) if item),
                _payload(state),
                ReviewResult,
            )
        except Exception as exc:  # noqa: BLE001 - review failures finalize with explicit limitations
            # 步骤 3-失败：Reviewer 自身失败时不无限重试模型，而是生成“未批准但
            # 结束”的 ReviewResult。Finalize 会依据限制输出安全结果。
            review = ReviewResult(approved=False, revision_instruction="自动复核未完成，最终回答应保留风险提示。", next_action="finalize")
            return {
                "review_result": review,
                "errors": [*state["errors"], AgentError(agent="case_analyst_reviewer", message=str(exc))],
                "model_call_count": runtime.context.metrics.model_call_count,
            }
        # 步骤 4：LLM 复核之后仍执行代码级引用校验。Reviewer 的 approved 只是模型
        # 意见，不能覆盖“引用必须属于本轮 EvidencePacket”的硬约束。
        deterministic_errors = _citation_errors(state["counsel_draft"], state["evidence_packet"])
        packet = state["evidence_packet"]

        # 步骤 5：no_match 额外检查是否出现具体法名/条号及是否披露未找到依据。
        if packet and packet.retrieval_status == "no_match":
            deterministic_errors.extend(_no_match_violations(state["counsel_draft"].answer if state["counsel_draft"] else ""))

        # 步骤 6：检查回答是否继续使用已被当前用户消息替换的旧事实。
        deterministic_errors.extend(
            _fact_boundary_errors(
                state,
                state["counsel_draft"].answer if state["counsel_draft"] else "",
            )
        )

        # 步骤 7：任何确定性错误都覆盖模型的批准结论，并要求仅修改草稿；这些问题
        # 不需要重新检索，因为证据集合没有发生变化。
        if deterministic_errors:
            review.approved = False
            review.citation_errors = list(dict.fromkeys([*review.citation_errors, *deterministic_errors]))
            review.next_action = "revise_draft"
            instruction = "删除或改写所有不在 EvidencePacket 中的法条和证据引用。"
            review.revision_instruction = " ".join(
                item for item in (review.revision_instruction, instruction) if item
            )
        # 步骤 8：no_match 没有硬错误时，根据模型指出的无依据论断/矛盾决定改稿；
        # 两者都为空才批准结束，绝不因为 evidence_items=[] 返回 Research。
        elif packet and packet.retrieval_status == "no_match":
            if review.unsupported_claims or review.contradictions:
                review.approved = False
                review.next_action = "revise_draft"
            else:
                review.approved = True
                review.next_action = "finalize"
        # 步骤 9：返回 ReviewResult 和最新模型计数。after_review 随后把 next_action
        # 映射为 finalize、Research 或 Counsel。
        return {
            "review_result": review,
            "model_call_count": runtime.context.metrics.model_call_count,
        }

    async def review_gate(
        self, state: LegalConsultationState, runtime: Runtime[AgentInvocationContext]
    ) -> dict[str, Any]:
        """用确定性规则决定是否需要额外执行一次 LLM Reviewer。

        只有低风险、no_match、低置信度、无证据引用且通过披露检查的回答可以
        Fast Path。其余情况继续走 Reviewer，以节省简单问题延迟而不降低安全边界。
        """

        # 步骤 1：读取 Gate 所需的三个结构化产物；Gate 本身不调用模型或工具。
        analysis = state["case_analysis"]
        packet = state["evidence_packet"]
        draft = state["counsel_draft"]

        # 步骤 2：skip_reason 命名表示“不能跳过 Reviewer 的原因”。非空就进入 LLM；
        # 空字符串表示所有 Fast Path 条件都通过。
        skip_reason = ""

        # 步骤 3：评测可强制 always-llm，确保 Baseline 与自动 Fast Path 可公平对比。
        if self.settings.agent_review_mode == "always-llm":
            skip_reason = "评测配置要求始终执行模型复核"

        # 步骤 4：Reviewer 要求改稿后必须再次复核，不能让第二版草稿绕过安全检查。
        elif state["review_result"] is not None:
            skip_reason = "修订后的草稿必须再次复核"

        # 步骤 5：中高风险问题始终交给 LLM Reviewer；只有 low 才有资格 Fast Path。
        elif not analysis or analysis.risk_level != "low":
            skip_reason = "中高风险问题必须进行模型复核"

        # 步骤 6：Fast Path 当前只服务正常 no_match。matched 需要审查证据一致性，
        # tool_error/unavailable 需要审查降级表述，因此都不能跳过。
        elif not packet or packet.retrieval_status != "no_match":
            skip_reason = "存在法规依据或检索异常，必须进行模型复核"

        # 步骤 7：无法条回答必须是低置信度，否则说明 Counsel 越过证据边界。
        elif not draft or draft.confidence != "low":
            skip_reason = "无法条回答的置信度边界未满足"

        # 步骤 8：确定性检查披露和法条幻觉；失败时交给 Reviewer 指导改稿。
        elif _no_match_violations(draft.answer):
            skip_reason = "无法条回答未通过确定性边界校验"

        # 步骤 9：no_match 的 claims 不得携带任何证据 ID。
        elif any(claim.evidence_chunk_ids or claim.evidence_document_ids for claim in draft.claims):
            skip_reason = "无法条回答包含证据引用"

        # 步骤 10A：存在任一不能跳过的原因时，不写 ReviewResult；after_review_gate
        # 看到空结果会进入 reviewer 节点。review_mode 同步到评测和 Checkpoint 指标。
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

        # 步骤 10B：全部确定性门禁通过时，生成等价的已批准 ReviewResult，直接去
        # Finalize，节省一次 LLM 调用但保留相同最终安全出口。
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
        """LangGraph 路由：Gate 已批准则 finalize，否则进入 LLM Reviewer。"""
        # 步骤 1：Gate Fast Path 会写 approved=True；需要 LLM 时返回空增量，State
        # 中 review_result 仍为空（首次草稿）或未批准（修订场景）。
        review = state["review_result"]
        # 步骤 2：返回值对应 _compile() 中 finish->finalize、review->reviewer。
        return "finish" if review and review.approved else "review"

    def after_review(self, state: LegalConsultationState) -> str:
        """把 ReviewResult 映射为结束、补检索或改稿，并强制循环上限。"""

        # 步骤 1：读取 Reviewer 决策和当前证据状态；路由函数不调用外部服务。
        review = state["review_result"]
        packet = state["evidence_packet"]

        # 步骤 2：无复核结果、已经批准或明确 finalize 时结束，避免无意义回流。
        if not review or review.approved or review.next_action == "finalize":
            return "finish"

        # 步骤 3：空结果或工具故障无法通过相同查询可靠补足；即使模型建议
        # research_again，也只允许一次必要草稿修订，不再次消耗 MCP 调用。
        if packet and packet.retrieval_status in {"no_match", "tool_unavailable", "tool_error"}:
            return "revise" if review.next_action == "revise_draft" and state["revision_count"] < 1 else "finish"

        # 步骤 4：matched 模式只有明确 research_again 且尚未补检索过，才返回
        # research；retry_count>=1 后强制结束。
        if review.next_action == "research_again" and state["retry_count"] < 1:
            return "research"

        # 步骤 5：表达/论证问题最多返回 Counsel 修改一次。
        if review.next_action == "revise_draft" and state["revision_count"] < 1:
            return "revise"

        # 步骤 6：未知动作或额度耗尽统一结束，防止 Graph 无限循环。
        return "finish"

    async def finalize(self, state: LegalConsultationState, runtime: Runtime[AgentInvocationContext]) -> dict[str, Any]:
        """纯代码最终出口：收敛回答、执行安全边界并生成可追踪 Citation。

        本节点不调用模型或 MCP。它检查 no_match、复核结果和引用归属，必要时替换
        为安全模板；只有草稿 claims 实际引用且存在于 EvidencePacket 的 chunk 才
        会进入 citations，随后由 AgentRuntime/AgentRunManager 输出并持久化。
        """

        # 步骤 1：确定正文来源。闲聊/澄清由 Analyst 直接写 final_answer；法律咨询
        # 通常使用 CounselDraft.answer。此时只在内存 State 中，尚未保存业务消息。
        answer = state["final_answer"] or (state["counsel_draft"].answer if state["counsel_draft"] else "")

        # 步骤 2：读取 Research 和 Review 产物，后续所有安全判断都使用结构化对象，
        # 不根据自然语言猜测工具是否成功或 Reviewer 是否批准。
        packet = state["evidence_packet"]
        review = state["review_result"]

        # 步骤 3A：no_match 必须同时满足“Reviewer 已批准”和确定性无幻觉检查。
        # 任一条件失败都用固定安全模板覆盖草稿，而不是把越界内容继续输出。
        if packet and packet.retrieval_status == "no_match":
            violations = _no_match_violations(answer)
            if not review or not review.approved or violations:
                answer = _no_match_safe_answer(state)

        # 步骤 3B：非 no_match 且 Reviewer 明确发现无依据论断、遗漏、引用错误或
        # 矛盾时，不输出原草稿；只列出已经核验的证据摘要和重新咨询建议。
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

        # 步骤 4：初始化 Citation 聚合器。seen_documents 实际存放 chunk/document
        # 证据键，用于避免多个 claim 引用同一 chunk 时重复展示。
        citations: list[Citation] = []
        seen_documents: set[str] = set()

        # 步骤 5：检索候选不等于用户可见引用。这里只收集 Counsel claims 明确使用
        # 的 chunk_id，以及兼容旧 Fixture 的 document_id。
        referenced_ids = {
            evidence_id
            for claim in (state["counsel_draft"].claims if state["counsel_draft"] else [])
            for evidence_id in [*claim.evidence_chunk_ids, *claim.evidence_document_ids]
        }
        # 步骤 6：统计同一原始法条在 EvidencePacket 中有几个 chunk。只有计数为 1
        # 时，旧 document_id 才能无歧义映射到具体片段。
        document_counts: dict[str, int] = {}
        for item in packet.evidence_items if packet else []:
            document_counts[item.document_id] = document_counts.get(item.document_id, 0) + 1
        # 步骤 7：只遍历 matched EvidencePacket。Citation 的法名、条号和摘录来自
        # 服务端验证后的 EvidenceItem，不能从回答
        # 文本反向猜测，因此可以追溯到本轮 MCP 返回的具体 chunk。
        for item in packet.evidence_items if packet and packet.retrieval_status == "matched" else []:
            # 步骤 7A：判断当前 chunk 是否被草稿实际引用。旧 document_id 只有单
            # chunk 时才兼容，避免长法条错误引用第一个片段。
            legacy_match = item.document_id in referenced_ids and document_counts[item.document_id] == 1
            if item.chunk_id not in referenced_ids and not legacy_match:
                continue

            # 步骤 7B：去重证据键，同一个法条片段最多生成一条 Citation。
            evidence_key = item.chunk_id or item.document_id
            if evidence_key in seen_documents:
                continue
            seen_documents.add(evidence_key)
            # 步骤 7C：构造用户可见引用，摘录限制为 240 字；完整检索候选和内部
            # 分数不会进入最终 Citation/SSE。
            citations.append(Citation(document_id=item.document_id, chunk_id=item.chunk_id, law_name=item.law_name, article_number=item.article_number, quoted_excerpt=item.content[:240], data_version=item.data_version))

        # 步骤 8：发布最终阶段状态。正文和 citations 仍由节点返回值交给 Runtime，
        # stream_writer 在这里仅表示“图分析完成”。
        runtime.stream_writer({"event": "agent_status", "data": {"agent": "case_analyst", "status": "completed", "message": "分析已完成"}})

        # 步骤 9：返回最终 State 增量。LangGraph 随后到达 END 并写最终 Checkpoint；
        # AgentRuntime 才把 final_answer 切成安全 token，AgentRunManager 再幂等落库。
        return {
            "final_answer": answer,
            "citations": citations,
            "model_call_count": runtime.context.metrics.model_call_count,
            "tool_call_count": runtime.context.metrics.tool_call_count,
            "tool_trajectory": list(runtime.context.metrics.tool_trajectory),
        }
