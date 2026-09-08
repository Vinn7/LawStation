"""Graph 节点共用的纯函数：JSON 解析、证据权威映射、边界校验。"""

import json
import logging
import re
from typing import Any

from langchain_core.messages import BaseMessage, ToolMessage
from sqlalchemy import or_, select

from backend.app.agent.schemas import CounselDraft, EvidenceItem, EvidencePacket
from backend.app.agent.state import AgentInvocationContext, LegalConsultationState
from backend.app.core.logging import audit
from backend.app.db.models import UserMemory
from backend.app.db.session import SessionLocal


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
    }
