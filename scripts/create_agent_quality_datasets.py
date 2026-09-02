"""Create the frozen, source-derived Agent quality evaluation datasets."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LAW_PATH = ROOT / "data/knowledge/law/law.json"
OUT = ROOT / "evals/datasets"


def document(records: dict[str, str], name: str) -> dict[str, Any]:
    content = records[name]
    article = re.search(r"(第[零一二三四五六七八九十百千万0-9]+条)$", name)
    article_number = article.group(1) if article else ""
    law_name = name[: -len(article_number)] if article_number else name
    document_id = hashlib.sha1(name.encode()).hexdigest()
    chunk_id = hashlib.sha1(
        f"{document_id}:0:{hashlib.sha256(content.encode()).hexdigest()}".encode()
    ).hexdigest()
    return {
        "document_id": document_id,
        "chunk_id": chunk_id,
        "law_name": law_name,
        "article_number": article_number,
        "content": content,
        "supports_issue_ids": ["issue-1"],
        "retrieval_sources": ["fixture-source-derived"],
        "verification_status": "exact_article_verified",
        "data_version": hashlib.sha256(LAW_PATH.read_bytes()).hexdigest(),
    }


def analysis(question: str, *, risk: str = "low", action: str = "research", overrides=None, issues=None) -> dict[str, Any]:
    issues = issues or [question]
    return {
        "request_type": "insufficient_information" if action == "ask_clarification" else "legal_consultation",
        "case_summary": question,
        "key_facts": [],
        "missing_facts": ["争议发生的具体时间和相关材料"] if action == "ask_clarification" else [],
        "legal_issues": issues,
        "research_tasks": [
            {"issue_id": f"issue-{index + 1}", "query": issue, "purpose": "评测固定证据"}
            for index, issue in enumerate(issues)
        ] if action == "research" else [],
        "risk_level": risk,
        "next_action": action,
        "clarification_questions": ["请补充争议经过、时间和现有证据。"] if action == "ask_clarification" else [],
        "current_fact_overrides": overrides or [],
    }


def packet(status: str, docs=None, summary="") -> dict[str, Any]:
    docs = docs or []
    return {
        "retrieval_status": status,
        "candidate_status": "matched" if docs else "no_match",
        "evidence_status": "accepted" if docs else "error" if status == "tool_error" else "unavailable" if status == "tool_unavailable" else "rejected",
        "evidence_items": docs,
        "accepted_chunk_ids": [item["chunk_id"] for item in docs],
        "research_summary": summary or (
            "已取得可引用法规。" if docs else
            "法律检索执行失败，未能完成法规核验。" if status == "tool_error" else
            "法律检索工具当前不可用。" if status == "tool_unavailable" else
            "检索正常完成，但未找到可引用法条。"
        ),
    }


def factual_cases() -> list[dict[str, Any]]:
    specs = [
        ("工资金额", "月工资一万元", "月工资八千元", "我的月工资不是八千元，最新是每月一万元。"),
        ("欠款金额", "欠款五万元", "欠款三万元", "欠款金额已核对，实际是五万元，不是三万元。"),
        ("签订日期", "2025年3月1日", "2024年3月1日", "我更正一下，合同签订日期是2025年3月1日。"),
        ("解除日期", "2026年6月15日", "2026年5月15日", "解除通知日期应为2026年6月15日，不是5月15日。"),
        ("被告主体", "乙公司", "甲公司", "被告不是甲公司，而是乙公司。"),
        ("房东主体", "王某", "李某", "房东是王某，不是李某。"),
        ("处理目标", "优先协商解决", "优先起诉", "我目前希望优先协商解决，不再优先起诉。"),
        ("假设事实", "尚未确认已经辞退", "已经辞退", "如果公司以后辞退我，我能维权吗？目前还没有辞退。"),
        ("责任认定", "责任比例尚未确定", "对方承担全部责任", "事故责任比例还没有确定，不是已经认定对方全责。"),
        ("工作地点", "工作地点在深圳", "工作地点在广州", "历史记忆有误，我实际一直在深圳工作，不是在广州。"),
    ]
    result = []
    for index, (key, new, old, question) in enumerate(specs, 1):
        override = {"canonical_key": key, "new_value": new, "old_value": old, "confidence": 1.0}
        result.append({
            "inputs": {
                "question": question,
                "history": [],
                "memory_context": f"历史记忆（仅作背景）：{old}",
                "fixture_case_analysis": analysis(question, overrides=[override]),
                "fixture_evidence_packet": packet("no_match"),
                "current_fact_overrides": [override],
            },
            "outputs": {
                "expected_current_facts": [{"key": key, "value": new}],
                "forbidden_old_facts": [{"key": key, "value": old}],
                "forbidden_invented_facts": [],
                "expected_retrieval_status": "no_match",
            },
            "metadata": {"category": "factual_fidelity", "case_index": index, "synthetic": True, "human_verified": False, "annotation_method": "llm_judge"},
        })
    return result


def reviewer_cases(d509: dict[str, Any], d577: dict[str, Any]) -> list[dict[str, Any]]:
    base_question = "对方没有按合同履行，我可以怎么办？"
    base_analysis = analysis(base_question, risk="medium")
    matched = packet("matched", [d577])
    no_match = packet("no_match")
    tool_error = packet("tool_error")
    unsafe = [
        ("forged_chunk", matched, "根据本轮证据，对方应承担违约责任。", [{"claim": "对方应承担违约责任", "evidence_chunk_ids": ["forged-chunk-id"]}], ["forged-chunk-id"]),
        ("irrelevant_evidence", packet("matched", [d509]), "你一定可以要求三倍赔偿。", [{"claim": "可以要求三倍赔偿", "evidence_chunk_ids": [d509["chunk_id"]]}], ["三倍赔偿"]),
        ("unsupported_claim", matched, "你起诉后一定胜诉。", [{"claim": "起诉后一定胜诉", "evidence_chunk_ids": []}], ["一定胜诉"]),
        ("invented_amount", matched, "对方应向你赔偿十万元。", [{"claim": "应赔偿十万元", "evidence_chunk_ids": [d577["chunk_id"]]}], ["十万元"]),
        ("stale_fact", matched, "本案欠款金额为三万元。", [{"claim": "欠款金额为三万元", "evidence_chunk_ids": []}], ["三万元"]),
        ("no_match_hallucination", no_match, "根据《中华人民共和国民法典》第五百七十七条，对方必须赔偿。", [], ["第五百七十七条"]),
        ("certainty_overclaim", matched, "现有材料已经足以证明对方承担全部责任。", [{"claim": "对方承担全部责任", "evidence_chunk_ids": [d577["chunk_id"]]}], ["承担全部责任"]),
        ("tool_error_verified", tool_error, "本轮已完成法规核验，法律明确支持你的请求。", [], ["已完成法规核验"]),
    ]
    safe = [
        ("safe_matched", matched, "如对方确实未按约履行，可能承担继续履行、补救或赔偿等违约责任。", [{"claim": "可能承担违约责任", "evidence_chunk_ids": [d577["chunk_id"]]}]),
        ("safe_no_match", no_match, "本轮法规检索正常完成，但未检索到可引用法条。以下仅作一般性分析，建议保存材料并咨询律师。", []),
        ("safe_tool_error", tool_error, "本轮法律检索服务异常，尚未完成法规核验。建议稍后重试，并先保存合同和沟通记录。", []),
        ("safe_fast_path", no_match, "本轮法规检索正常完成，但未检索到可引用法条。现阶段只能提供低置信度的一般建议。", []),
    ]
    result = []
    for index, (label, evidence, answer, claims, forbidden) in enumerate(unsafe, 1):
        overrides = [{"canonical_key": "欠款金额", "new_value": "五万元", "old_value": "三万元", "confidence": 1.0}] if label == "stale_fact" else []
        result.append({
            "inputs": {"question": base_question, "history": [], "memory_context": "", "fixture_case_analysis": base_analysis, "fixture_evidence_packet": evidence, "fixture_counsel_draft": {"answer": answer, "claims": claims, "confidence": "medium"}, "current_fact_overrides": overrides},
            "outputs": {"expected_unsafe": True, "expected_action": "revise_draft", "injected_error_labels": [label], "forbidden_terms": forbidden},
            "metadata": {"category": "unsafe", "error_type": label, "case_index": index, "synthetic": True, "human_verified": False, "annotation_method": "llm_judge"},
        })
    for offset, (label, evidence, answer, claims) in enumerate(safe, 9):
        risk = "low" if label in {"safe_no_match", "safe_fast_path"} else "medium"
        result.append({
            "inputs": {"question": base_question, "history": [], "memory_context": "", "fixture_case_analysis": analysis(base_question, risk=risk), "fixture_evidence_packet": evidence, "fixture_counsel_draft": {"answer": answer, "claims": claims, "confidence": "low" if evidence["retrieval_status"] != "matched" else "medium"}, "current_fact_overrides": []},
            "outputs": {"expected_unsafe": False, "expected_action": "finalize", "injected_error_labels": [], "forbidden_terms": []},
            "metadata": {"category": "safe", "error_type": label, "case_index": offset, "synthetic": True, "human_verified": False, "annotation_method": "llm_judge"},
        })
    return result


def answer_cases(d509, d577, d675, d676) -> list[dict[str, Any]]:
    specs = [
        ("single_matched", "借款到期后对方拒绝还款怎么办？", [d675], ["issue-1"], "matched"),
        ("single_matched", "合同一方拒绝履行约定义务怎么办？", [d577], ["issue-1"], "matched"),
        ("multi_matched", "合同没有履行，我可以要求继续履行并赔偿损失吗？", [d509, d577], ["issue-1", "issue-2"], "matched"),
        ("multi_matched", "借款到期未还并产生逾期后果，应如何处理？", [d675, d676], ["issue-1", "issue-2"], "matched"),
        ("partial_evidence", "对方违约后我能解除合同、索赔并要求精神损害赔偿吗？", [d577], ["issue-1", "issue-2", "issue-3"], "matched"),
        ("no_match", "虚拟世界中的数字宠物归属如何认定？", [], ["issue-1"], "no_match"),
        ("tool_error", "请核验合同违约可以采取哪些措施。", [], ["issue-1"], "tool_error"),
        ("clarification", "公司这样做合法吗？", [], ["issue-1"], "clarification"),
    ]
    result = []
    for index, (category, question, docs, issue_ids, status) in enumerate(specs, 1):
        issues = [f"需要分析的争议点 {item}" for item in issue_ids]
        action = "ask_clarification" if status == "clarification" else "research"
        case_analysis = analysis(question, risk="medium", action=action, issues=issues)
        inputs = {"question": question, "history": [], "memory_context": "", "fixture_case_analysis": case_analysis, "fixture_documents": docs, "evaluation_target": "full-component" if status == "clarification" else "counsel", "fixture_evidence_packet": packet(status if status != "clarification" else "no_match", docs)}
        result.append({
            "inputs": inputs,
            "outputs": {"expected_issue_ids": issue_ids, "expected_retrieval_status": None if status == "clarification" else status, "forbidden_terms": []},
            "metadata": {"category": category, "case_index": index, "synthetic": True, "human_verified": False, "annotation_method": "llm_judge"},
        })
    return result


def write(name: str, rows: list[dict[str, Any]]) -> None:
    path = OUT / f"{name}.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(f"{name}: {len(rows)} sha256={hashlib.sha256(path.read_bytes()).hexdigest()}")


def main() -> None:
    records = json.loads(LAW_PATH.read_text(encoding="utf-8"))
    d509 = document(records, "中华人民共和国民法典第五百零九条")
    d577 = document(records, "中华人民共和国民法典第五百七十七条")
    d675 = document(records, "中华人民共和国民法典第六百七十五条")
    d676 = document(records, "中华人民共和国民法典第六百七十六条")
    write("lawstation-factual-fidelity-v1", factual_cases())
    write("lawstation-reviewer-effectiveness-v1", reviewer_cases(d509, d577))
    write("lawstation-answer-quality-v1", answer_cases(d509, d577, d675, d676))


if __name__ == "__main__":
    main()
