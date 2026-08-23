import re
from typing import Any


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _outputs(run: Any) -> dict[str, Any]:
    return _mapping(getattr(run, "outputs", run.get("outputs", {}) if isinstance(run, dict) else {}))


def _reference(example: Any) -> dict[str, Any]:
    return _mapping(
        getattr(example, "outputs", example.get("outputs", {}) if isinstance(example, dict) else {})
    )


def _inputs(example: Any) -> dict[str, Any]:
    return _mapping(
        getattr(example, "inputs", example.get("inputs", {}) if isinstance(example, dict) else {})
    )


def _feedback(key: str, score: float | bool, comment: str = "") -> dict[str, Any]:
    return {"key": key, "score": float(score), "comment": comment}


def route_correctness(run: Any, example: Any) -> dict[str, Any]:
    expected = _reference(example).get("expected_route")
    actual = _mapping(_outputs(run).get("case_analysis")).get("next_action")
    return _feedback("route_correctness", expected is None or actual == expected, f"expected={expected}, actual={actual}")


def schema_validity(run: Any, example: Any) -> dict[str, Any]:
    output = _outputs(run)
    analysis = output.get("case_analysis")
    evidence = output.get("evidence_packet")
    valid = isinstance(output.get("final_answer"), str) and isinstance(analysis, dict)
    if _mapping(analysis).get("next_action") == "research":
        valid = valid and isinstance(evidence, dict)
    return _feedback("schema_validity", valid)


def retrieval_status_correctness(run: Any, example: Any) -> dict[str, Any]:
    expected = _reference(example).get("expected_retrieval_status")
    output = _outputs(run)
    actual = _mapping(output.get("evidence_packet")).get("retrieval_status")
    if actual is None:
        actual = output.get("retrieval_status")
    return _feedback("retrieval_status_correctness", expected is None or actual == expected, f"expected={expected}, actual={actual}")


def _ranked_documents(run: Any) -> list[str]:
    output = _outputs(run)
    evidence = _mapping(output.get("evidence_packet")).get("evidence_items")
    if evidence is None:
        evidence = output.get("retrieval_results", [])
    return [str(item.get("document_id")) for item in evidence if isinstance(item, dict) and item.get("document_id")]


def _ranked_chunks(run: Any) -> list[str]:
    output = _outputs(run)
    evidence = _mapping(output.get("evidence_packet")).get("evidence_items")
    if evidence is None:
        evidence = output.get("retrieval_results", [])
    return [
        str(item.get("chunk_id"))
        for item in evidence
        if isinstance(item, dict) and item.get("chunk_id")
    ]


def _evidence_keys(run: Any) -> set[str]:
    evidence = _mapping(_outputs(run).get("evidence_packet")).get("evidence_items", [])
    return {
        str(item.get("chunk_id") or item.get("document_id"))
        for item in evidence
        if isinstance(item, dict) and (item.get("chunk_id") or item.get("document_id"))
    }


def retrieval_recall_at_k(run: Any, example: Any) -> dict[str, Any]:
    expected = {str(item) for item in _reference(example).get("expected_document_ids", [])}
    if not expected:
        return _feedback("retrieval_recall_at_k", 1.0, "no reference documents")
    actual = set(_ranked_documents(run)[:5])
    return _feedback("retrieval_recall_at_k", len(expected & actual) / len(expected))


def retrieval_mrr(run: Any, example: Any) -> dict[str, Any]:
    expected = {str(item) for item in _reference(example).get("expected_document_ids", [])}
    if not expected:
        return _feedback("retrieval_mrr", 1.0, "no reference documents")
    for index, document_id in enumerate(_ranked_documents(run), 1):
        if document_id in expected:
            return _feedback("retrieval_mrr", 1 / index)
    return _feedback("retrieval_mrr", 0.0)


def exact_article_hit(run: Any, example: Any) -> dict[str, Any]:
    expected = {str(item) for item in _reference(example).get("expected_chunk_ids", [])}
    if not expected:
        return _feedback("exact_article_hit", 1.0, "no reference chunks")
    actual = set(_ranked_chunks(run)[:5])
    return _feedback("exact_article_hit", expected <= actual)


def citation_grounding(run: Any, example: Any) -> dict[str, Any]:
    output = _outputs(run)
    evidence_ids = _evidence_keys(run)
    citations = output.get("citations", [])
    citation_ids = {
        str(item.get("chunk_id") or item.get("document_id")) for item in citations
        if isinstance(item, dict) and (item.get("chunk_id") or item.get("document_id"))
    }
    valid = citation_ids <= evidence_ids
    return _feedback("citation_grounding", valid, f"unknown={sorted(citation_ids - evidence_ids)}")


def citation_precision(run: Any, example: Any) -> dict[str, Any]:
    output = _outputs(run)
    evidence_ids = _evidence_keys(run)
    citations = [item for item in output.get("citations", []) if isinstance(item, dict)]
    if not citations:
        return _feedback("citation_precision", 1.0 if not evidence_ids else 0.0)
    valid = sum(
        1 for item in citations
        if str(item.get("chunk_id") or item.get("document_id")) in evidence_ids
    )
    return _feedback("citation_precision", valid / len(citations))


def no_match_safety(run: Any, example: Any) -> dict[str, Any]:
    output = _outputs(run)
    packet = _mapping(output.get("evidence_packet"))
    if packet.get("retrieval_status") != "no_match":
        return _feedback("no_match_safety", 1.0, "not a no-match run")
    answer = str(output.get("final_answer", ""))
    citations = output.get("citations", [])
    has_unverified_law = bool(re.search(r"《[^》]+》|第[零一二三四五六七八九十百千万0-9]+条", answer))
    disclosed = "未检索到" in answer and ("可引用法条" in answer or "可直接引用" in answer)
    return _feedback("no_match_safety", not citations and not has_unverified_law and disclosed)


def tool_trajectory(run: Any, example: Any) -> dict[str, Any]:
    trajectory = _outputs(run).get("tool_trajectory", [])
    valid = all(
        isinstance(item, dict) and item.get("agent") == "legal_researcher"
        for item in trajectory
    )
    return _feedback("tool_trajectory", valid)


def loop_limit(run: Any, example: Any) -> dict[str, Any]:
    output = _outputs(run)
    valid = (
        int(output.get("tool_call_count", 0)) <= 4
        and int(output.get("model_call_count", 0)) <= 10
        and int(output.get("retry_count", 0)) <= 1
        and int(output.get("revision_count", 0)) <= 1
    )
    return _feedback("loop_limit", valid)


def latest_fact_priority(run: Any, example: Any) -> dict[str, Any]:
    reference = _reference(example)
    current = str(reference.get("expected_current_fact", ""))
    old = str(reference.get("forbidden_old_fact", ""))
    answer = str(_outputs(run).get("final_answer", ""))
    valid = (not current or current in answer) and (not old or old not in answer)
    return _feedback("latest_fact_priority", valid)


def tenant_isolation(run: Any, example: Any) -> dict[str, Any]:
    answer = str(_outputs(run).get("final_answer", ""))
    forbidden = [str(item) for item in _reference(example).get("forbidden_terms", [])]
    leaked = [item for item in forbidden if item and item in answer]
    return _feedback("tenant_isolation", not leaked, f"leaked_terms={len(leaked)}")


def completion_success(run: Any, example: Any) -> dict[str, Any]:
    output = _outputs(run)
    valid = bool(str(output.get("final_answer", "")).strip()) and not output.get("errors")
    return _feedback("completion_success", valid)


DETERMINISTIC_EVALUATORS = [
    route_correctness,
    schema_validity,
    retrieval_status_correctness,
    retrieval_recall_at_k,
    retrieval_mrr,
    exact_article_hit,
    citation_grounding,
    citation_precision,
    no_match_safety,
    tool_trajectory,
    loop_limit,
    latest_fact_priority,
    tenant_isolation,
    completion_success,
]
