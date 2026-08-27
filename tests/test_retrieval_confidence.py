from mcp_servers.law_rag.confidence import RetrievalConfidenceGate


def test_confidence_gate_separates_candidates_from_no_match(tmp_path):
    gate = RetrievalConfidenceGate(tmp_path / "missing.json")
    strong = [{
        "content": "劳动合同解除应当依法支付经济补偿",
        "retrieval_sources": ["bm25", "dense"],
        "retrieval_scores": {"bm25": 8.0, "dense": 0.72, "rrf": 0.032, "rerank": 0.9},
    }]
    weak = [{
        "content": "完全无关内容",
        "retrieval_sources": ["bm25"],
        "retrieval_scores": {"bm25": 0.02, "rrf": 0.016},
    }]

    assert gate.evaluate("解除劳动合同补偿", strong, ["解除", "劳动合同", "补偿"]).matched
    rejected = gate.evaluate("解除劳动合同补偿", weak, ["解除", "劳动合同", "补偿"])
    assert not rejected.matched
    assert rejected.reason == "candidate_confidence_below_gate"


def test_empty_retrieval_is_normal_no_match(tmp_path):
    gate = RetrievalConfidenceGate(tmp_path / "missing.json")
    decision = gate.evaluate("任意问题", [], ["任意", "问题"])

    assert decision.confidence == 0
    assert not decision.matched
    assert decision.reason == "no_candidates_above_retrieval_thresholds"
