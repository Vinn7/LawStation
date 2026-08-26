import json
from types import SimpleNamespace

import pytest

from backend.app.evaluation.challenge_datasets import (
    DEFAULT_RERANK_CANDIDATE_SIZE,
    assert_unchanged_prefix,
    generation_prompt,
    leaks_source_text,
    prepare_source_pack,
    validate_and_build,
    validate_examples_against_chunks,
)
from backend.app.evaluation.evaluators import (
    retrieval_gold_rank,
    retrieval_hit_at_1,
    retrieval_hit_at_3,
)
from scripts import create_resume_challenge_datasets as dataset_builder
from scripts.run_resume_rag_challenge_eval import (
    AGENT_SMOKE_CATEGORIES,
    DENSE_CATEGORIES,
    evaluate_gates,
    resource_plan,
)


def chunks():
    values = []
    for law_index in range(4):
        for article_index in range(8):
            values.append({
                "document_id": f"doc-{law_index}-{article_index}",
                "chunk_id": f"chunk-{law_index}-{article_index}",
                "law_name": f"测试法律{law_index}",
                "article_number": f"第{article_index}条",
                "content": (
                    f"测试主体{law_index}在符合条件{article_index}时，应当履行相应义务，"
                    "否则依法承担对应责任并及时采取补救措施。有关方面还需要保存材料、"
                    "核对事实经过，并在规定流程内完成通知和后续处理。"
                ),
            })
    return values


def test_source_pack_is_deterministic_and_separates_challenges():
    first = prepare_source_pack(chunks(), dense_count=4, rerank_candidate_count=4, seed=42)
    second = prepare_source_pack(chunks(), dense_count=4, rerank_candidate_count=4, seed=42)
    assert first == second
    assert len(first) == 8
    assert {item["target"]["chunk_id"] for item in first[:4]}.isdisjoint(
        {item["target"]["chunk_id"] for item in first[4:]}
    )
    assert all(len(item["distractors"]) >= 2 for item in first[4:])


def test_source_pack_expansion_preserves_existing_tasks_as_prefix():
    original = prepare_source_pack(
        chunks(), dense_count=4, rerank_candidate_count=4, seed=42
    )
    expanded = prepare_source_pack(
        chunks(), dense_count=4, rerank_candidate_count=8, seed=42
    )
    assert original[:4] == expanded[:4]
    assert original[4:] == expanded[4:8]
    assert_unchanged_prefix(original[:4], expanded[:4], label="dense")
    assert_unchanged_prefix(original[4:], expanded[4:], label="reranker")
    dense_ids = {item["target"]["chunk_id"] for item in expanded[:4]}
    reranker_ids = {item["target"]["chunk_id"] for item in expanded[4:]}
    assert dense_ids.isdisjoint(reranker_ids)
    assert len(reranker_ids) == 8


def test_prefix_guard_rejects_changed_or_shrunk_frozen_content():
    with pytest.raises(RuntimeError, match="发生变化"):
        assert_unchanged_prefix([{"id": 1}], [{"id": 2}], label="candidate")
    with pytest.raises(RuntimeError, match="不能缩减"):
        assert_unchanged_prefix([{"id": 1}, {"id": 2}], [{"id": 1}], label="candidate")


def test_prefix_guard_can_ignore_additive_generator_metadata():
    existing = [{"inputs": {"question": "q"}, "metadata": {"task_id": "dense-0001"}}]
    expanded = [
        {
            "inputs": {"question": "q"},
            "metadata": {"task_id": "dense-0001", "generator_model": "gpt-5.6-sol"},
        }
    ]

    assert_unchanged_prefix(
        existing,
        expanded,
        label="dense",
        ignored_metadata_keys={"generator_model"},
    )


def test_prefix_guard_can_preserve_frozen_question_while_checking_gold_fields():
    existing = [
        {
            "inputs": {"question": "旧问题", "top_k": 5},
            "outputs": {"expected_chunk_ids": ["chunk-1"]},
        }
    ]
    recalculated = [
        {
            "inputs": {"question": "盲修复后的新问题", "top_k": 5},
            "outputs": {"expected_chunk_ids": ["chunk-1"]},
        }
    ]

    assert_unchanged_prefix(
        existing,
        recalculated,
        label="reranker",
        ignored_input_keys={"question"},
    )
    recalculated[0]["outputs"]["expected_chunk_ids"] = ["chunk-2"]
    with pytest.raises(RuntimeError, match="发生变化"):
        assert_unchanged_prefix(
            existing,
            recalculated,
            label="reranker",
            ignored_input_keys={"question"},
        )


def test_generated_questions_are_strictly_validated():
    tasks = prepare_source_pack(chunks(), dense_count=1, rerank_candidate_count=1, seed=42)
    valid = [
        {"task_id": tasks[0]["task_id"], "question": "遇到这种责任纠纷时，普通人该怎么处理才稳妥？"},
        {"task_id": tasks[1]["task_id"], "question": "双方适用条件很相似，究竟哪种情况下需要承担后果？"},
    ]
    accepted, rejected = validate_and_build(tasks, valid)
    assert len(accepted) == 2
    assert not rejected
    assert accepted[0]["metadata"]["human_verified"] is False
    assert accepted[1]["metadata"]["qualification_status"] == "pending"
    assert not validate_examples_against_chunks(accepted, chunks())

    leaked = [{
        "task_id": tasks[0]["task_id"],
        "question": f"请问{tasks[0]['target']['law_name']}{tasks[0]['target']['article_number']}怎么规定？",
    }]
    accepted, rejected = validate_and_build(tasks[:1], leaked)
    assert not accepted
    assert {"law_name_leak", "article_number_leak"} <= set(rejected[0]["reasons"])


def test_source_overlap_and_prompt_contract():
    assert leaks_source_text("当事人应当履行相应义务吗", "当事人应当履行相应义务并承担责任")
    assert not leaks_source_text("这种情况下普通人应该怎么办", "当事人应当履行相应义务并承担责任")
    assert '"responses"' in generation_prompt()


def test_ranking_evaluators_report_hit_and_gold_rank():
    run = SimpleNamespace(outputs={
        "retrieval_results": [
            {"document_id": "other"},
            {"document_id": "gold"},
            {"document_id": "third"},
        ]
    })
    example = SimpleNamespace(outputs={"expected_document_ids": ["gold"]})
    assert retrieval_hit_at_1(run, example)["score"] == 0
    assert retrieval_hit_at_3(run, example)["score"] == 1
    assert retrieval_gold_rank(run, example)["score"] == 2


def _report(
    *,
    size: int,
    recall: float = 0.9,
    mrr: float = 0.7,
    hit1: float = 0.6,
    hit3: float = 0.8,
    gold_rank: float = 2.0,
    exact: float = 0.9,
    status: float = 1.0,
    rerank: bool = False,
    categories=(),
):
    return {
        "sample_size": size,
        "dataset_sha256": f"dataset-{size}",
        "resume_eligible": True,
        "missing_required_metrics": [],
        "batch": {
            "examples": [
                {"content_sha256": f"case-{index}"} for index in range(size)
            ]
        },
        "metrics": {
            "retrieval_recall_at_k": {"mean": recall},
            "retrieval_mrr": {"mean": mrr},
            "retrieval_hit_at_1": {"mean": hit1},
            "retrieval_hit_at_3": {"mean": hit3},
            "retrieval_gold_rank": {"mean": gold_rank},
            "exact_article_hit": {"mean": exact},
            "retrieval_status_correctness": {"mean": status},
        },
        "runtime_metrics": {
            "rerank_applied_rate": 1.0 if rerank else 0.0,
            "rerank_degraded_rate": 0.0,
            "rerank_duration_p95_ms": 2500 if rerank else None,
        },
        "metrics_by_category": {category: {} for category in categories},
    }


def _smoke_report():
    safety = {
        key: {"mean": 1.0}
        for key in (
            "route_correctness",
            "schema_validity",
            "citation_grounding",
            "no_match_safety",
            "loop_limit",
            "tenant_isolation",
            "completion_success",
        )
    }
    return {
        "sample_size": 6,
        "missing_required_metrics": [],
        "metrics": safety,
        "example_results": [
            {"category": category} for category in AGENT_SMOKE_CATEGORIES
        ],
    }


def test_resume_quantitative_resource_plan_has_no_cloud_evaluation_usage():
    plan = resource_plan(include_agent_smoke=True)
    assert plan["rag_searches"] == 1200
    assert plan["qualification_searches_max"] == DEFAULT_RERANK_CANDIDATE_SIZE == 600
    assert plan["agent_smoke_examples"] == 6
    assert plan["langsmith_traces"] == 0
    assert plan["judge_calls"] == 0


def test_resume_quantitative_gates_accept_improvements_and_safe_agent():
    reports = {
        "regression-rrf.json": _report(size=100),
        "regression-bge.json": _report(size=100, mrr=0.75, rerank=True),
        "dense-bm25.json": _report(
            size=300, recall=0.7, mrr=0.5, categories=DENSE_CATEGORIES
        ),
        "dense-hybrid.json": _report(
            size=300, recall=0.8, mrr=0.6, categories=DENSE_CATEGORIES
        ),
        "reranker-rrf.json": _report(size=200, mrr=0.5, hit1=0.4, gold_rank=2.5),
        "reranker-bge.json": _report(
            size=200, mrr=0.7, hit1=0.6, gold_rank=1.7, rerank=True
        ),
        "agent-smoke.json": _smoke_report(),
    }
    gates = evaluate_gates(reports, include_agent_smoke=True)
    assert gates == {"passed": True, "failures": [], "warnings": []}


def test_resume_quantitative_gates_reject_rerank_degradation():
    reports = {
        "regression-rrf.json": _report(size=100),
        "regression-bge.json": _report(size=100, rerank=True),
        "dense-bm25.json": _report(
            size=300, recall=0.7, mrr=0.5, categories=DENSE_CATEGORIES
        ),
        "dense-hybrid.json": _report(
            size=300, recall=0.8, mrr=0.6, categories=DENSE_CATEGORIES
        ),
        "reranker-rrf.json": _report(size=200, mrr=0.5, hit1=0.4, gold_rank=2.5),
        "reranker-bge.json": _report(
            size=200, recall=0.8, mrr=0.7, hit1=0.6, gold_rank=1.7, rerank=True
        ),
        "agent-smoke.json": _smoke_report(),
    }
    reports["reranker-bge.json"]["runtime_metrics"]["rerank_degraded_rate"] = 0.01
    gates = evaluate_gates(reports, include_agent_smoke=True)
    assert gates["passed"] is False
    assert any("发生精排降级" in item for item in gates["failures"])


def _qualification_candidates(count):
    return [
        {
            "metadata": {"task_id": f"rerank-{index + 1:04d}"},
        }
        for index in range(count)
    ]


def _qualification_results(count, *, qualified=True):
    return [
        {
            "task_id": f"rerank-{index + 1:04d}",
            "category": "adjacent_article",
            "gold_in_top_12": qualified,
            "hybrid_gold_rank": 2 if qualified else None,
            "retrieved_hard_distractor_count": 2 if qualified else 0,
            "qualified": qualified,
            "rejection_reason": None if qualified else "gold_not_in_hybrid_top12",
        }
        for index in range(count)
    ]


def test_qualification_checkpoint_resumes_only_matching_fingerprint(monkeypatch, tmp_path):
    monkeypatch.setattr(dataset_builder, "WORK", tmp_path)
    candidates = _qualification_candidates(30)
    results = _qualification_results(25)
    configuration = {"candidate_sha256": "sha", "top_k": 12}
    fingerprint = dataset_builder._qualification_fingerprint(configuration)
    dataset_builder._save_qualification_checkpoint(
        results,
        fingerprint=fingerprint,
        configuration=configuration,
        total_count=30,
    )
    assert dataset_builder._load_qualification_checkpoint(
        candidates, fingerprint=fingerprint
    ) == results
    assert dataset_builder._load_qualification_checkpoint(
        candidates, fingerprint="changed"
    ) == []


def test_qualification_configuration_explicitly_disables_reranking():
    settings = SimpleNamespace(
        rag_bm25_min_score=0.01,
        rag_dense_min_score=0.2,
        rag_rrf_min_score=0.01,
    )
    configuration = dataset_builder._qualification_configuration(
        settings, candidate_sha256="candidate", index_fingerprint="index"
    )
    assert configuration["retrieval_mode"] == "hybrid"
    assert configuration["rerank_enabled"] is False
    assert configuration["top_k"] == 12
    assert configuration["required_distractors"] == 2


def test_qualification_requires_200_and_selects_frozen_order():
    candidates = _qualification_candidates(201)
    with pytest.raises(ValueError, match="199/200"):
        dataset_builder._select_qualified_candidates(
            candidates[:199], _qualification_results(199), requested_size=200
        )

    selected = dataset_builder._select_qualified_candidates(
        candidates, _qualification_results(201), requested_size=200
    )
    assert len(selected) == 200
    assert selected[0]["metadata"]["task_id"] == "rerank-0001"
    assert selected[-1]["metadata"]["task_id"] == "rerank-0200"
    assert selected[0]["metadata"]["qualification_status"] == "qualified"


def test_existing_codex_batch_is_reused_without_invocation(monkeypatch, tmp_path):
    monkeypatch.setattr(dataset_builder, "WORK", tmp_path)
    task = prepare_source_pack(
        chunks(), dense_count=1, rerank_candidate_count=0, seed=42
    )[0]
    response = {
        "task_id": task["task_id"],
        "question": "普通人遇到这种责任纠纷时应当如何妥善处理？",
    }
    (tmp_path / "batches").mkdir()
    (tmp_path / "codex-responses").mkdir()
    (tmp_path / "batches" / "batch-01.json").write_text(
        json.dumps([task], ensure_ascii=False), encoding="utf-8"
    )
    (tmp_path / "source-pack.jsonl").write_text(
        json.dumps(task, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (tmp_path / "response-schema.json").write_text("{}", encoding="utf-8")
    (tmp_path / "codex-responses" / "batch-01.json").write_text(
        json.dumps({"responses": [response]}, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(dataset_builder.shutil, "which", lambda _command: "/bin/codex")

    def unexpected_call(*_args, **_kwargs):
        raise AssertionError("existing response must not invoke Codex")

    monkeypatch.setattr(dataset_builder.subprocess, "run", unexpected_call)
    dataset_builder.generate_codex(SimpleNamespace(
        codex_command="codex",
        model="",
        force=False,
        concurrency=1,
        repair_rounds=0,
        repair_batch_size=25,
    ))
    assert (tmp_path / "responses.jsonl").is_file()
