import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from langsmith import Client, aevaluate, evaluate, tracing_context

from backend.app.core.config import get_settings
from backend.app.core.resource_budget import MonthlyResourceBudget
from backend.app.evaluation import DETERMINISTIC_EVALUATORS
from backend.app.evaluation.judge import LegalQualityJudge
from backend.app.evaluation.profiles import (
    batch_manifest,
    case_hash,
    dataset_sha256,
    effective_seed,
    load_cases,
    select_cases,
    to_examples,
)
from backend.app.evaluation.reporting import ReportRun
from backend.app.evaluation.targets import RetrievalTarget, agent_target

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = {
    "schema_validity": 1.0,
    "citation_grounding": 1.0,
    "no_match_safety": 1.0,
    "tenant_isolation": 1.0,
    "loop_limit": 1.0,
    "route_correctness": 0.95,
    "retrieval_recall_at_k": 0.85,
    "retrieval_mrr": 0.70,
    "exact_article_hit": 0.85,
    "completion_success": 1.0,
    "judge_evidence_consistency": 0.86,
    "judge_factual_fidelity": 0.86,
    "judge_risk_calibration": 0.8,
    "judge_helpfulness": 0.8,
}

REQUIRED_BY_MODE = {
    "retrieval": {
        "retrieval_status_correctness",
        "retrieval_recall_at_k",
        "retrieval_mrr",
        "retrieval_hit_at_1",
        "retrieval_hit_at_3",
        "retrieval_gold_rank",
        "exact_article_hit",
    },
    "component": {
        "route_correctness",
        "schema_validity",
        "citation_grounding",
        "no_match_safety",
        "loop_limit",
        "tenant_isolation",
        "completion_success",
    },
    "live": {
        "route_correctness",
        "schema_validity",
        "retrieval_status_correctness",
        "citation_grounding",
        "no_match_safety",
        "loop_limit",
        "tenant_isolation",
        "completion_success",
    },
}
JUDGE_KEYS = {
    "judge_legal_issue_coverage",
    "judge_evidence_consistency",
    "judge_factual_fidelity",
    "judge_risk_calibration",
    "judge_completeness",
    "judge_actionability",
    "judge_clarity",
    "judge_helpfulness",
}

PROFILE_DEFAULTS = {
    "learn": {
        "dataset": "lawstation-agent-v3",
        "mode": "component",
        "limit": 6,
        "categories": ["casual", "clarification", "matched", "no_match", "tool_error", "memory"],
    },
    "smoke": {
        "dataset": "lawstation-agent-v3",
        "mode": "component",
        "limit": 6,
        "categories": ["casual", "clarification", "matched", "no_match", "tool_error", "memory"],
    },
}


def metadata(
    mode: str, rag_mode: str, review_mode: str, rerank_mode: str = "off"
) -> dict:
    settings = get_settings()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        commit = "unknown"
    manifest_path = Path(settings.index_dir) / "law" / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8")) if manifest_path.is_file() else {}
    return {
        "git_commit": commit,
        "mode": mode,
        "graph_version": "three-agent-v2-chunk-evidence-fast-review",
        "prompt_version": "legal-consultation-v2-no-match-safe",
        "model": settings.deepseek_model,
        "models": settings.deepseek_model,
        "prompts": ["legal-consultation-v2-no-match-safe"],
        "tools": [{"name": "search_laws"}, {"name": "get_law_article"}],
        "embedding_model": settings.embedding_model,
        "law_data_version": manifest.get("fingerprint", "unknown"),
        "rag_mode": rag_mode,
        "rerank_mode": rerank_mode,
        "reranker_model": settings.rag_rerank_model if rerank_mode == "on" else "disabled",
        "review_mode": review_mode,
    }


def client() -> Client:
    settings = get_settings()
    if not settings.langsmith_api_key:
        raise SystemExit("请先在 .env 配置 LANGSMITH_API_KEY")
    return Client(
        api_url=settings.langsmith_endpoint,
        api_key=settings.langsmith_api_key,
        workspace_id=settings.langsmith_workspace_id or None,
    )


def _row_evaluations(row: dict[str, Any]) -> list[Any]:
    return row.get("evaluation_results", {}).get("results", [])


def _row_outputs(row: dict[str, Any]) -> dict[str, Any]:
    run = row.get("run")
    if run is not None:
        return getattr(run, "outputs", {}) or {}
    return row.get("outputs", {}) or {}


def _row_example(row: dict[str, Any]) -> Any | None:
    return row.get("example") or row.get("reference_example")


def _example_metadata(example: Any | None) -> dict[str, Any]:
    if example is None:
        return {}
    value = getattr(example, "metadata", None)
    if isinstance(value, dict):
        return value
    if isinstance(example, dict) and isinstance(example.get("metadata"), dict):
        return example["metadata"]
    return {}


def _example_inputs(example: Any | None) -> dict[str, Any]:
    if example is None:
        return {}
    value = getattr(example, "inputs", None)
    if isinstance(value, dict):
        return value
    if isinstance(example, dict) and isinstance(example.get("inputs"), dict):
        return example["inputs"]
    return {}


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "sample_count": len(values),
        "mean": round(statistics.fmean(values), 6),
        "standard_deviation": round(statistics.pstdev(values), 6) if len(values) > 1 else 0.0,
        "pass_rate": round(sum(value >= 1.0 for value in values) / len(values), 6),
    }


def _metric_summary(key: str, values: list[float]) -> dict[str, Any]:
    result: dict[str, Any] = _summary(values)
    result["direction"] = "lower" if key == "retrieval_gold_rank" else "higher"
    if key == "retrieval_gold_rank":
        result["pass_rate"] = None
    return result


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 6)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 6)


def _project_stats(
    ls_client: Client, experiment_name: str, expected_runs: int
) -> dict[str, Any]:
    if not experiment_name:
        return {}
    project = None
    for _ in range(10):
        project = ls_client.read_project(project_name=experiment_name, include_stats=True)
        if int(getattr(project, "run_count", 0) or 0) >= expected_runs:
            break
        time.sleep(2)
    if project is None:
        return {}
    keys = (
        "id", "name", "run_count", "error_rate", "latency_p50", "latency_p95",
        "latency_p99", "first_token_p50", "first_token_p95", "first_token_p99",
        "total_tokens", "prompt_tokens", "completion_tokens", "total_cost",
        "prompt_cost", "completion_cost",
    )
    result = {}
    for key in keys:
        value = getattr(project, key, None)
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            result[key] = str(value) if key == "id" else value
    return result


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    os.replace(temporary, path)
    csv_path = path.with_suffix(".csv")
    csv_temporary = csv_path.with_suffix(csv_path.suffix + f".{os.getpid()}.tmp")
    with csv_temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "metric", "sample_count", "mean", "standard_deviation", "pass_rate", "direction"
            ],
        )
        writer.writeheader()
        for metric, values in sorted(payload.get("metrics", {}).items()):
            writer.writerow({"metric": metric, **values})
    os.replace(csv_temporary, csv_path)


def _learning_output(case: dict[str, Any]) -> dict[str, Any]:
    inputs = case.get("inputs", {})
    reference = case.get("outputs", {})
    analysis = inputs.get("fixture_case_analysis") or {
        "next_action": reference.get("expected_route"),
    }
    status = reference.get("expected_retrieval_status")
    documents = list(inputs.get("fixture_documents", []))
    packet = None
    if analysis.get("next_action") == "research":
        packet = {"retrieval_status": status or "no_match", "evidence_items": documents}
    if status == "no_match":
        current = reference.get("expected_current_fact")
        prefix = f"已按你最新说明的信息处理：{current}。" if current else ""
        answer = prefix + "本轮法规检索正常完成，但未检索到可引用法条。以下仅提供一般性分析。"
    else:
        answer = "这是用于学习确定性评估器的固定回答。"
    citations = documents[:1] if status == "matched" else []
    return {
        "case_analysis": analysis,
        "evidence_packet": packet,
        "final_answer": answer,
        "citations": citations,
        "tool_trajectory": (
            [{"agent": "legal_researcher", "tool": "search_laws"}]
            if analysis.get("next_action") == "research" else []
        ),
        "tool_call_count": 1 if analysis.get("next_action") == "research" else 0,
        "model_call_count": 0,
        "retry_count": 0,
        "revision_count": 0,
        "errors": [],
    }


def _run_learn(args) -> dict[str, Any]:
    defaults = PROFILE_DEFAULTS["learn"]
    dataset = args.dataset or defaults["dataset"]
    seed = effective_seed(args.sample_seed or 42)
    categories = args.sample_categories or defaults["categories"]
    cases = select_cases(
        load_cases(dataset),
        limit=args.max_examples or defaults["limit"],
        categories=categories,
        seed=seed,
    )
    scores: dict[str, list[float]] = {}
    details = []
    for case in cases:
        run = SimpleNamespace(outputs=_learning_output(case))
        example = SimpleNamespace(inputs=case["inputs"], outputs=case["outputs"])
        evaluations = []
        for evaluator in DETERMINISTIC_EVALUATORS:
            result = evaluator(run, example)
            if result.get("score") is not None:
                scores.setdefault(result["key"], []).append(float(result["score"]))
            evaluations.append(result)
        details.append({
            "content_sha256": case_hash(case),
            "category": case.get("metadata", {}).get("category"),
            "inputs": case.get("inputs", {}),
            "expected": case.get("outputs", {}),
            "actual": run.outputs,
            "evaluations": evaluations,
        })
    payload = {
        "profile": "learn",
        "dataset": dataset,
        "sample_size": len(cases),
        "dataset_sha256": dataset_sha256(dataset),
        "local_reproducible": True,
        "langsmith_witness_complete": False,
        "resume_eligible": False,
        "resource_usage": {
            "planned_traces": 0,
            "uploaded_traces": 0,
            "agent_model_calls": 0,
            "judge_calls": 0,
            "cache_hits": 0,
            "estimated_token_usage": 0,
        },
        "metrics": {key: _metric_summary(key, values) for key, values in scores.items()},
        "batch": batch_manifest(dataset, cases, seed),
        "details": details,
    }
    return payload


def _profile_args(args) -> None:
    defaults = PROFILE_DEFAULTS.get(args.profile)
    if defaults:
        args.dataset = args.dataset or defaults["dataset"]
        args.mode = args.mode or defaults["mode"]
        args.max_examples = args.max_examples or defaults["limit"]
        args.sample_categories = args.sample_categories or defaults["categories"]
    else:
        args.dataset = args.dataset or "lawstation-e2e-v2"
        args.mode = args.mode or "component"
    if args.profile in {"learn", "smoke"}:
        args.upload_results = False
        args.repetitions = 1
        args.concurrency = 1


def _select_examples(args, ls_client: Client | None) -> tuple[list[Any], dict[str, Any]]:
    seed = effective_seed(args.sample_seed)
    cases = select_cases(
        load_cases(args.dataset),
        limit=args.max_examples,
        categories=args.sample_categories,
        seed=seed,
    )
    manifest = batch_manifest(args.dataset, cases, seed)
    if not args.upload_results:
        return to_examples(args.dataset, cases), manifest
    if ls_client is None:
        raise RuntimeError("上传 LangSmith 时缺少 Client")
    selected_hashes = {case_hash(item) for item in cases}
    remote = list(ls_client.list_examples(dataset_name=args.dataset))
    examples = [
        item for item in remote
        if case_hash({"inputs": item.inputs or {}, "outputs": item.outputs or {}}) in selected_hashes
    ]
    if len(examples) != len(cases):
        raise RuntimeError("LangSmith 数据集与本地内容不一致，请先运行 seed_langsmith_datasets.py")
    return examples, manifest


def _budget_plan(args, example_count: int) -> dict[str, int]:
    runs = example_count * args.repetitions
    judged_examples = 0
    if args.judge:
        judged_examples = min(example_count, args.judge_max_examples or example_count)
    return {
        "evaluation_traces": runs if args.upload_results and not args.reuse_experiment else 0,
        "agent_model_calls": (
            runs * get_settings().agent_max_model_calls if args.mode in {"component", "live"}
            and not args.reuse_experiment else 0
        ),
        "judge_calls": judged_examples * args.repetitions,
    }


@contextmanager
def _tracing_disabled():
    previous = {
        key: os.environ.get(key)
        for key in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2")
    }
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    try:
        with tracing_context(enabled=False):
            yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def _formal_eval_cache_disabled():
    """Prevent VCR replay from contaminating uploaded latency/token comparisons."""
    previous = os.environ.pop("LANGSMITH_TEST_CACHE", None)
    try:
        yield
    finally:
        if previous is not None:
            os.environ["LANGSMITH_TEST_CACHE"] = previous


async def _local_evaluate(
    target,
    examples: list[Any],
    evaluators: list[Any],
    *,
    repetitions: int,
    cache_dir: Path,
    cache_metadata: dict[str, Any],
    use_cache: bool = True,
) -> tuple[list[dict[str, Any]], int]:
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    cache_hits = 0
    with _tracing_disabled():
        for _ in range(repetitions):
            for example in examples:
                payload = {
                    "inputs": example.inputs or {},
                    "metadata": cache_metadata,
                }
                digest = hashlib.sha256(
                    json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode()
                ).hexdigest()
                cache_path = cache_dir / f"target-{digest}.json"
                if use_cache and cache_path.is_file():
                    outputs = json.loads(cache_path.read_text("utf-8"))
                    cache_hits += 1
                else:
                    outputs = await target(example.inputs or {})
                    if use_cache:
                        temporary = cache_path.with_suffix(f".{os.getpid()}.tmp")
                        temporary.write_text(
                            json.dumps(outputs, ensure_ascii=False, default=str), encoding="utf-8"
                        )
                        os.replace(temporary, cache_path)
                run = SimpleNamespace(outputs=outputs)
                feedback = []
                for evaluator in evaluators:
                    value = evaluator(run, example)
                    if asyncio.iscoroutine(value):
                        value = await value
                    values = value if isinstance(value, list) else [value]
                    feedback.extend(SimpleNamespace(**item) for item in values)
                rows.append({
                    "run": run,
                    "example": example,
                    "evaluation_results": {"results": feedback},
                })
    return rows, cache_hits


async def run(args) -> dict[str, Any]:
    _profile_args(args)
    if args.profile == "learn":
        return _run_learn(args)
    if args.upload_results and not args.confirm_upload:
        raise SystemExit("上传评测必须显式提供 --confirm-upload")
    if args.reuse_experiment and not args.upload_results:
        raise SystemExit("复用 LangSmith 实验必须同时提供 --upload-results --confirm-upload")
    settings = get_settings()
    if not args.upload_results:
        os.environ.setdefault("LANGSMITH_TEST_CACHE", settings.langsmith_test_cache)
    ls_client = (
        client()
        if (args.upload_results or args.reuse_experiment or args.compare) and not args.plan_only
        else None
    )
    if args.compare:
        if not args.confirm_upload:
            raise SystemExit("比较云端实验必须显式提供 --confirm-upload")
        if ls_client is None:
            raise RuntimeError("比较云端实验时缺少 LangSmith Client")
        result = evaluate(
            (args.compare[0], args.compare[1]),
            data=args.dataset,
            client=ls_client,
        )
        return {"comparison": str(result)}
    evaluators = list(DETERMINISTIC_EVALUATORS)
    judge_calls = 0
    judge_ids: set[str] = set()
    if args.judge and not args.plan_only:
        judge = LegalQualityJudge()

        async def judge_evaluator(run: Any, example: Any):
            nonlocal judge_calls
            if judge_ids and str(getattr(example, "id", "")) not in judge_ids:
                return []
            judge_calls += 1
            return await judge(run, example)

        evaluators.append(judge_evaluator)
    settings = settings.model_copy(update={
        "rag_retrieval_mode": args.rag_mode,
        "rag_rerank_enabled": args.rerank_mode == "on",
        "agent_review_mode": args.review_mode,
    })
    retrieval_target = None
    target = None
    if args.reuse_experiment and args.plan_only:
        if not args.max_examples:
            raise SystemExit("复用实验的 --plan-only 需要通过 --max-examples 指定已有 Run 数")
        example_count = args.max_examples
        data = None
        batch = {"reuse_experiment": args.reuse_experiment, "sample_count": example_count}
    elif args.reuse_experiment:
        if ls_client is None:
            ls_client = client()
        project = ls_client.read_project(project_name=args.reuse_experiment, include_stats=True)
        example_count = int(getattr(project, "run_count", 0) or 0)
        data: Any = None
        batch = {"reuse_experiment": args.reuse_experiment, "sample_count": example_count}
    else:
        upload_for_selection = args.upload_results
        if args.plan_only:
            args.upload_results = False
        data, batch = _select_examples(args, ls_client)
        args.upload_results = upload_for_selection
        example_count = len(data)
        if args.judge:
            judge_limit = min(example_count, args.judge_max_examples or example_count)
            judge_ids = {str(item.id) for item in data[:judge_limit]}
    plan = _budget_plan(args, example_count)
    ledger = MonthlyResourceBudget(settings.eval_resource_budget_path)
    usage_before = ledger.snapshot()
    preview = {
        "profile": args.profile,
        "dataset": args.dataset,
        "sample_count": example_count,
        "repetitions": args.repetitions,
        "upload_results": args.upload_results,
        "planned": plan,
        "monthly_budget_enforced": False,
        "monthly_usage_before": usage_before,
        "batch": batch,
    }
    if args.plan_only:
        return preview
    started_at = datetime.now(UTC)
    success = False
    cache_hits = 0
    try:
        if not args.reuse_experiment:
            if args.mode == "retrieval":
                retrieval_target = await RetrievalTarget.create(settings, args.rag_mode)

                async def target(inputs: dict[str, Any]) -> dict[str, Any]:
                    return await retrieval_target(inputs)
            else:
                target = agent_target(args.mode, args.review_mode, settings)
        if args.upload_results:
            with _formal_eval_cache_disabled(), tracing_context(enabled=True):
                results = await aevaluate(
                    args.reuse_experiment or target,
                    data=data,
                    evaluators=evaluators,
                    metadata=metadata(
                        args.mode, args.rag_mode, args.review_mode, args.rerank_mode
                    ),
                    experiment_prefix=args.experiment or f"{args.dataset}-{args.mode}",
                    max_concurrency=args.concurrency,
                    num_repetitions=args.repetitions,
                    client=ls_client,
                    upload_results=True,
                )
                rows = [row async for row in results]
            experiment_name = str(getattr(results, "experiment_name", "") or "")
        else:
            rows, cache_hits = await _local_evaluate(
                target,
                data,
                evaluators,
                repetitions=args.repetitions,
                cache_dir=Path(settings.langsmith_test_cache),
                cache_metadata=metadata(
                    args.mode, args.rag_mode, args.review_mode, args.rerank_mode
                ),
                use_cache=not getattr(args, "no_test_cache", False),
            )
            experiment_name = ""
        success = True
    finally:
        if retrieval_target is not None:
            await retrieval_target.close()
    scores: dict[str, list[float]] = {}
    category_scores: dict[str, dict[str, list[float]]] = {}
    difficulty_scores: dict[str, dict[str, list[float]]] = {}
    example_results: list[dict[str, Any]] = []
    for row in rows:
        example = _row_example(row)
        example_metadata = _example_metadata(example)
        example_inputs = _example_inputs(example)
        category = str(example_metadata.get("category") or "unknown")
        difficulty = str(example_metadata.get("difficulty") or "unspecified")
        evaluations: dict[str, float] = {}
        for item in _row_evaluations(row):
            score = getattr(item, "score", None)
            key = getattr(item, "key", None)
            if key and score is not None:
                numeric = float(score)
                scores.setdefault(key, []).append(numeric)
                category_scores.setdefault(category, {}).setdefault(key, []).append(numeric)
                difficulty_scores.setdefault(difficulty, {}).setdefault(key, []).append(numeric)
                evaluations[str(key)] = numeric
        outputs = _row_outputs(row)
        example_results.append({
            "content_sha256": example_metadata.get("content_sha256"),
            "category": category,
            "difficulty": difficulty,
            "question": (
                example_inputs.get("question")
                if example_metadata.get("synthetic") or example_metadata.get("source_derived")
                else None
            ),
            "retrieval_status": outputs.get("retrieval_status"),
            "ranked_document_ids": [
                item.get("document_id")
                for item in outputs.get("retrieval_results", [])
                if isinstance(item, dict)
            ],
            "ranked_chunk_ids": [
                item.get("chunk_id")
                for item in outputs.get("retrieval_results", [])
                if isinstance(item, dict)
            ],
            "evaluations": evaluations,
        })
    metrics = {key: _metric_summary(key, values) for key, values in scores.items() if values}
    metrics_by_category = {
        category: {
            key: _metric_summary(key, values) for key, values in grouped.items() if values
        }
        for category, grouped in sorted(category_scores.items())
    }
    metrics_by_difficulty = {
        difficulty: {
            key: _metric_summary(key, values) for key, values in grouped.items() if values
        }
        for difficulty, grouped in sorted(difficulty_scores.items())
    }
    outputs = [_row_outputs(row) for row in rows]
    retrieval_durations = [
        float(item.get("retrieval_duration_ms", 0)) for item in outputs
        if item.get("retrieval_duration_ms") is not None
    ]
    rerank_durations = [
        float(item.get("rerank_duration_ms", 0)) for item in outputs
        if item.get("rerank_applied")
    ]
    rerank_applicable = [
        item for item in outputs if item.get("retrieval_status") == "matched"
    ]
    reranker_model_digests = sorted({
        str(item["reranker_model_digest"])
        for item in outputs
        if item.get("reranker_model_digest")
    })
    ranking_versions = sorted({
        str(item["ranking_version"])
        for item in outputs
        if item.get("ranking_version")
    })
    runtime_metrics = {
        "mean_model_calls": round(statistics.fmean([
            float(item.get("model_call_count", 0)) for item in outputs
        ]), 6) if outputs else 0.0,
        "mean_tool_calls": round(statistics.fmean([
            float(item.get("tool_call_count", 0)) for item in outputs
        ]), 6) if outputs else 0.0,
        "llm_review_rate": round(sum(
            item.get("review_mode") == "llm" for item in outputs
        ) / len(outputs), 6) if outputs else 0.0,
        "mean_retrieval_duration_ms": round(
            statistics.fmean(retrieval_durations), 6
        ) if retrieval_durations else None,
        "retrieval_duration_p50_ms": _percentile(retrieval_durations, 0.5),
        "retrieval_duration_p95_ms": _percentile(retrieval_durations, 0.95),
        "mean_rerank_duration_ms": round(
            statistics.fmean(rerank_durations), 6
        ) if rerank_durations else 0.0,
        "rerank_duration_p50_ms": _percentile(rerank_durations, 0.5),
        "rerank_duration_p95_ms": _percentile(rerank_durations, 0.95),
        "mean_rerank_prompt_tokens": round(statistics.fmean([
            float(item.get("rerank_prompt_tokens", 0)) for item in outputs
        ]), 6) if outputs else 0.0,
        "mean_rerank_candidate_count": round(statistics.fmean([
            float(item.get("rerank_candidate_count", 0)) for item in outputs
        ]), 6) if outputs else 0.0,
        "rerank_applied_rate": round(sum(
            bool(item.get("rerank_applied")) for item in rerank_applicable
        ) / len(rerank_applicable), 6) if rerank_applicable else 1.0,
        "rerank_degraded_rate": round(sum(
            item.get("reranker_status") in {"degraded", "cooldown"}
            for item in outputs
        ) / len(outputs), 6) if outputs else 0.0,
        "reranker_model_digests": reranker_model_digests,
        "ranking_versions": ranking_versions,
    }
    required = set(REQUIRED_BY_MODE[args.mode]) | (JUDGE_KEYS if args.judge else set())
    missing = sorted(required - set(metrics))
    actual_model_calls = sum(int(item.get("model_call_count", 0)) for item in outputs)
    if success:
        ledger.record_many({
            "evaluation_traces": len(rows) if args.upload_results else 0,
            "agent_model_calls": actual_model_calls,
            "judge_calls": judge_calls,
        })
    project_stats = (
        _project_stats(ls_client, experiment_name, len(rows))
        if args.upload_results and ls_client is not None else {}
    )
    witness_complete = bool(
        args.upload_results
        and int(project_stats.get("run_count", 0) or 0) >= len(rows)
    )
    local_reproducible = bool(
        not args.upload_results and args.mode == "retrieval" and not missing
    )
    payload = {
        "profile": args.profile,
        "experiment_name": experiment_name,
        "dataset": args.dataset,
        "mode": args.mode,
        "rag_mode": args.rag_mode,
        "rerank_mode": args.rerank_mode,
        "review_mode": args.review_mode,
        "repetitions": args.repetitions,
        "run_count": len(rows),
        "sample_size": example_count,
        "dataset_sha256": dataset_sha256(args.dataset),
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "metadata": metadata(
            args.mode, args.rag_mode, args.review_mode, args.rerank_mode
        ),
        "metrics": metrics,
        "metrics_by_category": metrics_by_category,
        "metrics_by_difficulty": metrics_by_difficulty,
        "example_results": example_results,
        "runtime_metrics": runtime_metrics,
        "project_stats": project_stats,
        "missing_required_metrics": missing,
        "local_reproducible": local_reproducible,
        "langsmith_witness_complete": witness_complete,
        "resume_eligible": bool(not missing and (local_reproducible or witness_complete)),
        "test_cache_enabled": bool(
            not args.upload_results and not getattr(args, "no_test_cache", False)
        ),
        "batch": batch,
        "resource_usage": {
            "planned_traces": plan["evaluation_traces"],
            "uploaded_traces": len(rows) if args.upload_results else 0,
            "agent_model_calls": actual_model_calls,
            "judge_calls": judge_calls,
            "cache_hits": cache_hits,
            "estimated_token_usage": project_stats.get("total_tokens"),
            "monthly_budget_enforced": False,
            "monthly_usage": ledger.snapshot(),
        },
    }
    payload["langsmith_export_complete"] = witness_complete
    args._result_payload = payload
    if args.fail_on_threshold:
        if missing:
            raise SystemExit("评测门禁未通过：缺少关键指标 " + ", ".join(missing))
        failed = {
            key: {"actual": metrics[key]["mean"], "required": threshold}
            for key, threshold in THRESHOLDS.items()
            if key in required and metrics[key]["mean"] < threshold
        }
        if failed:
            raise SystemExit("评测门禁未通过：" + json.dumps(failed, ensure_ascii=False))
        if args.mode == "retrieval" and args.rerank_mode == "on":
            rerank_failed: dict[str, dict[str, float]] = {}
            if runtime_metrics["rerank_applied_rate"] < 1.0:
                rerank_failed["rerank_applied_rate"] = {
                    "actual": runtime_metrics["rerank_applied_rate"],
                    "required": 1.0,
                }
            if runtime_metrics["rerank_degraded_rate"] > 0:
                rerank_failed["rerank_degraded_rate"] = {
                    "actual": runtime_metrics["rerank_degraded_rate"],
                    "required": 0.0,
                }
            rerank_p95 = runtime_metrics["rerank_duration_p95_ms"]
            if rerank_p95 is not None and rerank_p95 > 3000:
                rerank_failed["rerank_duration_p95_ms"] = {
                    "actual": rerank_p95,
                    "required_max": 3000.0,
                }
            if rerank_failed:
                raise SystemExit(
                    "Reranker 评测门禁未通过："
                    + json.dumps(rerank_failed, ensure_ascii=False)
                )
    if args.require_export and not payload["langsmith_export_complete"]:
        raise SystemExit(
            "LangSmith 导出不完整：实验运行数小于本地结果数，可能已达到 Trace 配额"
        )
    return payload


def _default_report_name(args) -> str:
    if args.report_name:
        return ReportRun.validate_report_name(args.report_name)
    if args.profile == "learn":
        return "learn.json"
    return f"{args.profile}-{args.mode}.json"


async def execute(args) -> dict[str, Any]:
    _profile_args(args)
    if args.run_dir and args.run_id:
        raise SystemExit("--run-dir 与 --run-id 不能同时使用")
    if args.no_report and args.output:
        raise SystemExit("--no-report 与兼容参数 --output 不能同时使用")
    settings = get_settings()
    report_run = None
    report_path = None
    if not args.plan_only and not args.no_report:
        report_run = (
            ReportRun.attach(args.run_dir, settings.eval_report_timezone)
            if args.run_dir
            else ReportRun.create(
                args.report_root or settings.eval_report_root,
                args.profile,
                settings.eval_report_timezone,
                args.run_id,
            )
        )
    try:
        payload = await run(args)
        if report_run is not None:
            report_path = report_run.report_path(_default_report_name(args))
            payload.update({
                "run_id": report_run.run_id,
                "report_path": str(report_path),
                "created_at": report_run.started_at.isoformat(),
                "report_timezone": report_run.timezone.key,
            })
            report_run.write_metrics(report_path.name, payload)
            compatibility_output = ""
            if args.output:
                compatibility_output = str(Path(args.output).resolve())
                _write_report(Path(args.output), payload)
            report_run.complete({"compatibility_output": compatibility_output or None})
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        if report_run is not None:
            print(f"评测结果：{report_run.path}")
            print(f"汇总报告：{report_path}")
        return payload
    except BaseException as exc:
        payload = getattr(args, "_result_payload", None)
        if report_run is not None and payload:
            report_path = report_run.report_path(_default_report_name(args))
            payload.update({
                "run_id": report_run.run_id,
                "report_path": str(report_path),
                "created_at": report_run.started_at.isoformat(),
                "report_timezone": report_run.timezone.key,
            })
            report_run.write_metrics(report_path.name, payload)
        if report_run is not None:
            interrupted = isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError))
            report_run.fail(
                "interrupted" if interrupted else "failed",
                stage=f"{args.profile}.{args.mode}",
                error=exc,
            )
            print(f"评测结果：{report_run.path}")
        raise


def parse_args():
    parser = argparse.ArgumentParser(description="运行 LawStation LangSmith 评测")
    parser.add_argument("--profile", choices=["learn", "smoke", "compare", "release"], default="learn")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--mode", choices=["retrieval", "component", "live"], default=None)
    parser.add_argument("--rag-mode", choices=["bm25", "hybrid"], default="hybrid")
    parser.add_argument("--rerank-mode", choices=["off", "on"], default="off")
    parser.add_argument("--review-mode", choices=["always-llm", "auto"], default="auto")
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--fail-on-threshold", action="store_true")
    parser.add_argument("--experiment", default="")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--judge-max-examples", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--sample-categories", type=lambda value: [item for item in value.split(",") if item], default=[])
    parser.add_argument("--upload-results", action="store_true")
    parser.add_argument("--confirm-upload", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--reuse-experiment", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--report-root", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--report-name", default="")
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--require-export", action="store_true")
    parser.add_argument(
        "--no-test-cache",
        action="store_true",
        help="本地正式性能实验禁用模型/目标响应缓存",
    )
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE", "CURRENT"))
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(execute(parse_args()))
