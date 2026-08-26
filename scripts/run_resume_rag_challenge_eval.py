"""Run the frozen resume-oriented RAG ablation suite locally."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from scripts.run_staged_langsmith_eval import compare, write_comparison
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repository root.
    from run_staged_langsmith_eval import compare, write_comparison

from backend.app.core.config import get_settings
from backend.app.core.ollama import OllamaProcessManager
from backend.app.core.tei import TEIRerankerProcessManager
from backend.app.evaluation.challenge_datasets import (
    DEFAULT_RERANK_CANDIDATE_SIZE,
    DEFAULT_RERANK_SIZE,
    DENSE_DATASET,
    RERANK_CANDIDATE_DATASET,
    RERANK_DATASET,
    load_jsonl,
)
from backend.app.evaluation.reporting import ReportRun

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "evals" / "datasets"

EXPERIMENTS = (
    ("regression-rrf.json", "lawstation-live-retrieval-v1", 100, "hybrid", "off"),
    ("regression-bge.json", "lawstation-live-retrieval-v1", 100, "hybrid", "on"),
    ("dense-bm25.json", DENSE_DATASET, 300, "bm25", "off"),
    ("dense-hybrid.json", DENSE_DATASET, 300, "hybrid", "off"),
    ("reranker-rrf.json", RERANK_DATASET, 200, "hybrid", "off"),
    ("reranker-bge.json", RERANK_DATASET, 200, "hybrid", "on"),
)

UNIT_TESTS = (
    "tests/test_challenge_datasets.py",
    "tests/test_reranker.py",
    "tests/test_tei_manager.py",
    "tests/test_embeddings.py",
    "tests/test_index_manager.py",
    "tests/test_eval_reporting.py",
)
AGENT_SMOKE_CATEGORIES = (
    "casual",
    "clarification",
    "matched",
    "no_match",
    "tool_error",
    "memory",
)
DENSE_CATEGORIES = (
    "semantic_paraphrase",
    "everyday_scenario",
    "consequence_only",
    "colloquial_noise",
)


class ExperimentGateError(RuntimeError):
    """Raised after all reports have been preserved but a release gate failed."""


def _dataset_status() -> list[dict]:
    values = []
    candidate_path = DATASETS / f"{RERANK_CANDIDATE_DATASET}.jsonl"
    candidate_sha = (
        hashlib.sha256(candidate_path.read_bytes()).hexdigest()
        if candidate_path.is_file()
        else ""
    )
    for name, expected in (
        ("lawstation-live-retrieval-v1", 100),
        (DENSE_DATASET, 300),
        (RERANK_CANDIDATE_DATASET, DEFAULT_RERANK_CANDIDATE_SIZE),
        (RERANK_DATASET, DEFAULT_RERANK_SIZE),
    ):
        path = DATASETS / f"{name}.jsonl"
        actual = len(load_jsonl(path)) if path.is_file() else 0
        manifest_path = path.with_suffix(".manifest.json")
        manifest_valid = True
        if name in {DENSE_DATASET, RERANK_CANDIDATE_DATASET, RERANK_DATASET}:
            if not manifest_path.is_file() or not path.is_file():
                manifest_valid = False
            else:
                manifest = json.loads(manifest_path.read_text("utf-8"))
                manifest_valid = manifest.get("dataset_sha256") == hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                if name == RERANK_CANDIDATE_DATASET:
                    manifest_valid = manifest_valid and (
                        manifest.get("expanded_candidate_count") == expected
                    )
                elif name == RERANK_DATASET:
                    manifest_valid = manifest_valid and (
                        manifest.get("candidate_dataset_sha256") == candidate_sha
                        and manifest.get("candidate_count")
                        == DEFAULT_RERANK_CANDIDATE_SIZE
                        and manifest.get("rerank_enabled_during_qualification") is False
                    )
        values.append({
            "dataset": name,
            "role": (
                "candidate"
                if name == RERANK_CANDIDATE_DATASET
                else ("final" if name == RERANK_DATASET else "evaluation")
            ),
            "expected": expected,
            "actual": actual,
            "manifest_valid": manifest_valid,
            "ready": actual == expected and manifest_valid,
        })
    return values


def _command(report_run: ReportRun, experiment: tuple[str, str, int, str, str]) -> list[str]:
    report, dataset, size, rag_mode, rerank_mode = experiment
    return [
        sys.executable,
        "scripts/run_langsmith_eval.py",
        "--profile", "compare",
        "--dataset", dataset,
        "--mode", "retrieval",
        "--rag-mode", rag_mode,
        "--rerank-mode", rerank_mode,
        "--max-examples", str(size),
        "--sample-seed", "42",
        "--repetitions", "1",
        "--run-dir", str(report_run.path),
        "--report-name", report,
        "--no-test-cache",
    ]


def _agent_smoke_command(report_run: ReportRun) -> list[str]:
    return [
        sys.executable,
        "scripts/run_langsmith_eval.py",
        "--profile", "smoke",
        "--dataset", "lawstation-agent-v3",
        "--mode", "component",
        "--review-mode", "auto",
        "--max-examples", "6",
        "--sample-categories", ",".join(AGENT_SMOKE_CATEGORIES),
        "--sample-seed", "42",
        "--repetitions", "1",
        "--run-dir", str(report_run.path),
        "--report-name", "agent-smoke.json",
        "--no-test-cache",
    ]


def resource_plan(*, include_agent_smoke: bool = True) -> dict[str, Any]:
    return {
        "qualification_searches_max": DEFAULT_RERANK_CANDIDATE_SIZE,
        "rag_searches": sum(item[2] for item in EXPERIMENTS),
        "rerank_requests_max": 300,
        "agent_smoke_examples": 6 if include_agent_smoke else 0,
        "agent_model_calls_upper_bound": 36 if include_agent_smoke else 0,
        "langsmith_traces": 0,
        "judge_calls": 0,
    }


def _run_checked(command: list[str], env: dict[str, str] | None = None) -> None:
    print("执行：" + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def _record_report_artifacts(report_run: ReportRun, report_name: str) -> None:
    report = report_run.path / report_name
    for path in (report, report.with_suffix(".csv")):
        if path.is_file():
            report_run.record_artifact(path)


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=True
    )
    return completed.stdout.strip()


def _snapshot(settings, statuses: list[dict], *, unit_tests_run: bool) -> dict[str, Any]:
    source = Path(settings.law_data_path)
    if not source.is_absolute():
        source = ROOT / source
    index_manifest_path = Path(settings.index_dir) / "law" / "manifest.json"
    if not index_manifest_path.is_absolute():
        index_manifest_path = ROOT / index_manifest_path
    index_manifest = (
        json.loads(index_manifest_path.read_text("utf-8"))
        if index_manifest_path.is_file()
        else {}
    )
    return {
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_dirty": bool(_git_output("status", "--short")),
        "source_file": source.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "index_fingerprint": index_manifest.get("fingerprint"),
        "embedding_model": settings.embedding_model,
        "embedding_dimension": settings.embedding_dimension,
        "embedding_model_digest": index_manifest.get("embedding_model_digest"),
        "reranker_model": settings.rag_rerank_model,
        "reranker_revision": settings.rag_rerank_model_revision,
        "reranker_candidate_count": settings.rag_rerank_candidate_count,
        "sample_seed": 42,
        "unit_tests_run": unit_tests_run,
        "unit_tests": list(UNIT_TESTS),
        "datasets": statuses,
    }


def _metric(report: dict[str, Any], key: str) -> float:
    value = report.get("metrics", {}).get(key, {}).get("mean")
    if value is None:
        raise ExperimentGateError(f"报告缺少关键指标：{key}")
    return float(value)


def _validate_pair(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    expected_size: int,
) -> list[str]:
    failures: list[str] = []
    for label, report in (("baseline", baseline), ("candidate", candidate)):
        if report.get("sample_size") != expected_size:
            failures.append(
                f"{label} sample_size={report.get('sample_size')}，期望 {expected_size}"
            )
        missing = report.get("missing_required_metrics", [])
        if missing:
            failures.append(f"{label} 缺少指标：{','.join(missing)}")
        if report.get("resume_eligible") is not True:
            failures.append(f"{label} resume_eligible=false")
    if baseline.get("dataset_sha256") != candidate.get("dataset_sha256"):
        failures.append("Baseline/Candidate dataset_sha256 不一致")
    baseline_hashes = [
        item.get("content_sha256")
        for item in baseline.get("batch", {}).get("examples", [])
    ]
    candidate_hashes = [
        item.get("content_sha256")
        for item in candidate.get("batch", {}).get("examples", [])
    ]
    if baseline_hashes != candidate_hashes:
        failures.append("Baseline/Candidate 样本内容或顺序不一致")
    return failures


def evaluate_gates(
    reports: dict[str, dict[str, Any]],
    *,
    include_agent_smoke: bool,
) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    pairs = {
        "regression": ("regression-rrf.json", "regression-bge.json", 100),
        "dense": ("dense-bm25.json", "dense-hybrid.json", 300),
        "reranker": ("reranker-rrf.json", "reranker-bge.json", 200),
    }
    for name, (baseline_name, candidate_name, size) in pairs.items():
        failures.extend(
            f"{name}: {message}"
            for message in _validate_pair(
                reports[baseline_name], reports[candidate_name], expected_size=size
            )
        )

    regression_base = reports["regression-rrf.json"]
    regression_candidate = reports["regression-bge.json"]
    for key in (
        "retrieval_recall_at_k",
        "exact_article_hit",
        "retrieval_status_correctness",
    ):
        if _metric(regression_candidate, key) < _metric(regression_base, key):
            failures.append(f"regression: {key} 发生回退")

    dense_base = reports["dense-bm25.json"]
    dense_candidate = reports["dense-hybrid.json"]
    for label, report in (("BM25", dense_base), ("Hybrid", dense_candidate)):
        actual_groups = set(report.get("metrics_by_category", {}))
        if actual_groups != set(DENSE_CATEGORIES):
            failures.append(f"dense: {label} 未完整输出四类分组指标")
    if _metric(dense_candidate, "retrieval_recall_at_k") < _metric(
        dense_base, "retrieval_recall_at_k"
    ):
        failures.append("dense: Hybrid Recall@5 低于 BM25")
    if not any(
        _metric(dense_candidate, key) > _metric(dense_base, key)
        for key in ("retrieval_mrr", "retrieval_hit_at_1")
    ):
        failures.append("dense: MRR 与 Hit@1 均未提升")

    rerank_base = reports["reranker-rrf.json"]
    rerank_candidate = reports["reranker-bge.json"]
    for key in ("retrieval_recall_at_k", "exact_article_hit", "retrieval_hit_at_3"):
        if _metric(rerank_candidate, key) < _metric(rerank_base, key):
            failures.append(f"reranker: {key} 发生回退")
    if not any(
        _metric(rerank_candidate, key) > _metric(rerank_base, key)
        for key in ("retrieval_mrr", "retrieval_hit_at_1")
    ):
        failures.append("reranker: MRR 与 Hit@1 均未提升")
    if _metric(rerank_candidate, "retrieval_gold_rank") >= _metric(
        rerank_base, "retrieval_gold_rank"
    ):
        failures.append("reranker: Gold 平均排名未改善")
    for label, report in (
        ("regression", regression_candidate),
        ("reranker", rerank_candidate),
    ):
        candidate_runtime = report.get("runtime_metrics", {})
        if float(candidate_runtime.get("rerank_applied_rate", 0)) != 1.0:
            failures.append(f"{label}: rerank_applied_rate 不是 100%")
        if float(candidate_runtime.get("rerank_degraded_rate", 1)) != 0.0:
            failures.append(f"{label}: 发生精排降级")
    runtime = rerank_candidate.get("runtime_metrics", {})
    p95 = runtime.get("rerank_duration_p95_ms")
    if p95 is not None and float(p95) > 3000:
        warnings.append(f"reranker: p95={float(p95):.2f}ms，超过 3000ms 性能目标")

    if include_agent_smoke:
        smoke = reports["agent-smoke.json"]
        if smoke.get("sample_size") != 6:
            failures.append(f"agent: sample_size={smoke.get('sample_size')}，期望 6")
        if smoke.get("missing_required_metrics"):
            failures.append("agent: 缺少确定性指标")
        actual_categories = {
            item.get("category") for item in smoke.get("example_results", [])
        }
        if actual_categories != set(AGENT_SMOKE_CATEGORIES):
            failures.append("agent: 六类分层样本未完整覆盖")
        for key in (
            "route_correctness",
            "schema_validity",
            "citation_grounding",
            "no_match_safety",
            "loop_limit",
            "tenant_isolation",
            "completion_success",
        ):
            if _metric(smoke, key) != 1.0:
                failures.append(f"agent: {key} 未达到 100%")

    return {
        "passed": not failures,
        "failures": failures,
        "warnings": warnings,
    }


def _percent(value: Any) -> str:
    return "N/A" if value is None else f"{float(value) * 100:.2f}%"


def _number(value: Any, digits: int = 4) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def _markdown(
    comparisons: dict[str, dict],
    reports: dict[str, dict[str, Any]],
    gates: dict[str, Any],
    *,
    include_agent_smoke: bool,
) -> str:
    lines = [
        "# LawStation 本地 RAG 与 Agent 量化评测",
        "",
        "> 定向合成挑战集，不代表真实用户总体准确率；所有数字来自冻结数据集实测。",
        "",
        "## 验收门禁",
        "",
        f"- 总体状态：{'通过' if gates['passed'] else '未通过'}",
        f"- 失败项：{len(gates['failures'])}",
        f"- 性能告警：{len(gates['warnings'])}",
        "",
    ]
    lines.extend(f"- 失败：{item}" for item in gates["failures"])
    lines.extend(f"- 告警：{item}" for item in gates["warnings"])
    if gates["failures"] or gates["warnings"]:
        lines.append("")
    for name, comparison in comparisons.items():
        lines.extend([
            f"## {name}", "",
            "| Metric | Baseline | Candidate | Absolute | Relative | Direction | Improvement |",
            "|---|---:|---:|---:|---:|---|---:|",
        ])
        for metric, values in comparison.get("metrics", {}).items():
            lines.append(
                f"| {metric} | {values.get('baseline')} | {values.get('candidate')} | "
                f"{values.get('absolute_delta')} | {values.get('relative_delta')} | "
                f"{values.get('direction')} | {values.get('improvement_delta')} |"
            )
        lines.append("")
    if include_agent_smoke:
        smoke = reports["agent-smoke.json"]
        lines.extend([
            "## Agent 分层冒烟", "",
            f"- 样本数：{smoke.get('sample_size')}",
            f"- 平均模型调用数：{smoke.get('runtime_metrics', {}).get('mean_model_calls')}",
            f"- 平均工具调用数：{smoke.get('runtime_metrics', {}).get('mean_tool_calls')}",
            f"- LLM Reviewer 调用率：{_percent(smoke.get('runtime_metrics', {}).get('llm_review_rate'))}",
            "",
            "该结果用于证明工程安全约束，不表示法律回答专业准确率。",
            "",
        ])

    dense_base = reports["dense-bm25.json"]
    dense_candidate = reports["dense-hybrid.json"]
    rerank_base = reports["reranker-rrf.json"]
    rerank_candidate = reports["reranker-bge.json"]
    regression_candidate = reports["regression-bge.json"]
    lines.extend([
        "## 可用于简历的实测表述", "",
        (
            "- 在 300 条 Codex 生成、真实法规 ID 约束的合成语义挑战集上，"
            f"BM25 与 Hybrid 的 Recall@5 分别为 {_percent(_metric(dense_base, 'retrieval_recall_at_k'))} "
            f"和 {_percent(_metric(dense_candidate, 'retrieval_recall_at_k'))}，MRR 从 "
            f"{_number(_metric(dense_base, 'retrieval_mrr'))} 变化至 "
            f"{_number(_metric(dense_candidate, 'retrieval_mrr'))}。"
        ),
        (
            "- 使用 TEI 部署 BAAI/bge-reranker-v2-m3，对 Hybrid Top 12 执行 Cross-Encoder 精排；"
            f"在 200 条合成排序挑战集上，Hit@1 从 "
            f"{_percent(_metric(rerank_base, 'retrieval_hit_at_1'))} 变化至 "
            f"{_percent(_metric(rerank_candidate, 'retrieval_hit_at_1'))}，MRR 从 "
            f"{_number(_metric(rerank_base, 'retrieval_mrr'))} 变化至 "
            f"{_number(_metric(rerank_candidate, 'retrieval_mrr'))}，Gold 平均排名从 "
            f"{_number(_metric(rerank_base, 'retrieval_gold_rank'), 2)} 变化至 "
            f"{_number(_metric(rerank_candidate, 'retrieval_gold_rank'), 2)}。"
        ),
        (
            "- 在 100 条源数据派生通用回归集上，BGE Candidate 的 Recall@5 为 "
            f"{_percent(_metric(regression_candidate, 'retrieval_recall_at_k'))}，"
            f"Exact Article Hit 为 {_percent(_metric(regression_candidate, 'exact_article_hit'))}。"
        ),
        "- 建立 Dataset SHA256、索引指纹、模型 revision、Git Commit 与时间戳报告共同约束的可复现实验流程。",
        "",
        "## 不能据此得出的结论", "",
        "- 数据集不是律师人工标注，也不代表真实用户总体准确率。",
        "- Agent 冒烟只能证明测试覆盖的安全约束，不能证明法律结论准确率。",
        "- 如果 Recall@5 未提升而 MRR 提升，只能表述为排序改善。",
        "- 门禁目标、候选集成绩和降级后的 RRF 结果不能冒充正式 BGE 实测值。",
        "",
    ])
    return "\n".join(lines)


def run(args) -> None:
    statuses = _dataset_status()
    if args.plan_only:
        final_status = next(
            item for item in statuses if item["dataset"] == RERANK_DATASET
        )
        print(json.dumps({
            "datasets": statuses,
            "experiments": [
                {"report": item[0], "dataset": item[1], "sample_size": item[2], "rag_mode": item[3], "rerank_mode": item[4]}
                for item in EXPERIMENTS
            ],
            "agent_smoke": not args.skip_agent_smoke,
            "unit_tests": [] if args.skip_unit_tests else list(UNIT_TESTS),
            "auto_qualify_reranker": not args.no_auto_qualify,
            "qualification_required": not final_status["ready"],
            "resources": resource_plan(include_agent_smoke=not args.skip_agent_smoke),
            "external_calls": (
                "local Ollama + local TEI; Agent smoke additionally calls DeepSeek; "
                "no LangSmith upload or Judge"
            ),
        }, ensure_ascii=False, indent=2))
        return

    settings = get_settings()
    report_run = ReportRun.create(
        settings.eval_report_root, "compare", settings.eval_report_timezone
    )
    ollama = OllamaProcessManager(settings)
    tei = TEIRerankerProcessManager(settings)
    try:
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        if not args.skip_unit_tests:
            _run_checked([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *UNIT_TESTS], env)
        _run_checked([
            sys.executable,
            "scripts/create_resume_challenge_datasets.py",
            "validate",
        ], env)
        statuses = _dataset_status()
        candidate_status = next(
            item for item in statuses
            if item["dataset"] == RERANK_CANDIDATE_DATASET
        )
        if not candidate_status["ready"]:
            raise RuntimeError(
                "Reranker 候选池必须先扩容并冻结为600条；请依次运行："
                "prepare --dense-size 300 --rerank-candidate-size 600 --seed 42、"
                "generate-codex、build、validate"
            )
        ollama.ensure_ready()
        runtime = tei.ensure_ready()
        if not runtime.available:
            raise RuntimeError("正式 BGE 消融要求 TEI ready，不允许降级")

        statuses = _dataset_status()
        reranker_status = next(
            item for item in statuses if item["dataset"] == RERANK_DATASET
        )
        if not reranker_status["ready"] and not args.no_auto_qualify:
            _run_checked([
                sys.executable,
                "scripts/create_resume_challenge_datasets.py",
                "qualify-reranker",
                "--rerank-size", "200",
            ], env)
            statuses = _dataset_status()
        if not all(item["ready"] for item in statuses):
            raise RuntimeError(
                "挑战集未冻结或数量不正确；不得降低资格规则，请检查数据状态"
            )

        snapshot = _snapshot(
            settings, statuses, unit_tests_run=not args.skip_unit_tests
        )
        report_run.write_json("experiment-snapshot.json", snapshot)
        for experiment in EXPERIMENTS:
            _run_checked(_command(report_run, experiment), env)
            _record_report_artifacts(report_run, experiment[0])

        if not args.skip_agent_smoke:
            _run_checked(_agent_smoke_command(report_run), env)
            _record_report_artifacts(report_run, "agent-smoke.json")

        reports = {
            name: json.loads((report_run.path / name).read_text("utf-8"))
            for name, *_ in EXPERIMENTS
        }
        if not args.skip_agent_smoke:
            reports["agent-smoke.json"] = json.loads(
                (report_run.path / "agent-smoke.json").read_text("utf-8")
            )
        comparisons = {
            "通用回归：RRF vs BGE": compare(reports["regression-rrf.json"], reports["regression-bge.json"]),
            "Dense：BM25 vs Hybrid": compare(reports["dense-bm25.json"], reports["dense-hybrid.json"]),
            "Reranker：RRF vs BGE": compare(reports["reranker-rrf.json"], reports["reranker-bge.json"]),
        }
        for index, (name, value) in enumerate(comparisons.items(), 1):
            write_comparison(report_run, f"comparison-{index}.json", {"title": name, **value})
        gates = evaluate_gates(
            reports, include_agent_smoke=not args.skip_agent_smoke
        )
        report_run.write_json("suite-gates.json", gates)
        report_run.write_text(
            "EVAL_REPORT.md",
            _markdown(
                comparisons,
                reports,
                gates,
                include_agent_smoke=not args.skip_agent_smoke,
            ),
        )
        if not gates["passed"]:
            raise ExperimentGateError("；".join(gates["failures"]))
        report_run.complete({
            "suite": "resume-rag-agent-quantitative",
            "dataset_status": statuses,
            "resource_plan": resource_plan(
                include_agent_smoke=not args.skip_agent_smoke
            ),
            "gates": gates,
        })
        print(f"评测结果：{report_run.path}")
    except BaseException as exc:
        report_run.fail("failed", stage="resume-rag-challenges", error=exc)
        raise
    finally:
        tei.close()
        ollama.close()


def parse_args():
    parser = argparse.ArgumentParser(description="运行简历导向的本地 RAG 与 Agent 量化评测")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--skip-unit-tests", action="store_true")
    parser.add_argument("--skip-agent-smoke", action="store_true")
    parser.add_argument(
        "--no-auto-qualify",
        action="store_true",
        help="最终200条精排集缺失时不执行 Hybrid 资格冻结",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
