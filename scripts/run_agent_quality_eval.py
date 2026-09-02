"""Run the three frozen Agent-quality suites and produce resume-safe reports."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import subprocess
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langsmith import Client, aevaluate, tracing_context

from backend.app.core.config import get_settings
from backend.app.evaluation.agent_quality import (
    JUDGE_PROMPT_VERSION,
    JUDGE_SCHEMA_VERSION,
    SUITE_DETERMINISTIC,
    AgentQualityJudge,
)
from backend.app.evaluation.profiles import dataset_sha256
from backend.app.evaluation.reporting import ReportRun
from backend.app.evaluation.targets import agent_target, counsel_quality_target, reviewer_target

ROOT = Path(__file__).resolve().parents[1]
SUITES = {
    "factual-fidelity": ("lawstation-factual-fidelity-v1", 10),
    "reviewer-effectiveness": ("lawstation-reviewer-effectiveness-v1", 12),
    "answer-quality": ("lawstation-answer-quality-v1", 8),
}
LOWER_IS_BETTER = {
    "old_fact_leak_rate", "forbidden_fact_violation_rate",
    "reviewer_false_rejection_rate", "unsafe_draft_escape_rate",
}
REQUIRED = {
    "factual-fidelity": {
        "latest_fact_priority_rate", "current_fact_presence_rate", "old_fact_leak_rate",
        "entity_value_accuracy", "forbidden_fact_violation_rate",
        "judge_factual_fidelity", "judge_fabrication_control",
        "judge_uncertainty_preservation", "judge_context_consistency",
    },
    "reviewer-effectiveness": {
        "reviewer_detection_recall", "reviewer_detection_precision",
        "reviewer_false_rejection_rate", "reviewer_action_accuracy",
        "unsafe_draft_escape_rate", "revision_success_rate", "fast_path_safety_rate",
        "error_category_recall",
        "judge_instruction_specificity", "judge_decision_quality",
    },
    "answer-quality": {
        "citation_grounding", "no_match_safety", "latest_fact_priority",
        "unsupported_claim_control", "quality_gate_passed", "judge_issue_coverage",
        "judge_answer_relevance", "judge_actionability", "judge_risk_calibration",
        "judge_completeness", "judge_clarity", "judge_overclaiming_control",
        "judge_follow_up_quality",
    },
}
THRESHOLDS = {
    "latest_fact_priority_rate": (">=", 1.0),
    "old_fact_leak_rate": ("<=", 0.0),
    "forbidden_fact_violation_rate": ("<=", 0.0),
    "judge_factual_fidelity": (">=", 0.86),
    "judge_fabrication_control": (">=", 0.86),
    "reviewer_detection_recall": (">=", 0.90),
    "reviewer_detection_precision": (">=", 0.90),
    "reviewer_false_rejection_rate": ("<=", 0.10),
    "unsafe_draft_escape_rate": ("<=", 0.0),
    "revision_success_rate": (">=", 0.90),
    "fast_path_safety_rate": (">=", 1.0),
    "judge_issue_coverage": (">=", 0.80),
    "judge_answer_relevance": (">=", 0.86),
    "judge_actionability": (">=", 0.80),
    "judge_risk_calibration": (">=", 0.80),
    "judge_clarity": (">=", 0.80),
    "judge_overclaiming_control": (">=", 0.86),
}


def _row_evaluations(row: dict[str, Any]) -> list[Any]:
    return row.get("evaluation_results", {}).get("results", [])


def _row_outputs(row: dict[str, Any]) -> dict[str, Any]:
    run = row.get("run")
    return (getattr(run, "outputs", {}) or {}) if run is not None else row.get("outputs", {}) or {}


def _project_stats(client: Client, experiment_name: str, expected: int) -> dict[str, Any]:
    project = None
    for _ in range(10):
        project = client.read_project(project_name=experiment_name, include_stats=True)
        if int(getattr(project, "run_count", 0) or 0) >= expected:
            break
        time.sleep(2)
    if project is None:
        return {}
    keys = (
        "id", "name", "run_count", "error_rate", "latency_p50", "latency_p95",
        "latency_p99", "total_tokens", "prompt_tokens", "completion_tokens",
        "total_cost", "prompt_cost", "completion_cost",
    )
    result = {}
    for key in keys:
        value = getattr(project, key, None)
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            result[key] = str(value) if key == "id" else value
    return result


def _client() -> Client:
    settings = get_settings()
    if not settings.langsmith_api_key:
        raise RuntimeError("未配置 LANGSMITH_API_KEY")
    return Client(
        api_url=settings.langsmith_endpoint,
        api_key=settings.langsmith_api_key,
        workspace_id=settings.langsmith_workspace_id or None,
    )


def _load(name: str) -> list[dict[str, Any]]:
    path = ROOT / "evals/datasets" / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]


def _sync_dataset(client: Client, name: str, expected: int) -> None:
    rows = _load(name)
    if len(rows) != expected:
        raise RuntimeError(f"{name} 样本数错误：{len(rows)}/{expected}")
    digest = dataset_sha256(name)
    if client.has_dataset(dataset_name=name):
        remote = client.read_dataset(dataset_name=name)
        remote_digest = (remote.metadata or {}).get("source_sha256")
        if remote_digest != digest:
            raise RuntimeError(f"LangSmith 数据集 {name} 与本地 SHA256 不一致")
        return
    client.create_dataset(
        name,
        description="LawStation 合成分层 Agent 质量评测集；未经律师人工标注",
        metadata={
            "source_sha256": digest,
            "human_verified": False,
            "annotation_method": "llm_judge",
        },
    )
    client.create_examples(dataset_name=name, examples=rows)


def _target(suite: str):
    settings = get_settings()
    if suite == "reviewer-effectiveness":
        return reviewer_target(settings)
    counsel = counsel_quality_target(settings)
    if suite == "factual-fidelity":
        return counsel
    component = agent_target("component", "auto", settings)

    async def answer_target(inputs: dict[str, Any]) -> dict[str, Any]:
        if inputs.get("evaluation_target") == "full-component":
            return await component(inputs)
        return await counsel(inputs)

    return answer_target


def _summary(values: list[float], key: str) -> dict[str, Any]:
    passed = (
        sum(value <= 0 for value in values)
        if key in LOWER_IS_BETTER
        else sum(value >= 1 for value in values)
    )
    return {
        "sample_count": len(values),
        "applicable_sample_count": len(values),
        "mean": round(statistics.fmean(values), 6),
        "standard_deviation": round(statistics.pstdev(values), 6) if len(values) > 1 else 0.0,
        "pass_rate": round(passed / len(values), 6),
        "direction": "lower" if key in LOWER_IS_BETTER else "higher",
    }


@contextmanager
def _formal_eval_cache_disabled():
    """Ensure uploaded runs contain fresh model calls and real latency/token data."""
    previous = os.environ.pop("LANGSMITH_TEST_CACHE", None)
    try:
        yield
    finally:
        if previous is not None:
            os.environ["LANGSMITH_TEST_CACHE"] = previous


def _metadata(suite: str) -> dict[str, Any]:
    settings = get_settings()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        commit = "unknown"
    return {
        "suite": suite,
        "git_commit": commit,
        "graph_version": "three-agent-v4",
        "prompt_version": "legal-consultation-v4-no-runtime-skills",
        "judge_model": settings.langsmith_evaluator_model or settings.deepseek_model,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "judge_schema_version": JUDGE_SCHEMA_VERSION,
        "dataset_sha256": dataset_sha256(SUITES[suite][0]),
        "sample_seed": 42,
        "human_verified": False,
        "annotation_method": "llm_judge",
    }


async def _run_suite(client: Client, suite: str) -> dict[str, Any]:
    dataset, expected = SUITES[suite]
    judge = AgentQualityJudge(suite)
    judge_calls = 0

    async def deterministic(run: Any, example: Any):
        return SUITE_DETERMINISTIC[suite](run, example)

    async def judge_evaluator(run: Any, example: Any):
        nonlocal judge_calls
        judge_calls += 1
        return await judge(run, example)

    started = datetime.now(UTC)
    with _formal_eval_cache_disabled(), tracing_context(enabled=True):
        results = await aevaluate(
            _target(suite),
            data=dataset,
            evaluators=[deterministic, judge_evaluator],
            metadata=_metadata(suite),
            experiment_prefix=dataset,
            max_concurrency=1,
            num_repetitions=1,
            client=client,
            upload_results=True,
        )
        rows = [row async for row in results]
    experiment_name = str(getattr(results, "experiment_name", "") or "")
    values: dict[str, list[float]] = {}
    details = []
    model_calls = 0
    tool_calls = 0
    for row in rows:
        output = _row_outputs(row)
        model_calls += int(output.get("model_call_count", 0) or 0)
        tool_calls += int(output.get("tool_call_count", 0) or 0)
        evaluations = {}
        comments = {}
        for item in _row_evaluations(row):
            key = str(getattr(item, "key", "") or "")
            score = getattr(item, "score", None)
            if key and score is not None:
                values.setdefault(key, []).append(float(score))
                evaluations[key] = float(score)
                comments[key] = str(getattr(item, "comment", "") or "")
        example = row.get("example") or row.get("reference_example")
        details.append({
            "example_id": str(getattr(example, "id", "")),
            "category": (getattr(example, "metadata", {}) or {}).get("category"),
            "evaluations": evaluations,
            "comments": comments,
            "final_answer": output.get("final_answer", ""),
        })
    metrics = {key: _summary(items, key) for key, items in sorted(values.items())}
    missing = sorted(REQUIRED[suite] - set(metrics))
    gates = {}
    for key, (operator, threshold) in THRESHOLDS.items():
        if key not in metrics:
            continue
        actual = metrics[key]["mean"]
        passed = actual >= threshold if operator == ">=" else actual <= threshold
        gates[key] = {"actual": actual, "operator": operator, "threshold": threshold, "passed": passed}
    stats = _project_stats(client, experiment_name, expected)
    witness = int(stats.get("run_count", 0) or 0) >= expected
    return {
        "suite": suite,
        "dataset": dataset,
        "dataset_sha256": dataset_sha256(dataset),
        "sample_size": expected,
        "run_count": len(rows),
        "judge_calls": judge_calls,
        "agent_model_calls": model_calls,
        "tool_calls": tool_calls,
        "experiment_name": experiment_name,
        "langsmith_project_id": stats.get("id"),
        "project_stats": stats,
        "metrics": metrics,
        "thresholds": gates,
        "quality_gate_passed": bool(gates and all(item["passed"] for item in gates.values())),
        "missing_required_metrics": missing,
        "langsmith_witness_complete": witness,
        "resume_eligible": not missing and witness and len(rows) == expected and judge_calls == expected,
        "metadata": _metadata(suite),
        "details": details,
        "started_at": started.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
    }


def _metric(suites: dict[str, Any], suite: str, key: str) -> float | None:
    return suites.get(suite, {}).get("metrics", {}).get(key, {}).get("mean")


def _percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


def _score(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 5:.2f}/5"


def _resume_markdown(suites: dict[str, Any], eligible: bool) -> str:
    limitation = "n=30；合成分层样本；LLM Judge；未经律师人工标注。"
    if not eligible:
        return (
            "# Agent 评测简历摘要\n\n"
            "本次评测结果不完整，不能将数字写入简历。可描述为：\n\n"
            "> 建立覆盖事实忠实度、Reviewer 错误拦截和回答质量的 Agent 分层评测体系，"
            "使用确定性 Evaluator 与结构化 LLM Judge 形成质量回归闭环。\n\n"
            f"限制：{limitation}\n"
        )
    fidelity = _score(_metric(suites, "factual-fidelity", "judge_factual_fidelity"))
    leak = _percent(_metric(suites, "factual-fidelity", "old_fact_leak_rate"))
    recall = _percent(_metric(suites, "reviewer-effectiveness", "reviewer_detection_recall"))
    escape = _percent(_metric(suites, "reviewer-effectiveness", "unsafe_draft_escape_rate"))
    revision = _percent(_metric(suites, "reviewer-effectiveness", "revision_success_rate"))
    coverage = _score(_metric(suites, "answer-quality", "judge_issue_coverage"))
    risk = _score(_metric(suites, "answer-quality", "judge_risk_calibration"))
    gates_passed = all(
        payload.get("quality_gate_passed", False) for payload in suites.values()
    )
    gate_note = (
        "三套预设质量门禁均通过。"
        if gates_passed
        else "实验完整，但部分质量门禁未通过；简历不能宣称整体质量达标。"
    )
    return f"""# Agent 评测简历摘要

## 精简版

> 建立覆盖事实忠实度、Reviewer 错误拦截和回答质量的 Agent 分层评测体系，通过确定性 Evaluator 与专项 LLM Judge 评估最新事实覆盖、错误草稿拦截、争议点覆盖和风险校准。

## 量化版

> 在 n=30 的合成 Agent 分层评测中，专项 Judge 的事实忠实度为 {fidelity}；Reviewer 错误草稿检出率为 {recall}、一次修订成功率为 {revision}；回答争议点覆盖度与风险校准分别为 {coverage}、{risk}。安全前置项单独报告，不使用综合质量分掩盖引用或 no-match 失败。

诊断边界：关键词确定性检查记录的旧事实提及率为 {leak}，其中包含“旧事实已被更正”等否定性披露；Unsafe Draft 关键词提及率为 {escape}。二者是待校准的保守诊断值，不应直接写成语义泄漏率。{gate_note}

## 技术版

> 基于 LangGraph 生产节点构建固定 State/EvidencePacket 的 Component Eval Target，通过错误草稿注入验证 Reviewer 检出、误拒和修订能力，并使用 Fact/Decision/Answer 三类结构化 LLM Judge；以 Dataset SHA256、Graph/Prompt/Judge 版本和 LangSmith Trace 固化实验环境。

限制：{limitation}
"""


def _report_markdown(suites: dict[str, Any], eligible: bool) -> str:
    lines = [
        "# LawStation Agent 三维质量评测", "",
        f"- 完成时间：{datetime.now(UTC).isoformat()}",
        "- 标注方式：LLM Judge", "- 人工验证：否", "- 样本：30 条合成分层样本", "",
        "| 套件 | Run | Judge | Agent 模型调用 | 可写简历 |", "|---|---:|---:|---:|---|",
    ]
    for suite, payload in suites.items():
        lines.append(
            f"| {suite} | {payload['run_count']} | {payload['judge_calls']} | "
            f"{payload['agent_model_calls']} | {'是' if payload['resume_eligible'] else '否'} |"
        )
    lines.extend(["", f"总体 resume_eligible：`{str(eligible).lower()}`", "", "## 指标", ""])
    for suite, payload in suites.items():
        lines.append(f"### {suite}\n")
        stats = payload.get("project_stats", {})
        lines.append(
            "LangSmith 实验统计："
            f"run_count={stats.get('run_count', 'N/A')}，"
            f"latency_p50={stats.get('latency_p50', 'N/A')}，"
            f"latency_p95={stats.get('latency_p95', 'N/A')}，"
            f"total_tokens={stats.get('total_tokens', 'N/A')}，"
            f"total_cost={stats.get('total_cost', 'N/A')}。\n"
        )
        lines.append("| 指标 | 均值 | 适用样本 |\n|---|---:|---:|")
        for key, metric in payload["metrics"].items():
            lines.append(f"| {key} | {metric['mean']:.4f} | {metric['applicable_sample_count']} |")
        lines.append("")
    lines.extend([
        "## 结论边界", "",
        "这些结果适用于固定的合成分层样本和当前 Judge 版本，可用于工程回归与版本比较；",
        "不能解释为律师人工评审、真实用户总体表现或法律结论准确率。", "",
    ])
    return "\n".join(lines)


async def execute(args: argparse.Namespace) -> dict[str, Any]:
    if not args.upload_results or not args.confirm_upload:
        raise SystemExit(
            "正式 Agent 质量评测必须同时提供 --upload-results --confirm-upload"
        )
    settings = get_settings()
    report_run = ReportRun.create(
        settings.eval_report_root, "agent-quality", settings.eval_report_timezone
    )
    results: dict[str, Any] = {}
    try:
        if not settings.deepseek_api_key:
            raise RuntimeError("未配置 DEEPSEEK_API_KEY")
        if not (settings.langsmith_evaluator_api_key or settings.deepseek_api_key):
            raise RuntimeError("未配置 Judge API Key")
        selected = list(SUITES) if args.suite == "all" else [args.suite]
        client = _client()
        # LangSmith 鉴权/Workspace 在任何模型调用前验证。预检失败也会
        # 留下时间戳 manifest，但不会产生任何 Agent/Judge 调用。
        list(client.list_datasets(limit=1))
        for suite in selected:
            name, expected = SUITES[suite]
            _sync_dataset(client, name, expected)
        for suite in selected:
            payload = await _run_suite(client, suite)
            results[suite] = payload
            report_run.write_metrics(f"{suite}.json", payload)
        expected_total = sum(SUITES[item][1] for item in selected)
        eligible = (
            sum(item["run_count"] for item in results.values()) == expected_total
            and sum(item["judge_calls"] for item in results.values()) == expected_total
            and all(item["resume_eligible"] for item in results.values())
        )
        details_path = report_run.write_text(
            "case-details.jsonl",
            "\n".join(
                json.dumps({"suite": suite, **detail}, ensure_ascii=False)
                for suite, payload in results.items() for detail in payload["details"]
            ) + "\n",
        )
        report_run.write_text("AGENT_QUALITY_REPORT.md", _report_markdown(results, eligible))
        resume_path = report_run.write_text("RESUME_SUMMARY.md", _resume_markdown(results, eligible))
        manifest = {
            "status": "completed", "sample_size": expected_total,
            "langsmith_run_count": sum(item["run_count"] for item in results.values()),
            "judge_result_count": sum(item["judge_calls"] for item in results.values()),
            "resume_eligible": eligible,
            "dataset_sha256": {item: dataset_sha256(SUITES[item][0]) for item in selected},
            "case_details": str(details_path), "resume_summary": str(resume_path),
        }
        report_run.complete(manifest)
        return {"run_dir": str(report_run.path), "resume_eligible": eligible, "suites": results}
    except BaseException as exc:
        report_run.fail(
            "failed",
            stage="agent-quality" if results else "preflight",
            error=exc,
            extra={"completed_suites": list(results)},
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 LawStation Agent 三维质量评测")
    parser.add_argument("--suite", choices=["all", *SUITES], default="all")
    parser.add_argument("--upload-results", action="store_true")
    parser.add_argument("--confirm-upload", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(execute(parse_args()))
