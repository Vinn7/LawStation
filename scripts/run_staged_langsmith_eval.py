"""Run guarded LawStation LangSmith ablations and produce local comparison reports."""

import argparse
import csv
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from langsmith import Client

from backend.app.core.config import get_settings
from backend.app.core.ollama import OllamaProcessManager
from backend.app.evaluation.profiles import (
    case_hash,
    dataset_sha256,
    effective_seed,
    load_cases,
    select_cases,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "evals" / "reports"
PYTHON = sys.executable


def run_checked(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def load_report(name: str) -> dict[str, Any]:
    return json.loads((REPORTS / name).read_text("utf-8"))


def reusable_report(
    name: str,
    *,
    dataset: str,
    runs: int,
    categories: list[str],
    seed: int,
) -> dict[str, Any] | None:
    path = REPORTS / name
    if not path.is_file():
        return None
    report = load_report(name)
    resolved_seed = effective_seed(seed)
    expected = select_cases(
        load_cases(dataset), limit=runs, categories=categories, seed=resolved_seed
    )
    expected_hashes = [case_hash(item) for item in expected]
    actual_hashes = [
        item.get("content_sha256") for item in report.get("batch", {}).get("examples", [])
    ]
    if (
        report.get("dataset") == dataset
        and int(report.get("run_count", 0)) == runs
        and report.get("dataset_sha256") == dataset_sha256(dataset)
        and report.get("batch", {}).get("seed") == resolved_seed
        and actual_hashes == expected_hashes
        and not report.get("missing_required_metrics")
        and report.get("langsmith_export_complete") is True
    ):
        print(f"复用已完成实验：{report.get('experiment_name')}", flush=True)
        return report
    return None


def metric(report: dict[str, Any], key: str) -> float:
    value = report.get("metrics", {}).get(key, {}).get("mean")
    if value is None:
        raise RuntimeError(f"实验 {report.get('experiment_name')} 缺少指标 {key}")
    return float(value)


def compare(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    metric_keys: set[str] | None = None,
) -> dict[str, Any]:
    metrics = {}
    available = set(baseline.get("metrics", {})) | set(candidate.get("metrics", {}))
    keys = sorted(available if metric_keys is None else available & metric_keys)
    for key in keys:
        before = baseline.get("metrics", {}).get(key, {}).get("mean")
        after = candidate.get("metrics", {}).get(key, {}).get("mean")
        absolute = None if before is None or after is None else round(float(after) - float(before), 6)
        relative = None
        if absolute is not None and float(before) != 0:
            relative = round(absolute / abs(float(before)), 6)
        metrics[key] = {
            "baseline": before,
            "candidate": after,
            "absolute_delta": absolute,
            "relative_delta": relative,
        }
    runtime = {}
    runtime_keys = sorted(
        set(baseline.get("runtime_metrics", {})) | set(candidate.get("runtime_metrics", {}))
    )
    for key in runtime_keys:
        before = baseline.get("runtime_metrics", {}).get(key)
        after = candidate.get("runtime_metrics", {}).get(key)
        runtime[key] = {"baseline": before, "candidate": after}
    return {
        "baseline_experiment": baseline.get("experiment_name"),
        "candidate_experiment": candidate.get("experiment_name"),
        "baseline_run_count": baseline.get("run_count"),
        "candidate_run_count": candidate.get("run_count"),
        "metrics": metrics,
        "runtime_metrics": runtime,
        "project_stats": {
            "baseline": baseline.get("project_stats", {}),
            "candidate": candidate.get("project_stats", {}),
        },
    }


def write_json(name: str, value: dict[str, Any]) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def write_comparison(name: str, value: dict[str, Any]) -> None:
    write_json(name, value)
    csv_path = (REPORTS / name).with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["metric", "baseline", "candidate", "absolute_delta", "relative_delta"],
        )
        writer.writeheader()
        for key, values in sorted(value.get("metrics", {}).items()):
            writer.writerow({"metric": key, **values})


def run_eval(
    *,
    dataset: str,
    mode: str,
    rag_mode: str,
    review_mode: str,
    repetitions: int,
    experiment: str,
    output: str,
    judge: bool = False,
    fail_on_threshold: bool = False,
    max_examples: int = 0,
    categories: str = "",
    seed: int = 0,
    profile: str = "compare",
) -> dict[str, Any]:
    command = [
        PYTHON,
        "scripts/run_langsmith_eval.py",
        "--dataset", dataset,
        "--mode", mode,
        "--rag-mode", rag_mode,
        "--review-mode", review_mode,
        "--repetitions", str(repetitions),
        "--concurrency", "2",
        "--experiment", experiment,
        "--output", str(REPORTS / output),
        "--require-export",
        "--profile", profile,
        "--upload-results",
        "--confirm-upload",
    ]
    if max_examples:
        command.extend(["--max-examples", str(max_examples)])
    if categories:
        command.extend(["--sample-categories", categories])
    if seed:
        command.extend(["--sample-seed", str(seed)])
    if judge:
        command.append("--judge")
    if fail_on_threshold:
        command.append("--fail-on-threshold")
    run_checked(command)
    return load_report(output)


def port_in_use(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.3)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def wait_for_service(port: int, timeout: float = 120.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        try:
            with urlopen(f"http://127.0.0.1:{port}/api/index/status", timeout=2) as response:
                value = json.loads(response.read().decode("utf-8"))
            if value.get("status") == "ready" and value.get("dense_enabled") is True:
                return value
            last_error = str(value)
        except Exception as exc:  # noqa: BLE001 - bounded readiness polling
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(1)
    raise RuntimeError(f"LawStation 未在限定时间内进入 Dense ready：{last_error}")


def start_service(rag_mode: str) -> tuple[subprocess.Popen, Any]:
    settings = get_settings()
    if port_in_use(settings.app_port):
        raise RuntimeError(f"端口 {settings.app_port} 已被占用；为避免结束外部服务，实验停止")
    log_path = Path(settings.log_dir) / f"eval-lawstation-{rag_mode}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8")
    env = os.environ.copy()
    env["RAG_RETRIEVAL_MODE"] = rag_mode
    process = subprocess.Popen(
        [PYTHON, "run.py", "--no-build"],
        cwd=ROOT,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        wait_for_service(settings.app_port)
    except Exception:
        stop_service(process, handle)
        raise
    return process, handle


def stop_service(process: subprocess.Popen | None, handle: Any | None) -> None:
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
    if handle is not None:
        handle.close()


def preflight() -> None:
    settings = get_settings()
    missing = []
    if not settings.langsmith_api_key:
        missing.append("LANGSMITH_API_KEY")
    if not settings.deepseek_api_key:
        missing.append("DEEPSEEK_API_KEY")
    if not Path(settings.law_data_path).is_file():
        missing.append("LAW_DATA_PATH")
    manifest_path = Path(settings.index_dir) / "law" / "manifest.json"
    if not manifest_path.is_file():
        missing.append("正式 FAISS manifest")
    if missing:
        raise RuntimeError("预检缺少：" + ", ".join(missing))
    manifest = json.loads(manifest_path.read_text("utf-8"))
    if manifest.get("status") != "ready" or not manifest.get("embedding_model_digest"):
        raise RuntimeError("正式 FAISS manifest 未处于 ready 或缺少模型 digest")
    client = Client(
        api_url=settings.langsmith_endpoint,
        api_key=settings.langsmith_api_key,
        workspace_id=settings.langsmith_workspace_id or None,
    )
    list(client.list_datasets(limit=1))


def stage_two_gate(baseline: dict[str, Any], candidate: dict[str, Any]) -> None:
    for key in (
        "schema_validity", "citation_grounding", "no_match_safety",
        "tenant_isolation", "loop_limit",
    ):
        if metric(candidate, key) != 1.0:
            raise RuntimeError(f"Reviewer Candidate 安全门禁未通过：{key}")
    for key in ("judge_evidence_consistency", "judge_risk_calibration"):
        if metric(candidate, key) < metric(baseline, key):
            raise RuntimeError(f"Reviewer Candidate 的 {key} 低于 Baseline")
    before_calls = float(baseline["runtime_metrics"]["mean_model_calls"])
    after_calls = float(candidate["runtime_metrics"]["mean_model_calls"])
    before_latency = baseline.get("project_stats", {}).get("latency_p50")
    after_latency = candidate.get("project_stats", {}).get("latency_p50")
    faster = (
        before_latency is not None and after_latency is not None
        and float(after_latency) < float(before_latency)
    )
    if after_calls >= before_calls and not faster:
        raise RuntimeError("Reviewer Candidate 未降低模型调用次数或 p50 延迟")


def markdown_report(comparisons: dict[str, dict[str, Any]], status: str) -> str:
    lines = [
        "# LawStation LangSmith Eval Report", "",
        f"- 状态：`{status}`",
        f"- 生成时间：`{datetime.now(UTC).isoformat()}`", "",
        "> 检索集由 law.json 源数据派生并通过真实 document_id/chunk_id 校验，尚未经过律师人工标注；",
        "> 因此可用于工程检索回归，不能宣称为专家标注的法律准确率。", "",
    ]
    for title, comparison in comparisons.items():
        lines.extend([f"## {title}", "", "| 指标 | Baseline | Candidate | 变化 |", "|---|---:|---:|---:|"])
        for key, values in comparison.get("metrics", {}).items():
            lines.append(
                f"| {key} | {values.get('baseline')} | {values.get('candidate')} | "
                f"{values.get('absolute_delta')} |"
            )
        lines.append("")
    lines.extend([
        "## 简历表述规则", "",
        "仅使用本报告实际生成的样本数、均值和差值；不得把门禁阈值写成实测成绩。",
    ])
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description="运行低资源 LawStation LangSmith 分阶段实验")
    parser.add_argument("--profile", choices=["compare", "release"], default="compare")
    parser.add_argument("--confirm-upload", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--sample-seed", type=int, default=0)
    return parser.parse_args()


def resource_plan(profile: str) -> dict[str, Any]:
    sizes = (
        {"retrieval": 10, "reviewer": 3, "e2e": 2}
        if profile == "compare" else
        {"retrieval": 15, "reviewer": 6, "e2e": 4}
    )
    settings = get_settings()
    traces = sizes["retrieval"] * 2 + sizes["reviewer"] * 2 + sizes["e2e"] * 2
    judge = sizes["reviewer"] * 2 + sizes["e2e"] * 2
    return {
        "profile": profile,
        "sample_sizes": sizes,
        "planned_traces": traces,
        "planned_judge_calls": judge,
        "agent_model_call_upper_bound": (
            (sizes["reviewer"] * 2 + sizes["e2e"] * 2) * settings.agent_max_model_calls
        ),
        "note": "Agent 上限按每条最坏情况计算；各阶段成功后按实际调用结算。",
    }


def main() -> None:
    args = parse_args()
    plan = resource_plan(args.profile)
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.plan_only:
        return
    if not args.confirm_upload:
        raise SystemExit("云端分阶段实验必须显式提供 --confirm-upload；可先使用 --plan-only")
    REPORTS.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC)
    comparisons: dict[str, dict[str, Any]] = {}
    status = "running"
    ollama = None
    service = None
    service_log = None
    try:
        preflight()
        run_checked([PYTHON, "-m", "pytest", "tests/test_langsmith_observability.py", "tests/test_agent_runtime.py", "tests/test_index_manager.py", "-q"])
        run_checked([PYTHON, "scripts/create_eval_datasets.py"])
        run_checked([PYTHON, "scripts/create_live_retrieval_dataset.py"])
        run_checked([PYTHON, "scripts/seed_langsmith_datasets.py"])

        ollama = OllamaProcessManager(get_settings())
        ollama.ensure_ready()

        sizes = plan["sample_sizes"]
        rag_runs = sizes["retrieval"]
        rag_baseline = reusable_report(
            "rag-bm25-baseline.json", dataset="lawstation-live-retrieval-v1", runs=rag_runs,
            categories=["matched", "no_match"], seed=args.sample_seed,
        ) or run_eval(
            dataset="lawstation-live-retrieval-v1", mode="retrieval", rag_mode="bm25",
            review_mode="auto", repetitions=1, experiment="rag-bm25-baseline",
            output="rag-bm25-baseline.json", fail_on_threshold=True,
            max_examples=sizes["retrieval"], categories="matched,no_match",
            seed=args.sample_seed, profile=args.profile,
        )
        rag_candidate = reusable_report(
            "rag-hybrid-candidate.json", dataset="lawstation-live-retrieval-v1", runs=rag_runs,
            categories=["matched", "no_match"], seed=args.sample_seed,
        ) or run_eval(
            dataset="lawstation-live-retrieval-v1", mode="retrieval", rag_mode="hybrid",
            review_mode="auto", repetitions=1, experiment="rag-hybrid-candidate",
            output="rag-hybrid-candidate.json", fail_on_threshold=True,
            max_examples=sizes["retrieval"], categories="matched,no_match",
            seed=args.sample_seed, profile=args.profile,
        )
        if metric(rag_candidate, "retrieval_recall_at_k") < metric(rag_baseline, "retrieval_recall_at_k"):
            raise RuntimeError("Hybrid Recall@5 低于 BM25，停止后续实验")
        comparisons["RAG：BM25 vs Hybrid"] = compare(
            rag_baseline,
            rag_candidate,
            {
                "retrieval_status_correctness", "retrieval_recall_at_k",
                "retrieval_mrr", "exact_article_hit",
            },
        )
        write_comparison("rag-bm25-vs-hybrid.json", comparisons["RAG：BM25 vs Hybrid"])

        reviewer_baseline = reusable_report(
            "reviewer-llm-baseline.json", dataset="lawstation-agent-v3",
            runs=sizes["reviewer"],
            categories=["matched", "no_match", "memory"], seed=args.sample_seed,
        ) or run_eval(
            dataset="lawstation-agent-v3", mode="component", rag_mode="hybrid",
            review_mode="always-llm", repetitions=1, experiment="reviewer-llm-baseline",
            output="reviewer-llm-baseline.json", judge=True,
            max_examples=sizes["reviewer"], categories="matched,no_match,memory",
            seed=args.sample_seed, profile=args.profile,
        )
        reviewer_candidate = run_eval(
            dataset="lawstation-agent-v3", mode="component", rag_mode="hybrid",
            review_mode="auto", repetitions=1, experiment="reviewer-fastpath-candidate",
            output="reviewer-fastpath-candidate.json", judge=True,
            max_examples=sizes["reviewer"], categories="matched,no_match,memory",
            seed=args.sample_seed, profile=args.profile,
        )
        stage_two_gate(reviewer_baseline, reviewer_candidate)
        comparisons["Agent：LLM Reviewer vs Fast Path"] = compare(
            reviewer_baseline, reviewer_candidate
        )
        write_comparison(
            "reviewer-llm-vs-fastpath.json",
            comparisons["Agent：LLM Reviewer vs Fast Path"],
        )

        service, service_log = start_service("bm25")
        e2e_baseline = reusable_report(
            "e2e-baseline.json", dataset="lawstation-e2e-v2", runs=sizes["e2e"],
            categories=["matched", "no_match"], seed=args.sample_seed,
        ) or run_eval(
            dataset="lawstation-e2e-v2", mode="live", rag_mode="bm25",
            review_mode="always-llm", repetitions=1, experiment="lawstation-e2e-baseline",
            output="e2e-baseline.json", judge=True,
            max_examples=sizes["e2e"], categories="matched,no_match",
            seed=args.sample_seed, profile=args.profile,
        )
        stop_service(service, service_log)
        service = service_log = None
        service, service_log = start_service("hybrid")
        e2e_candidate = run_eval(
            dataset="lawstation-e2e-v2", mode="live", rag_mode="hybrid",
            review_mode="auto", repetitions=1, experiment="lawstation-e2e-candidate",
            output="e2e-candidate.json", judge=True, fail_on_threshold=True,
            max_examples=sizes["e2e"], categories="matched,no_match",
            seed=args.sample_seed, profile=args.profile,
        )
        comparisons["E2E：Baseline vs Candidate"] = compare(e2e_baseline, e2e_candidate)
        write_comparison("e2e-baseline-vs-candidate.json", comparisons["E2E：Baseline vs Candidate"])
        status = "completed"
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except Exception as exc:
        status = f"failed: {type(exc).__name__}: {exc}"
        raise
    finally:
        stop_service(service, service_log)
        if ollama is not None:
            ollama.close()
        manifest = {
            "status": status,
            "started_at": started.isoformat(),
            "completed_at": datetime.now(UTC).isoformat(),
            "comparisons": {
                key: {
                    "baseline": value.get("baseline_experiment"),
                    "candidate": value.get("candidate_experiment"),
                }
                for key, value in comparisons.items()
            },
        }
        write_json("experiment-manifest.json", manifest)
        (REPORTS / "EVAL_REPORT.md").write_text(
            markdown_report(comparisons, status), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
