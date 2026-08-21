import argparse
import asyncio
import json
import subprocess
from pathlib import Path

from langsmith import Client, aevaluate, evaluate

from backend.app.core.config import get_settings
from backend.app.evaluation import DETERMINISTIC_EVALUATORS
from backend.app.evaluation.judge import LegalQualityJudge
from backend.app.evaluation.targets import component_target, live_target

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = {
    "schema_validity": 1.0,
    "citation_grounding": 1.0,
    "no_match_safety": 1.0,
    "tenant_isolation": 1.0,
    "loop_limit": 1.0,
    "route_correctness": 0.95,
    "retrieval_recall_at_k": 0.85,
    "judge_evidence_consistency": 0.86,
    "judge_factual_fidelity": 0.86,
    "judge_risk_calibration": 0.8,
    "judge_helpfulness": 0.8,
}


def metadata(mode: str) -> dict:
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
        "graph_version": "three-agent-v1",
        "prompt_version": "legal-consultation-v1",
        "model": settings.deepseek_model,
        "embedding_model": settings.embedding_model,
        "law_data_version": manifest.get("fingerprint", "unknown"),
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


async def run(args) -> None:
    ls_client = client()
    if args.compare:
        result = evaluate(
            (args.compare[0], args.compare[1]),
            data=args.dataset,
            client=ls_client,
        )
        print(result)
        return
    evaluators = list(DETERMINISTIC_EVALUATORS)
    if args.judge:
        evaluators.append(LegalQualityJudge())
    target = component_target if args.mode == "component" else live_target
    results = await aevaluate(
        target,
        data=args.dataset,
        evaluators=evaluators,
        metadata=metadata(args.mode),
        experiment_prefix=args.experiment or f"{args.dataset}-{args.mode}",
        max_concurrency=args.concurrency,
        client=ls_client,
    )
    rows = [row async for row in results]
    scores: dict[str, list[float]] = {}
    for row in rows:
        evaluation_results = row.get("evaluation_results", {}).get("results", [])
        for item in evaluation_results:
            score = getattr(item, "score", None)
            key = getattr(item, "key", None)
            if key and score is not None:
                scores.setdefault(key, []).append(float(score))
    averages = {key: sum(values) / len(values) for key, values in scores.items() if values}
    print(json.dumps({"examples": len(rows), "averages": averages}, ensure_ascii=False, indent=2))
    if args.fail_on_threshold:
        failed = {
            key: {"actual": averages.get(key), "required": threshold}
            for key, threshold in THRESHOLDS.items()
            if key in averages and averages[key] < threshold
        }
        if failed:
            raise SystemExit("评测门禁未通过：" + json.dumps(failed, ensure_ascii=False))


def parse_args():
    parser = argparse.ArgumentParser(description="运行 LawStation LangSmith 评测")
    parser.add_argument("--dataset", default="lawstation-e2e-v1")
    parser.add_argument("--mode", choices=["component", "live"], default="component")
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--fail-on-threshold", action="store_true")
    parser.add_argument("--experiment", default="")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE", "CURRENT"))
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
