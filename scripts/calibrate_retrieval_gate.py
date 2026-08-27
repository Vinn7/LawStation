from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from mcp_servers.law_rag.engine import LawSearchEngine

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evals/datasets/lawstation-retrieval-calibration-v1.jsonl"
CONFIG = ROOT / "evals/config/retrieval-gate-v1.json"


def metrics(rows: list[dict], threshold: float) -> dict[str, float]:
    tp = sum(row["expected"] and row["confidence"] >= threshold for row in rows)
    tn = sum(not row["expected"] and row["confidence"] < threshold for row in rows)
    fp = sum(not row["expected"] and row["confidence"] >= threshold for row in rows)
    fn = sum(row["expected"] and row["confidence"] < threshold for row in rows)
    matched_count = sum(row["expected"] for row in rows)
    baseline_recall = sum(row["expected"] and row["gold_hit"] for row in rows) / max(1, matched_count)
    gated_recall = sum(
        row["expected"] and row["gold_hit"] and row["confidence"] >= threshold for row in rows
    ) / max(1, matched_count)
    return {
        "accuracy": (tp + tn) / max(1, len(rows)),
        "no_match_precision": tn / max(1, tn + fn),
        "no_match_recall": tn / max(1, tn + fp),
        "retrieval_recall_at_5": gated_recall,
        "retrieval_recall_drop": baseline_recall - gated_recall,
    }


async def collect() -> list[dict]:
    examples = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line]
    engine = LawSearchEngine()
    await engine.initialize_index()
    rows = []
    try:
        for example in examples:
            result = await engine.search(
                example["inputs"]["question"],
                top_k=5,
                filters=example["inputs"].get("filters"),
                retrieval_mode="hybrid",
                envelope=True,
                apply_confidence_gate=False,
            )
            expected_ids = set(example["outputs"].get("expected_chunk_ids") or [])
            documents = result["documents"]
            rows.append({
                "split": example["metadata"]["split"],
                "expected": example["outputs"]["expected_retrieval_status"] == "matched",
                "confidence": float(result["diagnostics"]["confidence"]),
                "gold_hit": not expected_ids or bool(expected_ids & {doc["chunk_id"] for doc in documents}),
            })
    finally:
        await engine.close()
    return rows


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="校准通过后原子更新 gate 配置")
    args = parser.parse_args()
    rows = await collect()
    development = [row for row in rows if row["split"] == "development"]
    validation = [row for row in rows if row["split"] == "validation"]
    candidates = []
    for step in range(101):
        threshold = step / 100
        result = metrics(development, threshold)
        if (
            result["accuracy"] >= 0.95
            and result["no_match_precision"] >= 0.90
            and result["no_match_recall"] >= 0.90
            and result["retrieval_recall_drop"] <= 0.01
        ):
            candidates.append((result["accuracy"], threshold, result))
    if not candidates:
        raise SystemExit("开发集没有阈值满足准确率、no-match 和 Recall@5 门禁；配置未修改")
    _, threshold, development_metrics = max(candidates)
    validation_metrics = metrics(validation, threshold)
    report = {
        "threshold": threshold,
        "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        "development": development_metrics,
        "validation": validation_metrics,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.write:
        if not (
            validation_metrics["accuracy"] >= 0.95
            and validation_metrics["no_match_precision"] >= 0.90
            and validation_metrics["no_match_recall"] >= 0.90
            and validation_metrics["retrieval_recall_drop"] <= 0.01
        ):
            raise SystemExit("冻结验证集未通过门禁；配置未修改")
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config.update({
            "threshold": threshold,
            "calibration_status": "validated",
            "dataset_sha256": report["dataset_sha256"],
            "validation_metrics": validation_metrics,
        })
        temporary = CONFIG.with_suffix(".tmp")
        temporary.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(CONFIG)


if __name__ == "__main__":
    asyncio.run(main())
