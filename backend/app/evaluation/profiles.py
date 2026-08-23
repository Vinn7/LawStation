"""Deterministic local dataset selection for low-resource evaluations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from langsmith.schemas import Example

ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = ROOT / "evals" / "datasets"


def dataset_path(dataset: str) -> Path:
    path = DATASET_DIR / f"{dataset}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"本地评测数据集不存在：{path}")
    return path


def dataset_sha256(dataset: str) -> str:
    return hashlib.sha256(dataset_path(dataset).read_bytes()).hexdigest()


def case_hash(case: dict[str, Any]) -> str:
    payload = {"inputs": case.get("inputs", {}), "outputs": case.get("outputs", {})}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_cases(dataset: str) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in dataset_path(dataset).read_text("utf-8").splitlines()
        if line.strip()
    ]


def effective_seed(seed: int) -> int:
    return seed if seed > 0 else int(datetime.now(UTC).strftime("%Y%m"))


def select_cases(
    cases: Iterable[dict[str, Any]],
    *,
    limit: int,
    categories: list[str],
    seed: int,
) -> list[dict[str, Any]]:
    values = list(cases)
    allowed = set(categories)
    if allowed:
        values = [item for item in values if item.get("metadata", {}).get("category") in allowed]
    values.sort(key=lambda item: hashlib.sha256(f"{seed}:{case_hash(item)}".encode()).hexdigest())
    if limit <= 0 or limit >= len(values):
        return values
    if not categories:
        return values[:limit]
    groups = {
        category: [
            item for item in values if item.get("metadata", {}).get("category") == category
        ]
        for category in categories
    }
    selected: list[dict[str, Any]] = []
    while len(selected) < limit:
        progressed = False
        for category in categories:
            if groups[category] and len(selected) < limit:
                selected.append(groups[category].pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def to_examples(dataset: str, cases: Iterable[dict[str, Any]]) -> list[Example]:
    examples = []
    for case in cases:
        digest = case_hash(case)
        examples.append(
            Example(
                id=uuid5(NAMESPACE_URL, f"lawstation:{dataset}:{digest}"),
                inputs=case.get("inputs", {}),
                outputs=case.get("outputs", {}),
                metadata={**case.get("metadata", {}), "content_sha256": digest},
            )
        )
    return examples


def batch_manifest(dataset: str, cases: Iterable[dict[str, Any]], seed: int) -> dict[str, Any]:
    values = list(cases)
    return {
        "dataset": dataset,
        "dataset_sha256": dataset_sha256(dataset),
        "seed": seed,
        "sample_count": len(values),
        "examples": [
            {
                "content_sha256": case_hash(item),
                "category": item.get("metadata", {}).get("category", "unknown"),
            }
            for item in values
        ],
    }
