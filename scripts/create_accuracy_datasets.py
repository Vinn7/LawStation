from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "evals" / "datasets"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> str:
    content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
    return hashlib.sha256(content.encode()).hexdigest()


def calibration_rows() -> list[dict]:
    live = read_jsonl(DATASETS / "lawstation-live-retrieval-v1.jsonl")
    matched = [row for row in live if row["outputs"]["expected_retrieval_status"] == "matched"]
    if len(matched) < 100:
        extra = read_jsonl(DATASETS / "lawstation-reranker-challenge-v1.jsonl")
        matched.extend(extra[: 100 - len(matched)])
    no_match = [row for row in live if row["outputs"]["expected_retrieval_status"] == "no_match"]
    variants = ["请判断：{}", "咨询一下，{}", "如果发生{}，应适用什么规则", "关于{}，法律如何处理", "{}具体怎么办"]
    expanded_no_match = []
    for source in no_match:
        for variant in variants:
            row = json.loads(json.dumps(source, ensure_ascii=False))
            row["inputs"]["question"] = variant.format(source["inputs"]["question"])
            expanded_no_match.append(row)
    rows = []
    for index, row in enumerate(matched[:100] + expanded_no_match[:100]):
        copied = json.loads(json.dumps(row, ensure_ascii=False))
        copied["metadata"] = {
            **copied.get("metadata", {}),
            "dataset": "retrieval-confidence-calibration-v1",
            "split": "development" if (index % 5) < 3 else "validation",
            "frozen": True,
            "human_verified": False,
        }
        rows.append(copied)
    return rows


def fact_conflict_rows() -> list[dict]:
    facts = [
        ("monthly_salary", "月工资八千元", "月工资一万元"),
        ("termination_date", "解除日期是3月1日", "解除日期是4月15日"),
        ("party_name", "公司名称是甲公司", "公司名称是乙公司"),
        ("rent_amount", "月租金三千元", "月租金四千元"),
        ("loan_amount", "借款金额五万元", "借款金额八万元"),
        ("contract_term", "合同期限一年", "合同期限两年"),
    ]
    forms = [
        "之前说错了，{new}。",
        "请更正一下，最新事实是{new}。",
        "不是原先那个数，实际为{new}。",
        "以我这次提供的信息为准：{new}。",
        "我核对材料后确认，{new}。",
    ]
    rows = []
    for key, old, new in facts:
        for form in forms:
            rows.append({
                "inputs": {
                    "question": form.format(new=new),
                    "history": [],
                    "memory_context": f"历史有效记忆：{old}",
                    "fixture_documents": [],
                    "fixture_tool_error": False,
                },
                "outputs": {
                    "expected_current_fact": new,
                    "forbidden_old_fact": old,
                    "expected_canonical_key": key,
                },
                "metadata": {
                    "category": "fact_conflict",
                    "synthetic": True,
                    "frozen": True,
                    "human_verified": False,
                },
            })
    return rows


def main() -> None:
    DATASETS.mkdir(parents=True, exist_ok=True)
    calibration = calibration_rows()
    conflicts = fact_conflict_rows()
    calibration_sha = write_jsonl(DATASETS / "lawstation-retrieval-calibration-v1.jsonl", calibration)
    conflicts_sha = write_jsonl(DATASETS / "lawstation-fact-conflicts-v1.jsonl", conflicts)
    manifest = {
        "schema_version": "accuracy-datasets-v1",
        "seed": 42,
        "retrieval_calibration": {
            "count": len(calibration),
            "matched": 100,
            "no_match": 100,
            "development": 120,
            "validation": 80,
            "sha256": calibration_sha,
        },
        "fact_conflicts": {"count": len(conflicts), "sha256": conflicts_sha},
    }
    path = DATASETS / "lawstation-accuracy-datasets-v1.manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
