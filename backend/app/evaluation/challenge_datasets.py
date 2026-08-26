"""Prepare, validate, and freeze source-grounded resume challenge datasets."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from mcp_servers.law_rag.engine import load_chunks

DENSE_DATASET = "lawstation-dense-challenge-v1"
RERANK_CANDIDATE_DATASET = "lawstation-reranker-challenge-candidates-v1"
RERANK_DATASET = "lawstation-reranker-challenge-v1"
DEFAULT_DENSE_SIZE = 300
DEFAULT_RERANK_CANDIDATE_SIZE = 600
DEFAULT_RERANK_SIZE = 200
PROMPT_VERSION = "resume-rag-challenge-v1"
GENERATOR_MODEL = "gpt-5.6-sol"
MAX_SHARED_SOURCE_CHARS = 6

DENSE_STYLES = (
    "semantic_paraphrase",
    "everyday_scenario",
    "consequence_only",
    "colloquial_noise",
)
RERANK_STYLES = (
    "adjacent_article",
    "general_vs_exception",
    "definition_vs_liability",
    "subject_or_condition",
)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _jsonl(items: Iterable[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(item, ensure_ascii=False) for item in items) + "\n"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized(value: str) -> str:
    return re.sub(r"[\s\W_]+", "", value, flags=re.UNICODE).lower()


def leaks_source_text(question: str, source: str, length: int = 7) -> bool:
    normalized_question = _normalized(question)
    normalized_source = _normalized(source)
    if len(normalized_question) < length or len(normalized_source) < length:
        return False
    return any(
        normalized_source[index : index + length] in normalized_question
        for index in range(len(normalized_source) - length + 1)
    )


def _eligible_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        chunk
        for chunk in chunks
        if chunk.get("law_name")
        and chunk.get("article_number")
        and 50 <= len(str(chunk.get("content", ""))) <= 1_000
    ]


def _round_robin_sources(
    grouped: dict[str, list[dict[str, Any]]],
    *,
    count: int,
    seed: int,
    excluded: set[str] | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    excluded = excluded or set()
    names = sorted(grouped)
    rng.shuffle(names)
    pools: dict[str, list[dict[str, Any]]] = {}
    for name in names:
        values = [item for item in grouped[name] if item["chunk_id"] not in excluded]
        rng.shuffle(values)
        pools[name] = values
    selected: list[dict[str, Any]] = []
    round_index = 0
    while len(selected) < count:
        progressed = False
        for name in names:
            values = pools[name]
            if round_index < len(values):
                selected.append(values[round_index])
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
        round_index += 1
    if len(selected) != count:
        raise RuntimeError(f"符合条件的法规 chunk 不足：需要 {count}，实际 {len(selected)}")
    return selected


def prepare_source_pack(
    chunks: list[dict[str, Any]],
    *,
    dense_count: int = DEFAULT_DENSE_SIZE,
    rerank_candidate_count: int = DEFAULT_RERANK_CANDIDATE_SIZE,
    seed: int = 42,
) -> list[dict[str, Any]]:
    eligible = _eligible_chunks(chunks)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in eligible:
        grouped[str(chunk["law_name"])].append(chunk)

    dense_sources = _round_robin_sources(grouped, count=dense_count, seed=seed)
    dense_ids = {item["chunk_id"] for item in dense_sources}
    rerank_groups = {name: values for name, values in grouped.items() if len(values) >= 3}
    rerank_sources = _round_robin_sources(
        rerank_groups,
        count=rerank_candidate_count,
        seed=seed + 1,
        excluded=dense_ids,
    )

    tasks: list[dict[str, Any]] = []
    for index, source in enumerate(dense_sources):
        tasks.append(_task(source, f"dense-{index + 1:04d}", DENSE_STYLES[index % 4]))
    for index, source in enumerate(rerank_sources):
        same_law = rerank_groups[str(source["law_name"])]
        position = next(i for i, item in enumerate(same_law) if item["chunk_id"] == source["chunk_id"])
        neighbors = [
            same_law[i]
            for i in (position - 2, position - 1, position + 1, position + 2)
            if 0 <= i < len(same_law)
        ][:3]
        tasks.append(
            _task(
                source,
                f"rerank-{index + 1:04d}",
                RERANK_STYLES[index % 4],
                distractors=neighbors,
            )
        )
    return tasks


def assert_unchanged_prefix(
    existing: list[dict[str, Any]],
    expanded: list[dict[str, Any]],
    *,
    label: str,
    ignored_input_keys: set[str] | None = None,
    ignored_metadata_keys: set[str] | None = None,
) -> None:
    """Protect frozen generation work when a deterministic pool is expanded."""

    if len(existing) > len(expanded):
        raise RuntimeError(
            f"{label} 不能缩减：现有 {len(existing)} 条，新结果仅 {len(expanded)} 条"
        )
    ignored_input_keys = ignored_input_keys or set()
    ignored_metadata_keys = ignored_metadata_keys or set()

    def comparable(item: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(item)
        if ignored_input_keys and isinstance(item.get("inputs"), dict):
            normalized["inputs"] = {
                key: value
                for key, value in item["inputs"].items()
                if key not in ignored_input_keys
            }
        if ignored_metadata_keys and isinstance(item.get("metadata"), dict):
            normalized["metadata"] = {
                key: value
                for key, value in item["metadata"].items()
                if key not in ignored_metadata_keys
            }
        return normalized

    for index, (before, after) in enumerate(zip(existing, expanded, strict=False), 1):
        if comparable(before) != comparable(after):
            raise RuntimeError(f"{label} 第 {index} 条发生变化，拒绝覆盖既有冻结内容")


def _task(
    source: dict[str, Any],
    task_id: str,
    style: str,
    *,
    distractors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "challenge_set": "dense" if task_id.startswith("dense-") else "reranker",
        "style": style,
        "target": {
            key: source[key]
            for key in ("document_id", "chunk_id", "law_name", "article_number", "content")
        },
        "distractors": [
            {
                key: item[key]
                for key in ("document_id", "chunk_id", "law_name", "article_number", "content")
            }
            for item in (distractors or [])
        ],
    }


def generation_prompt() -> str:
    return """你正在生成法律法规检索挑战题。每个输入任务只输出 task_id 和 question。

硬性要求：
1. 问题必须只由 target 法条直接支持，不得引入 target 中不存在的关键事实。
2. 不得出现法律名称、法条编号，且不得连续复制 target 正文超过 6 个字符。
3. dense 任务按 style 写成口语案情、语义改写、仅描述后果或带少量噪声的问题。
4. reranker 任务要突出 target 与 distractors 在主体、条件、例外或法律后果上的区别。
5. 问题长度为 15～180 个字符；不得输出答案、法条或解释。
6. 输出必须是 JSON 对象：{"responses":[{"task_id":"...", "question":"..."}]}。
"""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def validate_and_build(
    tasks: list[dict[str, Any]],
    responses: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_task = {str(item["task_id"]): item for item in tasks}
    seen_tasks: set[str] = set()
    responded_tasks: set[str] = set()
    seen_questions: set[str] = set()
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for response in responses:
        task_id = str(response.get("task_id", ""))
        question = str(response.get("question", "")).strip()
        reasons: list[str] = []
        task = by_task.get(task_id)
        if task is None:
            reasons.append("unknown_task_id")
        else:
            responded_tasks.add(task_id)
        if task_id in seen_tasks:
            reasons.append("duplicate_task_id")
        normalized_question = _normalized(question)
        if normalized_question in seen_questions:
            reasons.append("duplicate_question")
        if not 15 <= len(question) <= 180:
            reasons.append("invalid_question_length")
        if re.search(r"《[^》]+》", question):
            reasons.append("explicit_law_title_leak")
        if re.search(r"第[零一二三四五六七八九十百千万两0-9]+条", question):
            reasons.append("explicit_article_number_leak")
        if task is not None:
            target = task["target"]
            if str(target["law_name"]) in question:
                reasons.append("law_name_leak")
            if str(target["article_number"]) in question:
                reasons.append("article_number_leak")
            if leaks_source_text(question, str(target["content"]), MAX_SHARED_SOURCE_CHARS + 1):
                reasons.append("source_text_leak")
        if reasons:
            rejected.append({"task_id": task_id, "reasons": sorted(set(reasons))})
            continue
        seen_tasks.add(task_id)
        seen_questions.add(normalized_question)
        accepted.append(_dataset_example(task, question))
    missing = sorted(set(by_task) - responded_tasks)
    rejected.extend({"task_id": task_id, "reasons": ["missing_response"]} for task_id in missing)
    return accepted, rejected


def _dataset_example(task: dict[str, Any], question: str) -> dict[str, Any]:
    target = task["target"]
    prompt_sha = _sha256_text(generation_prompt())
    return {
        "inputs": {"question": question, "top_k": 5, "filters": None},
        "outputs": {
            "expected_route": None,
            "expected_retrieval_status": "matched",
            "expected_document_ids": [target["document_id"]],
            "expected_chunk_ids": [target["chunk_id"]],
            "expected_law_name": target["law_name"],
            "expected_article_number": target["article_number"],
        },
        "metadata": {
            "category": task["style"],
            "difficulty": "hard",
            "challenge_set": task["challenge_set"],
            "synthetic": True,
            "source_derived": True,
            "human_verified": False,
            "generator": "codex",
            "generator_model": GENERATOR_MODEL,
            "reasoning_effort": "xhigh",
            "generation_prompt_version": PROMPT_VERSION,
            "generation_prompt_sha256": prompt_sha,
            "source_content_sha256": _sha256_text(str(target["content"])),
            "task_id": task["task_id"],
            "hard_distractor_chunk_ids": [item["chunk_id"] for item in task["distractors"]],
            "qualification_status": (
                "not_required" if task["challenge_set"] == "dense" else "pending"
            ),
        },
    }


def write_frozen_dataset(
    path: Path,
    examples: list[dict[str, Any]],
    *,
    source_file: Path,
    selection_rule: str,
    extra_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _atomic_text(path, _jsonl(examples))
    manifest = {
        "dataset": path.stem,
        "dataset_sha256": file_sha256(path),
        "sample_count": len(examples),
        "source_file": source_file.name,
        "source_sha256": file_sha256(source_file),
        "selection_rule": selection_rule,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": _sha256_text(generation_prompt()),
        "human_verified": False,
        "generator": "codex",
        "generator_model": GENERATOR_MODEL,
        "reasoning_effort": "xhigh",
        **(extra_manifest or {}),
    }
    _atomic_text(path.with_suffix(".manifest.json"), json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def validate_examples_against_chunks(
    examples: list[dict[str, Any]], chunks: list[dict[str, Any]]
) -> list[str]:
    by_chunk = {str(item["chunk_id"]): item for item in chunks}
    errors: list[str] = []
    seen_questions: set[str] = set()
    for index, example in enumerate(examples, 1):
        outputs = example.get("outputs", {})
        metadata = example.get("metadata", {})
        chunk_ids = outputs.get("expected_chunk_ids", [])
        if len(chunk_ids) != 1 or str(chunk_ids[0]) not in by_chunk:
            errors.append(f"row {index}: invalid_gold_chunk")
            continue
        chunk = by_chunk[str(chunk_ids[0])]
        if outputs.get("expected_document_ids") != [chunk["document_id"]]:
            errors.append(f"row {index}: document_chunk_mismatch")
        if metadata.get("source_content_sha256") != _sha256_text(str(chunk["content"])):
            errors.append(f"row {index}: source_hash_mismatch")
        question = str(example.get("inputs", {}).get("question", ""))
        normalized = _normalized(question)
        if normalized in seen_questions:
            errors.append(f"row {index}: duplicate_question")
        seen_questions.add(normalized)
        if str(chunk["law_name"]) in question or str(chunk["article_number"]) in question:
            errors.append(f"row {index}: answer_leak")
        if re.search(r"《[^》]+》|第[零一二三四五六七八九十百千万两0-9]+条", question):
            errors.append(f"row {index}: explicit_legal_reference_leak")
        if leaks_source_text(question, str(chunk["content"]), MAX_SHARED_SOURCE_CHARS + 1):
            errors.append(f"row {index}: source_text_leak")
    return errors


def load_current_chunks(law_path: Path, max_chars: int, overlap_chars: int) -> list[dict[str, Any]]:
    return load_chunks(law_path, max_chars, overlap_chars)
