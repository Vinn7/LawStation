"""Build Codex-authored Dense and Reranker resume challenge datasets."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from backend.app.core.config import get_settings
from backend.app.evaluation.challenge_datasets import (
    DEFAULT_DENSE_SIZE,
    DEFAULT_RERANK_CANDIDATE_SIZE,
    DEFAULT_RERANK_SIZE,
    DENSE_DATASET,
    RERANK_CANDIDATE_DATASET,
    RERANK_DATASET,
    assert_unchanged_prefix,
    file_sha256,
    generation_prompt,
    load_current_chunks,
    load_jsonl,
    prepare_source_pack,
    validate_and_build,
    validate_examples_against_chunks,
    write_frozen_dataset,
)
from mcp_servers.law_rag.engine import LawSearchEngine

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "evals" / "datasets"
WORK = ROOT / "evals" / "generation" / "resume-rag-challenges-v1"
QUALIFICATION_BATCH_SIZE = 25


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _jsonl(items: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(item, ensure_ascii=False) for item in items) + "\n"


def _chunks():
    settings = get_settings()
    return load_current_chunks(
        Path(settings.law_data_path),
        settings.index_chunk_max_chars,
        settings.index_chunk_overlap_chars,
    )


def prepare(args) -> None:
    tasks = prepare_source_pack(
        _chunks(),
        dense_count=args.dense_size,
        rerank_candidate_count=args.rerank_candidate_size,
        seed=args.seed,
    )
    WORK.mkdir(parents=True, exist_ok=True)
    source_pack = WORK / "source-pack.jsonl"
    if source_pack.is_file():
        assert_unchanged_prefix(
            load_jsonl(source_pack), tasks, label="既有生成任务"
        )
    _atomic_text(source_pack, _jsonl(tasks))
    prompt = generation_prompt()
    _atomic_text(WORK / "GENERATION_PROMPT.md", prompt)
    _atomic_text(
        WORK / "response-schema.json",
        json.dumps({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "required": ["responses"],
            "properties": {
                "responses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["task_id", "question"],
                        "properties": {
                            "task_id": {"type": "string"},
                            "question": {"type": "string"},
                        },
                    },
                }
            },
        }, ensure_ascii=False, indent=2),
    )
    batches = WORK / "batches"
    batches.mkdir(parents=True, exist_ok=True)
    expected_batch_names: set[str] = set()
    for index in range(0, len(tasks), args.batch_size):
        batch = tasks[index : index + args.batch_size]
        number = index // args.batch_size + 1
        batch_path = batches / f"batch-{number:02d}.json"
        expected_batch_names.add(batch_path.name)
        if batch_path.is_file() and json.loads(batch_path.read_text("utf-8")) != batch:
            raise SystemExit(f"既有批次发生变化，拒绝覆盖：{batch_path}")
        if not batch_path.is_file():
            _atomic_text(batch_path, json.dumps(batch, ensure_ascii=False, indent=2))
    stale = sorted(
        path.name for path in batches.glob("batch-*.json")
        if path.name not in expected_batch_names
    )
    if stale:
        raise SystemExit(f"发现不属于本次扩容的旧批次，拒绝继续：{', '.join(stale)}")
    print(f"生成任务：{source_pack}（{len(tasks)} 条，{len(list(batches.glob('*.json')))} 批）")
    print("请使用 Codex xhigh 按 GENERATION_PROMPT.md 生成响应，并合并为 responses.jsonl。")


def generate_codex(args) -> None:
    executable = shutil.which(args.codex_command)
    if not executable:
        raise SystemExit(f"找不到 Codex CLI：{args.codex_command}")
    batches = sorted((WORK / "batches").glob("batch-*.json"))
    if not batches:
        raise SystemExit("没有生成批次，请先执行 prepare")
    output_dir = WORK / "codex-responses"
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = generation_prompt()
    schema = WORK / "response-schema.json"

    def invoke_tasks(
        task_file: Path,
        output: Path,
        *,
        extra_instruction: str = "",
    ) -> Path:
        expected = json.loads(task_file.read_text("utf-8"))
        expected_ids = {item.get("task_id") for item in expected}
        needs_generation = not output.is_file() or args.force
        if not needs_generation:
            try:
                existing_payload = json.loads(output.read_text("utf-8"))
                existing_ids = {
                    item.get("task_id")
                    for item in existing_payload.get("responses", [])
                }
            except (OSError, json.JSONDecodeError, AttributeError):
                existing_ids = set()
            if existing_ids != expected_ids:
                print(f"旧响应与当前任务不匹配，重新生成：{output.name}", flush=True)
                needs_generation = True
        if needs_generation:
            command = [
                executable,
                "exec",
                "--ephemeral",
                "--sandbox",
                "read-only",
                "--output-schema",
                str(schema),
                "--output-last-message",
                str(output),
                "-c",
                'model_reasoning_effort="xhigh"',
            ]
            if args.model:
                command.extend(["--model", args.model])
            request = (
                prompt
                + ("\n\n" + extra_instruction if extra_instruction else "")
                + "\n\n待生成任务：\n"
                + task_file.read_text("utf-8")
            )
            log_path = output.with_suffix(".log")
            with log_path.open("a", encoding="utf-8") as log_handle:
                completed = subprocess.run(
                    command + ["-"],
                    input=request,
                    text=True,
                    cwd=ROOT,
                    env={
                        key: value
                        for key, value in os.environ.items()
                        if key
                        not in {"DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY", "LANGSMITH_API_KEY"}
                    },
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if completed.returncode != 0:
                raise SystemExit(f"Codex 生成失败：{task_file.name}，exit={completed.returncode}")
        payload = json.loads(output.read_text("utf-8"))
        responses = payload.get("responses", [])
        if {item.get("task_id") for item in responses} != expected_ids:
            raise SystemExit(f"Codex 响应 task_id 不完整：{output}")
        return output

    def generate_batch(batch: Path) -> Path:
        return invoke_tasks(batch, output_dir / batch.name)

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(generate_batch, batch): batch for batch in batches}
        for future in as_completed(futures):
            output = future.result()
            print(f"已完成：{output.name}", flush=True)

    tasks = load_jsonl(WORK / "source-pack.jsonl")
    merged: list[dict] = []
    for batch in batches:
        payload = json.loads((output_dir / batch.name).read_text("utf-8"))
        merged.extend(payload.get("responses", []))
    response_map = {str(item["task_id"]): item for item in merged}
    task_map = {str(item["task_id"]): item for item in tasks}
    for repair_round in range(1, args.repair_rounds + 1):
        _accepted, rejected = validate_and_build(tasks, list(response_map.values()))
        rejected_ids = sorted({
            str(item["task_id"])
            for item in rejected
            if str(item.get("task_id", "")) in task_map
        })
        if not rejected_ids:
            break
        repair_dir = WORK / "repair-batches" / f"round-{repair_round:02d}"
        repair_dir.mkdir(parents=True, exist_ok=True)
        repair_files = []
        for index in range(0, len(rejected_ids), args.repair_batch_size):
            selected = [task_map[task_id] for task_id in rejected_ids[index : index + args.repair_batch_size]]
            task_file = repair_dir / f"repair-{index // args.repair_batch_size + 1:02d}.json"
            task_file.write_text(json.dumps(selected, ensure_ascii=False, indent=2), "utf-8")
            repair_files.append(task_file)
        instruction = (
            "这是自动校验失败后的盲修复。请完全重新表述问题，尤其禁止复用 target 中任何连续"
            "7个汉字或字符；不得只替换标点。仍需保持问题只能由 target 直接回答。"
        )
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {
                executor.submit(
                    invoke_tasks,
                    task_file,
                    output_dir / f"round-{repair_round:02d}-{task_file.name}",
                    extra_instruction=instruction,
                ): task_file
                for task_file in repair_files
            }
            for future in as_completed(futures):
                output = future.result()
                payload = json.loads(output.read_text("utf-8"))
                response_map.update({str(item["task_id"]): item for item in payload["responses"]})
                print(f"已修复：{output.name}", flush=True)
    merged = [response_map[task["task_id"]] for task in tasks if task["task_id"] in response_map]
    _atomic_text(WORK / "responses.jsonl", _jsonl(merged))
    print(f"Codex 生成完成：{len(merged)} 条；下一步执行 build")


def build(args) -> None:
    source_pack = Path(args.source_pack or WORK / "source-pack.jsonl")
    responses_path = Path(args.responses or WORK / "responses.jsonl")
    tasks = load_jsonl(source_pack)
    accepted, rejected = validate_and_build(tasks, load_jsonl(responses_path))
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "rejected.json").write_text(
        json.dumps(rejected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    dense = [item for item in accepted if item["metadata"]["challenge_set"] == "dense"]
    rerank = [item for item in accepted if item["metadata"]["challenge_set"] == "reranker"]
    expected_dense = sum(item["challenge_set"] == "dense" for item in tasks)
    expected_rerank = sum(item["challenge_set"] == "reranker" for item in tasks)
    if len(dense) != expected_dense or len(rerank) != expected_rerank or rejected:
        raise SystemExit(
            "挑战集校验未通过："
            f"dense={len(dense)}/{expected_dense}, reranker={len(rerank)}/{expected_rerank}, "
            f"rejected={len(rejected)}；详见 {WORK / 'rejected.json'}"
        )
    DATASETS.mkdir(parents=True, exist_ok=True)
    dense_path = DATASETS / f"{DENSE_DATASET}.jsonl"
    candidate_path = DATASETS / f"{RERANK_CANDIDATE_DATASET}.jsonl"
    existing_dense = load_jsonl(dense_path) if dense_path.is_file() else []
    existing_candidates = load_jsonl(candidate_path) if candidate_path.is_file() else []
    if existing_dense:
        assert_unchanged_prefix(
            existing_dense,
            dense,
            label="既有 Dense 数据集",
            ignored_metadata_keys={"generator_model"},
        )
        dense = [*existing_dense, *dense[len(existing_dense) :]]
    if existing_candidates:
        assert_unchanged_prefix(
            existing_candidates,
            rerank,
            label="既有 Reranker 候选集",
            ignored_input_keys={"question"},
            ignored_metadata_keys={"generator_model"},
        )
        rerank = [*existing_candidates, *rerank[len(existing_candidates) :]]
    previous_candidate_sha = (
        file_sha256(candidate_path) if candidate_path.is_file() else ""
    )
    previous_manifest_path = candidate_path.with_suffix(".manifest.json")
    previous_manifest = (
        json.loads(previous_manifest_path.read_text("utf-8"))
        if previous_manifest_path.is_file()
        else {}
    )
    write_frozen_dataset(
        dense_path,
        dense,
        source_file=Path(get_settings().law_data_path),
        selection_rule="all_valid_codex_dense_tasks_before_experiment",
    )
    write_frozen_dataset(
        candidate_path,
        rerank,
        source_file=Path(get_settings().law_data_path),
        selection_rule="all_valid_codex_reranker_candidates_before_hybrid_qualification",
        extra_manifest={
            "previous_candidate_count": previous_manifest.get(
                "previous_candidate_count", len(existing_candidates)
            ),
            "previous_candidate_sha256": previous_manifest.get(
                "previous_candidate_sha256", previous_candidate_sha
            ),
            "expanded_candidate_count": len(rerank),
            "expansion_seed": args.seed,
        },
    )
    print(f"已冻结 Dense 挑战集 {len(dense)} 条和 Reranker 候选集 {len(rerank)} 条")


def validate(args) -> None:
    datasets = args.datasets or [DENSE_DATASET, RERANK_CANDIDATE_DATASET]
    chunks = _chunks()
    failed = False
    for name in datasets:
        path = DATASETS / f"{name}.jsonl"
        examples = load_jsonl(path)
        errors = validate_examples_against_chunks(examples, chunks)
        print(f"{name}: {len(examples)} 条，errors={len(errors)}")
        if errors:
            failed = True
            print("\n".join(errors[:20]))
    if failed:
        raise SystemExit("挑战数据集校验失败")


def _qualification_configuration(
    settings,
    *,
    candidate_sha256: str,
    index_fingerprint: str,
) -> dict[str, Any]:
    return {
        "candidate_sha256": candidate_sha256,
        "index_fingerprint": index_fingerprint,
        "retrieval_mode": "hybrid",
        "rerank_enabled": False,
        "top_k": 12,
        "required_distractors": 2,
        "bm25_min_score": settings.rag_bm25_min_score,
        "dense_min_score": settings.rag_dense_min_score,
        "rrf_min_score": settings.rag_rrf_min_score,
    }


def _qualification_fingerprint(configuration: dict[str, Any]) -> str:
    payload = json.dumps(
        configuration, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_qualification_checkpoint(
    candidates: list[dict[str, Any]],
    *,
    fingerprint: str,
) -> list[dict[str, Any]]:
    checkpoint_path = WORK / "qualification-checkpoint.json"
    results_path = WORK / "qualification-results.jsonl"
    if not checkpoint_path.is_file() or not results_path.is_file():
        return []
    try:
        checkpoint = json.loads(checkpoint_path.read_text("utf-8"))
        results = load_jsonl(results_path)
        processed = int(checkpoint.get("processed_count", -1))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return []
    if checkpoint.get("qualification_fingerprint") != fingerprint:
        print("资格检查配置已变化，不复用旧 checkpoint", flush=True)
        return []
    if processed < 0 or processed > len(results) or processed > len(candidates):
        return []
    results = results[:processed]
    expected_ids = [str(item["metadata"]["task_id"]) for item in candidates[:processed]]
    if [str(item.get("task_id")) for item in results] != expected_ids:
        return []
    print(f"从资格检查 checkpoint 继续：{processed}/{len(candidates)}", flush=True)
    return results


def _save_qualification_checkpoint(
    results: list[dict[str, Any]],
    *,
    fingerprint: str,
    configuration: dict[str, Any],
    total_count: int,
) -> None:
    _atomic_text(WORK / "qualification-results.jsonl", _jsonl(results))
    _atomic_text(
        WORK / "qualification-checkpoint.json",
        json.dumps({
            "qualification_fingerprint": fingerprint,
            "processed_count": len(results),
            "total_count": total_count,
            "configuration": configuration,
        }, ensure_ascii=False, indent=2),
    )


def _qualification_summary(
    results: list[dict[str, Any]],
    *,
    configuration: dict[str, Any],
    fingerprint: str,
    requested_size: int,
) -> dict[str, Any]:
    qualified = [item for item in results if item["qualified"]]
    reasons = Counter(
        reason
        for item in results
        for reason in ([item["rejection_reason"]] if item["rejection_reason"] else [])
    )
    categories: dict[str, Counter] = defaultdict(Counter)
    for item in results:
        categories[str(item["category"])][
            "qualified" if item["qualified"] else "rejected"
        ] += 1
    return {
        "qualification_fingerprint": fingerprint,
        "configuration": configuration,
        "processed_count": len(results),
        "qualified_count": len(qualified),
        "rejected_count": len(results) - len(qualified),
        "requested_final_size": requested_size,
        "qualification_rate": round(len(qualified) / len(results), 6) if results else 0,
        "rejection_reasons": dict(sorted(reasons.items())),
        "categories": {
            category: dict(counts) for category, counts in sorted(categories.items())
        },
    }


def _select_qualified_candidates(
    candidates: list[dict[str, Any]],
    results: list[dict[str, Any]],
    *,
    requested_size: int,
) -> list[dict[str, Any]]:
    qualified_records = [item for item in results if item["qualified"]]
    if len(qualified_records) < requested_size:
        raise ValueError(
            f"符合资格的 Reranker 样本不足：{len(qualified_records)}/{requested_size}"
        )
    by_task = {
        str(item["metadata"]["task_id"]): item for item in candidates
    }
    accepted: list[dict[str, Any]] = []
    for record in qualified_records[:requested_size]:
        item = json.loads(json.dumps(by_task[str(record["task_id"])], ensure_ascii=False))
        item["metadata"].update({
            "qualification_status": "qualified",
            "qualification_rule": "gold_in_hybrid_top12_and_at_least_2_declared_distractors",
            "hybrid_gold_rank": record["hybrid_gold_rank"],
            "retrieved_hard_distractor_count": record[
                "retrieved_hard_distractor_count"
            ],
        })
        accepted.append(item)
    return accepted


async def qualify(args) -> None:
    settings = get_settings().model_copy(
        update={"rag_retrieval_mode": "hybrid", "rag_rerank_enabled": False}
    )
    candidate_path = DATASETS / f"{RERANK_CANDIDATE_DATASET}.jsonl"
    candidates = load_jsonl(candidate_path)
    if len(candidates) < DEFAULT_RERANK_CANDIDATE_SIZE:
        raise SystemExit(
            "Reranker 候选池不足600条；请先按固定种子扩容，不能重复运行300条资格筛选"
        )
    engine = await asyncio.to_thread(LawSearchEngine, settings)
    results: list[dict[str, Any]] = []
    engine_status: dict[str, Any] = {}
    try:
        await engine.initialize_index(wait=True)
        engine_status = engine.status()
        if not engine_status.get("dense_enabled"):
            raise RuntimeError("Reranker 候选资格检查要求有效 Dense 索引")
        configuration = _qualification_configuration(
            settings,
            candidate_sha256=file_sha256(candidate_path),
            index_fingerprint=str(engine_status.get("fingerprint") or ""),
        )
        fingerprint = _qualification_fingerprint(configuration)
        results = _load_qualification_checkpoint(
            candidates, fingerprint=fingerprint
        )
        for example in candidates[len(results):]:
            query = str(example["inputs"]["question"])
            retrieval_results = await engine.search(
                query, top_k=12, retrieval_mode="hybrid"
            )
            ranked_chunks = [str(item.get("chunk_id")) for item in retrieval_results]
            gold = str(example["outputs"]["expected_chunk_ids"][0])
            distractors = set(example["metadata"].get("hard_distractor_chunk_ids", []))
            retrieved_distractors = distractors.intersection(ranked_chunks)
            gold_in_top_12 = gold in ranked_chunks
            qualified = gold_in_top_12 and len(retrieved_distractors) >= 2
            rejection_reason = None
            if not gold_in_top_12:
                rejection_reason = "gold_not_in_hybrid_top12"
            elif len(retrieved_distractors) < 2:
                rejection_reason = "fewer_than_2_declared_distractors"
            results.append({
                "task_id": example["metadata"]["task_id"],
                "category": example["metadata"]["category"],
                "gold_in_top_12": gold_in_top_12,
                "hybrid_gold_rank": (
                    ranked_chunks.index(gold) + 1 if gold_in_top_12 else None
                ),
                "retrieved_hard_distractor_count": len(retrieved_distractors),
                "qualified": qualified,
                "rejection_reason": rejection_reason,
            })
            if len(results) % QUALIFICATION_BATCH_SIZE == 0:
                _save_qualification_checkpoint(
                    results,
                    fingerprint=fingerprint,
                    configuration=configuration,
                    total_count=len(candidates),
                )
                print(f"资格检查进度：{len(results)}/{len(candidates)}", flush=True)
        _save_qualification_checkpoint(
            results,
            fingerprint=fingerprint,
            configuration=configuration,
            total_count=len(candidates),
        )
    finally:
        await engine.close()

    WORK.mkdir(parents=True, exist_ok=True)
    rejected = [item for item in results if not item["qualified"]]
    _atomic_text(
        WORK / "qualification-rejected.json",
        json.dumps(rejected, ensure_ascii=False, indent=2),
    )
    summary = _qualification_summary(
        results,
        configuration=configuration,
        fingerprint=fingerprint,
        requested_size=args.rerank_size,
    )
    _atomic_text(
        WORK / "qualification-summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2),
    )
    qualified_records = [item for item in results if item["qualified"]]
    try:
        accepted = _select_qualified_candidates(
            candidates, results, requested_size=args.rerank_size
        )
    except ValueError as exc:
        raise SystemExit(
            f"{exc}；"
            "不得放宽规则或根据 BGE 结果挑选，请将候选池按相同规则扩容到800条"
        ) from exc
    manifest = write_frozen_dataset(
        DATASETS / f"{RERANK_DATASET}.jsonl",
        accepted,
        source_file=Path(settings.law_data_path),
        selection_rule="gold_in_hybrid_top12_and_at_least_2_declared_distractors",
    )
    manifest_path = DATASETS / f"{RERANK_DATASET}.manifest.json"
    manifest.update({
        "candidate_dataset": RERANK_CANDIDATE_DATASET,
        "candidate_dataset_sha256": file_sha256(candidate_path),
        "candidate_count": len(candidates),
        "qualified_count": len(qualified_records),
        "qualification_fingerprint": fingerprint,
        "index_fingerprint": engine_status.get("fingerprint"),
        "retrieval_mode": "hybrid",
        "rerank_enabled_during_qualification": False,
    })
    temporary = manifest_path.with_suffix(f".json.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
    os.replace(temporary, manifest_path)
    print(f"已冻结 Reranker 挑战集：{len(accepted)} 条")


def parse_args():
    parser = argparse.ArgumentParser(description="构建简历导向但可审计的 RAG 挑战集")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--dense-size", type=int, default=DEFAULT_DENSE_SIZE)
    prepare_parser.add_argument(
        "--rerank-candidate-size", type=int, default=DEFAULT_RERANK_CANDIDATE_SIZE
    )
    prepare_parser.add_argument("--batch-size", type=int, default=50)
    prepare_parser.add_argument("--seed", type=int, default=42)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--source-pack", default="")
    build_parser.add_argument("--responses", default="")
    build_parser.add_argument("--seed", type=int, default=42)
    codex_parser = subparsers.add_parser("generate-codex")
    codex_parser.add_argument("--codex-command", default="codex")
    codex_parser.add_argument("--model", default="")
    codex_parser.add_argument("--force", action="store_true")
    codex_parser.add_argument("--concurrency", type=int, choices=range(1, 5), default=3)
    codex_parser.add_argument("--repair-rounds", type=int, default=3)
    codex_parser.add_argument("--repair-batch-size", type=int, default=25)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("datasets", nargs="*")
    qualify_parser = subparsers.add_parser("qualify-reranker")
    qualify_parser.add_argument("--rerank-size", type=int, default=DEFAULT_RERANK_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "generate-codex":
        generate_codex(args)
    elif args.command == "build":
        build(args)
    elif args.command == "validate":
        validate(args)
    else:
        asyncio.run(qualify(args))


if __name__ == "__main__":
    main()
