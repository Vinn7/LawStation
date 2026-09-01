"""Generate, validate, and freeze source-controlled multi-turn dialogue scenarios."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI

from backend.app.core.config import get_settings
from backend.app.evaluation.conversation_scenarios import (
    DATASET_VERSION,
    PROMPT_VERSION,
    GeneratedBlueprintResponse,
    attach_sources,
    blueprint_definitions,
    file_sha256,
    generation_prompt,
    materialize_scenarios,
    sha256_json,
    validate_scenarios,
)
from mcp_servers.law_rag.engine import load_chunks

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "evals" / "conversations"
BLUEPRINTS = OUTPUT / "blueprints-v1.json"
CANDIDATES = OUTPUT / "generated-candidates-v1.jsonl"
FROZEN = OUTPUT / f"{DATASET_VERSION}.jsonl"
MANIFEST = OUTPUT / f"{DATASET_VERSION}.manifest.json"
VALIDATION = OUTPUT / "generation-validation.json"
CHECKPOINTS = OUTPUT / ".checkpoints"


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _jsonl(items: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(item, ensure_ascii=False) for item in items) + ("\n" if items else "")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def _response_text(response: Any) -> str:
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        ).strip()
    return str(content or "").strip()


def prepare(_args) -> None:
    settings = get_settings()
    chunks = load_chunks(
        Path(settings.law_data_path),
        settings.index_chunk_max_chars,
        settings.index_chunk_overlap_chars,
    )
    blueprints = attach_sources(
        blueprint_definitions(), chunks, seed=settings.test_scenario_generator_seed
    )
    if len(blueprints) != 18:
        raise SystemExit(f"场景蓝图数量错误：{len(blueprints)}/18")
    if BLUEPRINTS.is_file():
        existing = json.loads(BLUEPRINTS.read_text("utf-8"))
        if existing != blueprints:
            raise SystemExit("既有blueprints-v1.json与当前模板不一致，拒绝静默覆盖")
    else:
        _atomic_text(BLUEPRINTS, json.dumps(blueprints, ensure_ascii=False, indent=2) + "\n")
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    print(f"蓝图已准备：{BLUEPRINTS}（{len(blueprints)}类）")


def _checkpoint_responses(blueprints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    responses: list[dict[str, Any]] = []
    for blueprint in blueprints:
        path = CHECKPOINTS / f"{blueprint['id']}.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text("utf-8"))
        parsed = GeneratedBlueprintResponse.model_validate(payload["response"])
        if parsed.blueprint_id != blueprint["id"]:
            raise RuntimeError(f"checkpoint蓝图不匹配：{path.name}")
        if payload.get("blueprint_sha256") != sha256_json(blueprint):
            raise RuntimeError(f"checkpoint蓝图指纹已变化：{path.name}")
        responses.append(parsed.model_dump())
    return responses


def _write_candidates(blueprints: list[dict[str, Any]], model_name: str) -> tuple[list[dict], list[dict]]:
    responses = _checkpoint_responses(blueprints)
    candidates, rejected = materialize_scenarios(
        blueprints, responses, generator_model=model_name
    )
    _atomic_text(CANDIDATES, _jsonl(candidates))
    _atomic_text(
        VALIDATION,
        json.dumps({
            "status": "valid" if not rejected else "partial",
            "blueprint_count": len(blueprints),
            "response_count": len(responses),
            "candidate_count": len(candidates),
            "rejected_count": len(rejected),
            "rejected": rejected,
        }, ensure_ascii=False, indent=2) + "\n",
    )
    return candidates, rejected


async def generate(args) -> None:
    if not BLUEPRINTS.is_file():
        raise SystemExit("请先执行prepare")
    settings = get_settings()
    if not settings.deepseek_api_key:
        raise SystemExit("未配置DEEPSEEK_API_KEY，无法生成对话样例")
    blueprints = json.loads(BLUEPRINTS.read_text("utf-8"))
    model_name = settings.test_scenario_generator_model or settings.deepseek_model
    model = ChatOpenAI(
        model=model_name,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        streaming=False,
        temperature=settings.test_scenario_generator_temperature,
        timeout=settings.llm_request_timeout_seconds,
        max_retries=settings.llm_max_retries,
        max_tokens=4096,
        extra_body={"thinking": {"type": "disabled"}},
    ).bind(response_format={"type": "json_object"})
    existing_responses = _checkpoint_responses(blueprints)
    existing = {item["blueprint_id"] for item in existing_responses}
    pending = [item for item in blueprints if item["id"] not in existing]
    call_budget = args.max_calls or settings.test_scenario_generator_max_calls
    limit = min(call_budget, len(pending))
    failures: list[dict[str, str]] = []
    calls = 0
    for blueprint in pending[:limit]:
        calls += 1
        try:
            response = await model.ainvoke([
                ("system", "只返回合法JSON，不得输出Markdown或解释。"),
                ("human", generation_prompt(blueprint, settings.test_scenario_variants_per_blueprint)),
            ])
            raw = _response_text(response)
            parsed = GeneratedBlueprintResponse.model_validate(json.loads(raw))
            if parsed.blueprint_id != blueprint["id"]:
                raise ValueError("模型返回的blueprint_id不匹配")
            if len(parsed.variants) != settings.test_scenario_variants_per_blueprint:
                raise ValueError("模型返回的变体数量不正确")
            path = CHECKPOINTS / f"{blueprint['id']}.json"
            _atomic_text(path, json.dumps({
                "blueprint_sha256": sha256_json(blueprint),
                "generator_model": model_name,
                "prompt_version": PROMPT_VERSION,
                "response": parsed.model_dump(),
            }, ensure_ascii=False, indent=2) + "\n")
            print(f"已生成：{blueprint['id']}（{len(parsed.variants)}个变体）", flush=True)
        except Exception as exc:  # noqa: BLE001 - independent blueprints must continue after one provider failure
            failures.append({"blueprint_id": blueprint["id"], "error_type": type(exc).__name__})
            print(f"生成失败：{blueprint['id']}（{type(exc).__name__}）", flush=True)
    candidates, rejected = _write_candidates(blueprints, model_name)
    repair_targets: list[tuple[str, int]] = []
    repair_reasons: dict[tuple[str, int], list[str]] = {}
    for item in rejected:
        scenario_id = str(item.get("scenario_id", ""))
        blueprint_id = str(item.get("blueprint_id", ""))
        suffix = scenario_id.removeprefix(f"{blueprint_id}-")
        if blueprint_id and suffix.isdigit():
            target = (blueprint_id, int(suffix))
            repair_targets.append(target)
            repair_reasons[target] = list(item.get("reasons", []))
    blueprint_map = {item["id"]: item for item in blueprints}
    for blueprint_id, variant_number in repair_targets[:max(0, call_budget - calls)]:
        blueprint = blueprint_map[blueprint_id]
        calls += 1
        try:
            repair_instruction = (
                generation_prompt(blueprint, 1)
                + f"\n这是定向修复：variants数组只能包含variant={variant_number}，"
                "必须彻底改写原话，并特别避免复用source连续原文。"
            )
            response = await model.ainvoke([
                ("system", "只返回合法JSON，不得输出Markdown或解释。"),
                ("human", repair_instruction),
            ])
            parsed = GeneratedBlueprintResponse.model_validate(json.loads(_response_text(response)))
            if parsed.blueprint_id != blueprint_id or [item.variant for item in parsed.variants] != [variant_number]:
                raise ValueError("定向修复返回了错误的蓝图或变体编号")
            path = CHECKPOINTS / f"{blueprint_id}.json"
            payload = json.loads(path.read_text("utf-8"))
            current = GeneratedBlueprintResponse.model_validate(payload["response"])
            kept = [item for item in current.variants if item.variant != variant_number]
            merged = GeneratedBlueprintResponse(
                blueprint_id=blueprint_id,
                variants=sorted([*kept, *parsed.variants], key=lambda item: item.variant),
            )
            payload["response"] = merged.model_dump()
            payload.setdefault("repaired_variants", []).append(variant_number)
            payload.setdefault("repair_history", []).append({
                "variant": variant_number,
                "reasons": repair_reasons.get((blueprint_id, variant_number), []),
            })
            _atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            print(f"已定向修复：{blueprint_id}-{variant_number:02d}", flush=True)
        except Exception as exc:  # noqa: BLE001 - preserve all accepted variants when one repair fails
            failures.append({
                "blueprint_id": blueprint_id,
                "variant": str(variant_number),
                "error_type": type(exc).__name__,
            })
            print(f"修复失败：{blueprint_id}-{variant_number:02d}（{type(exc).__name__}）", flush=True)
    candidates, rejected = _write_candidates(blueprints, model_name)
    completed = len(_checkpoint_responses(blueprints))
    print(json.dumps({
        "model": model_name,
        "model_calls_this_run": calls,
        "completed_blueprints": completed,
        "remaining_blueprints": len(blueprints) - completed,
        "candidate_count": len(candidates),
        "rejected_count": len(rejected),
        "generation_failures": failures,
    }, ensure_ascii=False, indent=2))


def validate(_args) -> None:
    if not BLUEPRINTS.is_file():
        raise SystemExit("缺少blueprints-v1.json")
    blueprints = json.loads(BLUEPRINTS.read_text("utf-8"))
    settings = get_settings()
    model_name = settings.test_scenario_generator_model or settings.deepseek_model
    candidates, rejected = _write_candidates(blueprints, model_name)
    errors = validate_scenarios(candidates)
    expected = len(blueprints) * settings.test_scenario_variants_per_blueprint
    summary = {
        "status": "valid" if not errors and not rejected and len(candidates) == expected else "partial",
        "expected_count": expected,
        "candidate_count": len(candidates),
        "rejected_count": len(rejected),
        "errors": errors,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit("对话样例结构校验失败")


def freeze(_args) -> None:
    validate(_args)
    settings = get_settings()
    candidates = _load_jsonl(CANDIDATES)
    blueprints = json.loads(BLUEPRINTS.read_text("utf-8"))
    expected = len(blueprints) * settings.test_scenario_variants_per_blueprint
    validation = json.loads(VALIDATION.read_text("utf-8"))
    if len(candidates) != expected or validation.get("rejected_count"):
        raise SystemExit(f"有效样例不足，保留当前结果但不冻结：{len(candidates)}/{expected}")
    _atomic_text(FROZEN, _jsonl(candidates))
    checkpoints = _checkpoint_responses(blueprints)
    repair_call_count = sum(
        len(json.loads(path.read_text("utf-8")).get("repaired_variants", []))
        for path in CHECKPOINTS.glob("*.json")
    )
    repair_history = [
        {"blueprint_id": path.stem, **item}
        for path in CHECKPOINTS.glob("*.json")
        for item in json.loads(path.read_text("utf-8")).get("repair_history", [])
    ]
    models = sorted({
        json.loads(path.read_text("utf-8")).get("generator_model", "")
        for path in CHECKPOINTS.glob("*.json")
    })
    manifest = {
        "dataset": DATASET_VERSION,
        "schema_version": "dialogue-scenario-v1",
        "status": "frozen",
        "synthetic": True,
        "human_verified": False,
        "created_at": datetime.now(UTC).isoformat(),
        "prompt_version": PROMPT_VERSION,
        "generator_models": models,
        "blueprint_count": len(blueprints),
        "sample_count": len(candidates),
        "variants_per_blueprint": settings.test_scenario_variants_per_blueprint,
        "generator_seed": settings.test_scenario_generator_seed,
        "blueprints_sha256": file_sha256(BLUEPRINTS),
        "dataset_sha256": file_sha256(FROZEN),
        "generation_checkpoint_count": len(checkpoints),
        "model_call_count": len(checkpoints) + repair_call_count,
        "initial_model_call_count": len(checkpoints),
        "repair_model_call_count": repair_call_count,
        "historical_rejected_count": len(repair_history),
        "repair_history": repair_history,
        "rejected_count": 0,
        "category_counts": dict(sorted(Counter(item["category"] for item in candidates).items())),
        "limitations": [
            "由DeepSeek生成并经确定性规则校验，不是真实用户对话。",
            "未经过律师人工标注，不能用于声明法律结论准确率。",
            "场景断言用于Agent流程与安全约束测试。",
        ],
    }
    _atomic_text(MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"已冻结：{FROZEN}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=["prepare", "generate", "validate", "freeze"])
    result.add_argument("--max-calls", type=int, default=None)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.max_calls is not None and args.max_calls < 1:
        raise SystemExit("--max-calls必须大于0")
    if args.command == "prepare":
        prepare(args)
    elif args.command == "generate":
        asyncio.run(generate(args))
    elif args.command == "validate":
        validate(args)
    else:
        freeze(args)


if __name__ == "__main__":
    main()
