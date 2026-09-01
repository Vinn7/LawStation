"""Validated, read-only catalog for the opt-in dialogue scenario observer."""

from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

from backend.app.core.config import Settings, get_settings
from backend.app.evaluation.conversation_scenarios import file_sha256, validate_scenarios

ROOT = Path(__file__).resolve().parents[3]
SCENARIO_ROOT = ROOT / "evals" / "conversations"


class ScenarioCatalogError(RuntimeError):
    pass


class ScenarioCatalog:
    """Load frozen scenario JSONL files from an explicit project-local allowlist."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.enabled = self.settings.test_scenarios_enabled
        self._datasets: dict[str, dict[str, Any]] = {}
        if self.enabled:
            self._load()

    def _configured_paths(self) -> list[str]:
        paths = list(self.settings.test_scenario_data_paths)
        if not paths and self.settings.test_scenario_data_path:
            paths = [self.settings.test_scenario_data_path]
        if not paths:
            raise ScenarioCatalogError("测试场景已启用，但未配置数据集白名单")
        return paths

    @staticmethod
    def _resolve_path(raw_path: str) -> Path:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        resolved = candidate.resolve()
        allowed_root = SCENARIO_ROOT.resolve()
        if not resolved.is_relative_to(allowed_root):
            raise ScenarioCatalogError(f"测试场景路径越界：{raw_path}")
        if resolved.suffix != ".jsonl":
            raise ScenarioCatalogError(f"测试场景必须是JSONL：{raw_path}")
        if not resolved.is_file():
            raise ScenarioCatalogError(f"测试场景文件不存在：{raw_path}")
        return resolved

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        scenarios: list[dict[str, Any]] = []
        try:
            for line_number, line in enumerate(path.read_text("utf-8").splitlines(), 1):
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"第{line_number}行不是JSON对象")
                    scenarios.append(value)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise ScenarioCatalogError(f"无法读取测试场景：{path.name}") from exc
        return scenarios

    def _load(self) -> None:
        seen_paths: set[Path] = set()
        for raw_path in self._configured_paths():
            path = self._resolve_path(raw_path)
            if path in seen_paths:
                raise ScenarioCatalogError(f"测试场景路径重复：{path.name}")
            seen_paths.add(path)
            manifest_path = path.with_suffix(".manifest.json")
            if not manifest_path.is_file():
                raise ScenarioCatalogError(f"测试场景缺少manifest：{manifest_path.name}")
            try:
                manifest = json.loads(manifest_path.read_text("utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ScenarioCatalogError(f"测试场景manifest无效：{manifest_path.name}") from exc
            dataset_id = str(manifest.get("dataset") or path.stem).strip()
            if not dataset_id or dataset_id in self._datasets:
                raise ScenarioCatalogError(f"测试场景Dataset ID重复或为空：{dataset_id}")
            scenarios = self._read_jsonl(path)
            errors = validate_scenarios(scenarios)
            if errors:
                raise ScenarioCatalogError(f"测试场景结构无效：{errors[0]}")
            expected_sha = str(manifest.get("dataset_sha256") or "")
            actual_sha = file_sha256(path)
            if not expected_sha or expected_sha != actual_sha:
                raise ScenarioCatalogError(f"测试场景SHA256不匹配：{path.name}")
            if manifest.get("status") != "frozen":
                raise ScenarioCatalogError(f"测试场景尚未冻结：{path.name}")
            if int(manifest.get("sample_count") or -1) != len(scenarios):
                raise ScenarioCatalogError(f"测试场景样本数与manifest不一致：{path.name}")
            scenario_map = {str(item["scenario_id"]): item for item in scenarios}
            if len(scenario_map) != len(scenarios):
                raise ScenarioCatalogError(f"测试场景ID重复：{path.name}")
            self._datasets[dataset_id] = {
                "id": dataset_id,
                "schema_version": str(manifest.get("schema_version") or ""),
                "sha256": actual_sha,
                "sample_count": len(scenarios),
                "synthetic": bool(manifest.get("synthetic", True)),
                "human_verified": bool(manifest.get("human_verified", False)),
                "categories": dict(sorted(Counter(item.get("category", "unknown") for item in scenarios).items())),
                "scenarios": scenario_map,
            }

    def datasets(self) -> list[dict[str, Any]]:
        return [
            {
                **{key: value for key, value in dataset.items() if key != "scenarios"},
                "step_timeout_seconds": self.settings.test_scenario_step_timeout_seconds,
            }
            for dataset in self._datasets.values()
        ]

    def scenario_summaries(self, dataset_id: str) -> list[dict[str, Any]] | None:
        dataset = self._datasets.get(dataset_id)
        if dataset is None:
            return None
        return [
            {
                "scenario_id": scenario["scenario_id"],
                "title": scenario["title"],
                "category": scenario["category"],
                "description": scenario.get("description", ""),
                "actors": list(scenario.get("actors", [])),
                "step_count": len(scenario.get("steps", [])),
                "preconditions": list(scenario.get("preconditions", [])),
                "fixture_types": sorted((scenario.get("fixtures") or {}).keys()),
            }
            for scenario in dataset["scenarios"].values()
        ]

    def scenario(self, dataset_id: str, scenario_id: str) -> dict[str, Any] | None:
        dataset = self._datasets.get(dataset_id)
        if dataset is None:
            return None
        scenario = dataset["scenarios"].get(scenario_id)
        if scenario is None:
            return None
        safe = deepcopy(scenario)
        fixtures = safe.pop("fixtures", {}) or {}
        safe["fixture_types"] = sorted(fixtures.keys())
        safe["fixtures_applied"] = False
        safe["fixture_notice"] = (
            "本模式使用真实服务，不注入离线Fixture。依赖Fixture的断言将标记为不可判定。"
            if fixtures else ""
        )
        return safe

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "dataset_count": len(self._datasets),
            "status": "ready" if self.enabled else "disabled",
        }
