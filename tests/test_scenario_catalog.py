import json
from pathlib import Path

import pytest

from backend.app.core.config import Settings
from backend.app.evaluation import scenario_catalog as catalog_module
from backend.app.evaluation.conversation_scenarios import file_sha256
from backend.app.evaluation.scenario_catalog import ScenarioCatalog, ScenarioCatalogError

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "evals" / "conversations" / "lawstation-dialogue-scenarios-v2.jsonl"


def _catalog(tmp_path, monkeypatch, *, mutate_manifest=None):
    allowed = tmp_path / "evals" / "conversations"
    allowed.mkdir(parents=True)
    path = allowed / "observer.jsonl"
    scenario = json.loads(FROZEN.read_text("utf-8").splitlines()[0])
    path.write_text(json.dumps(scenario, ensure_ascii=False) + "\n", "utf-8")
    manifest = {
        "dataset": "observer-v1",
        "schema_version": "1.0",
        "dataset_sha256": file_sha256(path),
        "sample_count": 1,
        "status": "frozen",
        "synthetic": True,
        "human_verified": False,
    }
    if mutate_manifest:
        mutate_manifest(manifest)
    path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), "utf-8"
    )
    monkeypatch.setattr(catalog_module, "ROOT", tmp_path)
    monkeypatch.setattr(catalog_module, "SCENARIO_ROOT", allowed)
    settings = Settings(
        _env_file=None,
        test_scenarios_enabled=True,
        test_scenario_data_paths=[str(path)],
    )
    return ScenarioCatalog(settings)


def test_catalog_exposes_safe_fixture_metadata(tmp_path, monkeypatch):
    catalog = _catalog(tmp_path, monkeypatch)
    dataset = catalog.datasets()[0]
    summary = catalog.scenario_summaries(dataset["id"])[0]
    detail = catalog.scenario(dataset["id"], summary["scenario_id"])

    assert dataset["sample_count"] == 1
    assert "fixtures" not in detail
    assert detail["fixtures_applied"] is False
    assert isinstance(detail["fixture_types"], list)


def test_catalog_rejects_sha_mismatch(tmp_path, monkeypatch):
    with pytest.raises(ScenarioCatalogError, match="SHA256"):
        _catalog(
            tmp_path,
            monkeypatch,
            mutate_manifest=lambda value: value.update(dataset_sha256="wrong"),
        )


def test_catalog_rejects_path_outside_allowlist(tmp_path, monkeypatch):
    outside = tmp_path / "outside.jsonl"
    outside.write_text("{}\n", "utf-8")
    allowed = tmp_path / "evals" / "conversations"
    allowed.mkdir(parents=True)
    monkeypatch.setattr(catalog_module, "ROOT", tmp_path)
    monkeypatch.setattr(catalog_module, "SCENARIO_ROOT", allowed)
    settings = Settings(
        _env_file=None,
        test_scenarios_enabled=True,
        test_scenario_data_paths=[str(outside)],
    )

    with pytest.raises(ScenarioCatalogError, match="路径越界"):
        ScenarioCatalog(settings)


def test_disabled_catalog_does_not_read_paths():
    catalog = ScenarioCatalog(Settings(_env_file=None, test_scenarios_enabled=False))
    assert catalog.status() == {
        "enabled": False,
        "dataset_count": 0,
        "status": "disabled",
    }
