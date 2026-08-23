import asyncio
import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from backend.app.evaluation.profiles import (
    case_hash,
    dataset_sha256,
    effective_seed,
    load_cases,
    select_cases,
)
from backend.app.evaluation.reporting import ReportRun
from scripts import run_langsmith_eval, run_staged_langsmith_eval


def test_report_runs_are_unique_and_write_matching_json_csv(tmp_path):
    first = ReportRun.create(tmp_path, "learn", "Asia/Singapore")
    second = ReportRun.create(tmp_path, "learn", "Asia/Singapore")

    assert first.run_id != second.run_id
    assert first.path != second.path
    assert datetime.fromisoformat(first.started_at.isoformat()).utcoffset().total_seconds() == 8 * 3600

    json_path, csv_path = first.write_metrics(
        "learn.json",
        {"metrics": {"schema_validity": {"sample_count": 1, "mean": 1.0}}},
    )
    first.complete()

    assert json_path.stem == csv_path.stem == "learn"
    assert not list(first.path.glob("*.tmp"))
    manifest = json.loads((first.path / "run-manifest.json").read_text("utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["timezone"] == "Asia/Singapore"


def test_failed_run_updates_latest_but_not_latest_success(tmp_path):
    successful = ReportRun.create(tmp_path, "learn", "Asia/Singapore")
    successful.complete()
    failed = ReportRun.create(tmp_path, "smoke", "Asia/Singapore")
    failed.fail("failed", stage="smoke.component", error=RuntimeError("safe failure"))

    latest = json.loads((tmp_path / "latest.json").read_text("utf-8"))
    latest_success = json.loads((tmp_path / "latest-success.json").read_text("utf-8"))
    assert latest["run_id"] == failed.run_id
    assert latest["status"] == "failed"
    assert latest_success["run_id"] == successful.run_id
    assert latest_success["status"] == "completed"


def test_attach_validates_run_identity_and_timezone(tmp_path):
    report_run = ReportRun.create(tmp_path, "compare", "Asia/Singapore")
    attached = ReportRun.attach(report_run.path, "Asia/Singapore")
    assert attached.run_id == report_run.run_id
    assert attached.managed is False
    with pytest.raises(ValueError, match="时区不一致"):
        ReportRun.attach(report_run.path, "UTC")
    with pytest.raises(ValueError, match="非法评测 run_id"):
        ReportRun.create(
            tmp_path,
            "learn",
            "Asia/Singapore",
            "20260823-153045-123456-smoke",
        )


def _args(tmp_path, **changes):
    values = {
        "profile": "learn",
        "dataset": "",
        "mode": None,
        "max_examples": 0,
        "sample_categories": [],
        "upload_results": False,
        "repetitions": 1,
        "concurrency": 1,
        "run_dir": "",
        "run_id": "",
        "no_report": False,
        "output": "",
        "plan_only": False,
        "report_root": str(tmp_path),
        "report_name": "",
    }
    values.update(changes)
    return SimpleNamespace(**values)


def test_execute_archives_result_and_supports_no_report(monkeypatch, tmp_path):
    settings = SimpleNamespace(
        eval_report_root=str(tmp_path),
        eval_report_timezone="Asia/Singapore",
    )
    monkeypatch.setattr(run_langsmith_eval, "get_settings", lambda: settings)

    async def fake_run(_args):
        return {"profile": "learn", "metrics": {}}

    monkeypatch.setattr(run_langsmith_eval, "run", fake_run)
    payload = asyncio.run(run_langsmith_eval.execute(_args(tmp_path)))
    assert payload["run_id"]
    assert (tmp_path / payload["run_id"] / "learn.json").is_file()

    before = sorted(tmp_path.iterdir())
    payload = asyncio.run(
        run_langsmith_eval.execute(_args(tmp_path, no_report=True))
    )
    assert "run_id" not in payload
    assert sorted(tmp_path.iterdir()) == before


def test_execute_keeps_legacy_output_and_marks_manifest(monkeypatch, tmp_path):
    report_root = tmp_path / "runs"
    legacy_output = tmp_path / "legacy" / "result.json"
    settings = SimpleNamespace(
        eval_report_root=str(report_root),
        eval_report_timezone="Asia/Singapore",
    )
    monkeypatch.setattr(run_langsmith_eval, "get_settings", lambda: settings)

    async def fake_run(_args):
        return {"profile": "learn", "metrics": {}}

    monkeypatch.setattr(run_langsmith_eval, "run", fake_run)
    payload = asyncio.run(
        run_langsmith_eval.execute(
            _args(report_root, output=str(legacy_output))
        )
    )

    assert legacy_output.is_file()
    manifest = json.loads(
        (report_root / payload["run_id"] / "run-manifest.json").read_text("utf-8")
    )
    assert manifest["compatibility_output"] == str(legacy_output.resolve())


def test_plan_only_does_not_create_report_or_latest(monkeypatch, tmp_path):
    settings = SimpleNamespace(
        eval_report_root=str(tmp_path),
        eval_report_timezone="Asia/Singapore",
    )
    monkeypatch.setattr(run_langsmith_eval, "get_settings", lambda: settings)

    async def fake_run(_args):
        return {"profile": "compare", "planned": {"evaluation_traces": 0}}

    monkeypatch.setattr(run_langsmith_eval, "run", fake_run)
    asyncio.run(
        run_langsmith_eval.execute(
            _args(tmp_path, profile="compare", plan_only=True)
        )
    )
    assert not tmp_path.exists() or not list(tmp_path.iterdir())


def test_reusable_report_requires_matching_sample_and_versions(monkeypatch, tmp_path):
    settings = SimpleNamespace(eval_report_root=str(tmp_path))
    monkeypatch.setattr(run_staged_langsmith_eval, "get_settings", lambda: settings)
    seed = effective_seed(42)
    cases = select_cases(
        load_cases("lawstation-agent-v3"),
        limit=1,
        categories=["matched"],
        seed=seed,
    )
    report = {
        "dataset": "lawstation-agent-v3",
        "run_count": 1,
        "dataset_sha256": dataset_sha256("lawstation-agent-v3"),
        "batch": {
            "seed": seed,
            "examples": [{"content_sha256": case_hash(cases[0])}],
        },
        "rag_mode": "hybrid",
        "review_mode": "auto",
        "metadata": {
            "graph_version": "three-agent-v2-chunk-evidence-fast-review",
            "prompt_version": "legal-consultation-v2-no-match-safe",
        },
        "missing_required_metrics": [],
        "langsmith_export_complete": True,
        "metrics": {},
    }
    archived = ReportRun.create(tmp_path, "compare", "Asia/Singapore")
    archived.write_metrics("baseline.json", report)
    archived.complete()

    reused = run_staged_langsmith_eval.reusable_report(
        "baseline.json",
        dataset="lawstation-agent-v3",
        runs=1,
        categories=["matched"],
        seed=42,
        rag_mode="hybrid",
        review_mode="auto",
    )
    assert reused is not None
    assert reused["_reuse_source"]["run_id"] == archived.run_id

    assert run_staged_langsmith_eval.reusable_report(
        "baseline.json",
        dataset="lawstation-agent-v3",
        runs=1,
        categories=["matched"],
        seed=43,
        rag_mode="hybrid",
        review_mode="auto",
    ) is None
