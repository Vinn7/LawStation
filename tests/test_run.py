import os
from types import SimpleNamespace

import pytest

import run


def test_frontend_staleness(tmp_path, monkeypatch):
    frontend = tmp_path / "frontend"
    source = frontend / "src" / "main.tsx"
    output = frontend / "dist" / "index.html"
    source.parent.mkdir(parents=True)
    output.parent.mkdir(parents=True)
    source.write_text("source")
    output.write_text("built")
    monkeypatch.setattr(run, "FRONTEND", frontend)
    monkeypatch.setattr(run, "DIST_INDEX", output)
    os.utime(source, (1, 1))
    os.utime(output, (2, 2))
    assert run.frontend_is_stale() is False
    os.utime(source, (3, 3))
    assert run.frontend_is_stale() is True


def test_no_build_rejects_missing_dist(tmp_path, monkeypatch):
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    monkeypatch.setattr(run, "FRONTEND", frontend)
    monkeypatch.setattr(run, "DIST_INDEX", frontend / "dist" / "index.html")
    with pytest.raises(SystemExit, match="前端构建缺失或已过期"):
        run.build_frontend(force=False, no_build=True)


def test_langsmith_cli_all_sets_process_only_overrides(monkeypatch):
    for key in (
        "LANGSMITH_RUNTIME_MODE",
        "LANGSMITH_ENABLED",
        "LANGSMITH_STRICT_STARTUP",
        "LANGSMITH_TRACE_SAMPLE_RATE",
        "LANGSMITH_SESSION_TRACE_LIMIT",
    ):
        monkeypatch.delenv(key, raising=False)
    settings = SimpleNamespace(langsmith_session_trace_limit=200)
    args = SimpleNamespace(
        langsmith_trace_all=True,
        no_langsmith_trace=False,
        langsmith_trace_limit=7,
    )
    run.apply_langsmith_cli(settings, args)
    assert os.environ["LANGSMITH_RUNTIME_MODE"] == "all"
    assert os.environ["LANGSMITH_ENABLED"] == "true"
    assert os.environ["LANGSMITH_STRICT_STARTUP"] == "true"
    assert os.environ["LANGSMITH_TRACE_SAMPLE_RATE"] == "1.0"
    assert os.environ["LANGSMITH_SESSION_TRACE_LIMIT"] == "7"


def test_langsmith_cli_off_overrides_config(monkeypatch):
    monkeypatch.setenv("LANGSMITH_ENABLED", "true")
    settings = SimpleNamespace(langsmith_session_trace_limit=200)
    args = SimpleNamespace(
        langsmith_trace_all=False,
        no_langsmith_trace=True,
        langsmith_trace_limit=None,
    )
    run.apply_langsmith_cli(settings, args)
    assert os.environ["LANGSMITH_RUNTIME_MODE"] == "off"
    assert os.environ["LANGSMITH_ENABLED"] == "false"


def test_langsmith_trace_limit_requires_all_mode():
    settings = SimpleNamespace(langsmith_session_trace_limit=200)
    args = SimpleNamespace(
        langsmith_trace_all=False,
        no_langsmith_trace=False,
        langsmith_trace_limit=10,
    )
    with pytest.raises(SystemExit, match="只能与"):
        run.apply_langsmith_cli(settings, args)
