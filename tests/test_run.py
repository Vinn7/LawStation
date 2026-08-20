import os

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
