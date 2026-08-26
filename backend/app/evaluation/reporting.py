"""Timestamped, atomic report archives shared by evaluation entrypoints."""

from __future__ import annotations

import csv
import fcntl
import json
import os
import re
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from backend.app.core.logging import summary

_RUN_ID = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9]{6}-[a-z0-9][a-z0-9-]{0,39}$")
_PROFILE = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
_REPORT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.json$")


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(
        path.suffix + f".{os.getpid()}.{secrets.token_hex(4)}.tmp"
    )
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


class ReportRun:
    def __init__(
        self,
        *,
        root: Path,
        path: Path,
        run_id: str,
        profile: str,
        timezone: ZoneInfo,
        managed: bool,
        started_at: datetime,
    ) -> None:
        self.root = root
        self.path = path
        self.run_id = run_id
        self.profile = profile
        self.timezone = timezone
        self.managed = managed
        self.started_at = started_at
        self._artifacts: list[str] = []
        self._reused: list[dict[str, str]] = []

    @classmethod
    def create(
        cls,
        root: str | Path,
        profile: str,
        timezone_name: str,
        run_id: str = "",
    ) -> ReportRun:
        try:
            timezone = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"未知评测报告时区：{timezone_name}") from exc
        if not _PROFILE.fullmatch(profile):
            raise ValueError(f"非法评测 profile：{profile}")
        started_at = datetime.now(timezone)
        identifier = run_id or f"{started_at:%Y%m%d-%H%M%S-%f}-{profile}"
        if not _RUN_ID.fullmatch(identifier) or not identifier.endswith(f"-{profile}"):
            raise ValueError(f"非法评测 run_id：{identifier}")
        root_path = Path(root).resolve()
        root_path.mkdir(parents=True, exist_ok=True)
        path = root_path / identifier
        path.mkdir(parents=False, exist_ok=False)
        run = cls(
            root=root_path,
            path=path,
            run_id=identifier,
            profile=profile,
            timezone=timezone,
            managed=True,
            started_at=started_at,
        )
        run._write_manifest("running")
        run._write_latest("running", success=False)
        return run

    @classmethod
    def attach(cls, run_dir: str | Path, timezone_name: str) -> ReportRun:
        path = Path(run_dir).resolve()
        manifest_path = path / "run-manifest.json"
        if not path.is_dir() or not manifest_path.is_file():
            raise ValueError(f"评测运行目录无效：{path}")
        manifest = json.loads(manifest_path.read_text("utf-8"))
        run_id = str(manifest.get("run_id", ""))
        if path.name != run_id or not _RUN_ID.fullmatch(run_id):
            raise ValueError(f"评测运行目录与 run_id 不一致：{path}")
        manifest_timezone = str(manifest.get("timezone", ""))
        if manifest_timezone != timezone_name:
            raise ValueError(
                f"评测运行目录时区不一致：{manifest_timezone} != {timezone_name}"
            )
        timezone = ZoneInfo(manifest_timezone)
        started_at = datetime.fromisoformat(str(manifest["started_at"]))
        return cls(
            root=path.parent,
            path=path,
            run_id=run_id,
            profile=str(manifest.get("profile", "unknown")),
            timezone=timezone,
            managed=False,
            started_at=started_at,
        )

    @staticmethod
    def validate_report_name(name: str) -> str:
        if not _REPORT_NAME.fullmatch(name) or Path(name).name != name:
            raise ValueError(f"非法报告文件名：{name}")
        return name

    def report_path(self, name: str) -> Path:
        return self.path / self.validate_report_name(name)

    def write_json(self, name: str, value: dict[str, Any]) -> Path:
        path = self.report_path(name)
        _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, default=str))
        self.record_artifact(path)
        return path

    def write_metrics(self, name: str, value: dict[str, Any]) -> tuple[Path, Path]:
        json_path = self.write_json(name, value)
        csv_path = json_path.with_suffix(".csv")
        rows = []
        for metric, values in sorted(value.get("metrics", {}).items()):
            rows.append({"metric": metric, **values})
        temporary = csv_path.with_suffix(
            csv_path.suffix + f".{os.getpid()}.{secrets.token_hex(4)}.tmp"
        )
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            fields = [
                "metric", "sample_count", "mean", "standard_deviation", "pass_rate", "direction"
            ]
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, csv_path)
        self.record_artifact(csv_path)
        return json_path, csv_path

    def write_csv(
        self,
        name: str,
        fieldnames: list[str],
        rows: list[dict[str, Any]],
    ) -> Path:
        if Path(name).name != name or not name.endswith(".csv"):
            raise ValueError(f"非法 CSV 报告文件名：{name}")
        path = self.path / name
        temporary = path.with_suffix(
            path.suffix + f".{os.getpid()}.{secrets.token_hex(4)}.tmp"
        )
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
        self.record_artifact(path)
        return path

    def write_text(self, name: str, content: str) -> Path:
        if Path(name).name != name:
            raise ValueError(f"非法报告文件名：{name}")
        path = self.path / name
        _atomic_text(path, content)
        self.record_artifact(path)
        return path

    def record_artifact(self, path: str | Path) -> None:
        value = str(Path(path).resolve().relative_to(self.path))
        if value not in self._artifacts:
            self._artifacts.append(value)

    def record_reuse(self, name: str, source_run_id: str, source_path: str) -> None:
        self._reused.append({
            "artifact": name,
            "source_run_id": source_run_id,
            "source_path": source_path,
        })

    def complete(self, extra: dict[str, Any] | None = None) -> None:
        if not self.managed:
            return
        self._write_manifest("completed", extra=extra)
        self._write_latest("completed", success=True)

    def fail(
        self,
        status: str,
        *,
        stage: str,
        error: BaseException,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not self.managed:
            return
        fields = {
            "failed_stage": stage,
            "error_type": type(error).__name__,
            "error_summary": summary(str(error)),
            **(extra or {}),
        }
        self._write_manifest(status, extra=fields)
        self._write_latest(status, success=False)

    def _manifest(self, status: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        completed_at = None if status == "running" else datetime.now(self.timezone).isoformat()
        return {
            "run_id": self.run_id,
            "profile": self.profile,
            "status": status,
            "path": str(self.path),
            "timezone": self.timezone.key,
            "started_at": self.started_at.isoformat(),
            "completed_at": completed_at,
            "completed_artifacts": sorted(self._artifacts),
            "reused_artifacts": self._reused,
            **(extra or {}),
        }

    def _write_manifest(self, status: str, extra: dict[str, Any] | None = None) -> None:
        _atomic_text(
            self.path / "run-manifest.json",
            json.dumps(self._manifest(status, extra), ensure_ascii=False, indent=2, default=str),
        )

    def _write_latest(self, status: str, *, success: bool) -> None:
        value = self._manifest(status)
        index = {
            key: value[key]
            for key in ("run_id", "profile", "status", "path", "started_at", "completed_at")
        }
        lock_path = self.root / ".latest.lock"
        with lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                self._write_latest_if_newer(self.root / "latest.json", index)
                if success:
                    self._write_latest_if_newer(self.root / "latest-success.json", index)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _write_latest_if_newer(self, path: Path, index: dict[str, Any]) -> None:
        try:
            current = json.loads(path.read_text("utf-8"))
            current_started = datetime.fromisoformat(str(current["started_at"]))
        except (OSError, ValueError, KeyError, TypeError):
            current_started = None
        if current_started is None or self.started_at >= current_started:
            _atomic_text(path, json.dumps(index, ensure_ascii=False, indent=2))
