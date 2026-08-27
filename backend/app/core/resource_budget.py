"""Small, process-safe monthly counters for optional observability resources."""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar


class ResourceBudgetExceeded(RuntimeError):
    pass


class MonthlyResourceBudget:
    KEYS: ClassVar[set[str]] = {
        "evaluation_traces",
        "production_traces",
        "agent_model_calls",
        "judge_calls",
    }

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    @staticmethod
    def month() -> str:
        return datetime.now(UTC).strftime("%Y-%m")

    def _empty(self) -> dict:
        return {"month": self.month(), **{key: 0 for key in self.KEYS}}

    def _read(self) -> dict:
        try:
            value = json.loads(self.path.read_text("utf-8"))
        except (OSError, ValueError, TypeError):
            return self._empty()
        if value.get("month") != self.month():
            return self._empty()
        return {"month": self.month(), **{key: int(value.get(key, 0)) for key in self.KEYS}}

    def _write(self, value: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, self.path)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def snapshot(self) -> dict:
        with self._locked():
            return self._read()

    def reserve(self, key: str, amount: int, limit: int) -> None:
        if key not in self.KEYS:
            raise ValueError(f"未知资源预算类型：{key}")
        if amount < 0:
            raise ValueError("资源预留数量不能为负数")
        with self._locked():
            value = self._read()
            after = value[key] + amount
            if after > max(0, limit):
                raise ResourceBudgetExceeded(
                    f"{key} 月度预算不足：已使用 {value[key]}，计划 {amount}，上限 {limit}"
                )
            value[key] = after
            self._write(value)

    def reserve_many(self, amounts: dict[str, int], limits: dict[str, int]) -> None:
        unknown = set(amounts) - self.KEYS
        if unknown or any(amount < 0 for amount in amounts.values()):
            raise ValueError(f"非法资源预留：{sorted(unknown)}")
        with self._locked():
            value = self._read()
            failures = []
            for key, amount in amounts.items():
                limit = max(0, int(limits[key]))
                if value[key] + amount > limit:
                    failures.append(
                        f"{key}: 已使用 {value[key]}，计划 {amount}，上限 {limit}"
                    )
            if failures:
                raise ResourceBudgetExceeded("月度预算不足：" + "；".join(failures))
            for key, amount in amounts.items():
                value[key] += amount
            self._write(value)

    def check_many(self, amounts: dict[str, int], limits: dict[str, int]) -> None:
        """Validate a plan without consuming it; reserve_many repeats this check atomically."""
        unknown = set(amounts) - self.KEYS
        if unknown or any(amount < 0 for amount in amounts.values()):
            raise ValueError(f"非法资源预算检查：{sorted(unknown)}")
        with self._locked():
            value = self._read()
            failures = [
                f"{key}: 已使用 {value[key]}，计划 {amount}，上限 {max(0, int(limits[key]))}"
                for key, amount in amounts.items()
                if value[key] + amount > max(0, int(limits[key]))
            ]
            if failures:
                raise ResourceBudgetExceeded("月度预算不足：" + "；".join(failures))

    def settle(self, key: str, reserved: int, actual: int) -> None:
        """Replace a successful reservation with actual usage; failures remain conservative."""
        if key not in self.KEYS or reserved < 0 or actual < 0:
            raise ValueError("非法资源结算参数")
        with self._locked():
            value = self._read()
            value[key] = max(0, value[key] - reserved + actual)
            self._write(value)

    def record_many(self, amounts: dict[str, int]) -> None:
        """Record actual usage without enforcing a monthly ceiling."""
        unknown = set(amounts) - self.KEYS
        if unknown or any(amount < 0 for amount in amounts.values()):
            raise ValueError(f"非法资源用量：{sorted(unknown)}")
        with self._locked():
            value = self._read()
            for key, amount in amounts.items():
                value[key] += amount
            self._write(value)

    def remaining(self, limits: dict[str, int]) -> dict[str, int]:
        value = self.snapshot()
        return {key: max(0, int(limit) - value.get(key, 0)) for key, limit in limits.items()}

    def peek_remaining(self, limits: dict[str, int]) -> dict[str, int]:
        """Read-only estimate for plan commands; execution uses the locked method."""
        value = self._read()
        return {key: max(0, int(limit) - value.get(key, 0)) for key, limit in limits.items()}
