import json
import logging
import re
import sys
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from backend.app.core.config import get_settings

AUDIT_LOGGER = "lawstation.audit"
_configured = False


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "event": getattr(record, "event", record.getMessage()),
        }
        payload.update(getattr(record, "audit_fields", {}))
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging() -> None:
    global _configured
    if _configured:
        return
    settings = get_settings()
    logger = logging.getLogger(AUDIT_LOGGER)
    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    logger.propagate = False
    formatter = JsonFormatter()
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    try:
        log_dir = Path(settings.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / "lawstation.log",
            maxBytes=settings.log_max_bytes,
            backupCount=settings.log_backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as exc:
        print(f"无法创建审计日志文件，将仅输出到控制台：{exc}", file=sys.stderr)
    _configured = True


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "***" if any(secret in key.lower() for secret in ("key", "token", "authorization")) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if not isinstance(value, str):
        return value
    value = re.sub(r"(?<!\d)1[3-9]\d{9}(?!\d)", "1**********", value)
    value = re.sub(r"(?<!\d)\d{17}[\dXx](?!\d)", "******************", value)
    value = re.sub(r"(?<!\d)\d{16,19}(?!\d)", "****************", value)
    value = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "***@***", value)
    return value


def summary(value: Any, limit: int | None = None) -> str:
    maximum = limit or get_settings().audit_summary_max_chars
    text = json.dumps(value, ensure_ascii=False, default=str) if not isinstance(value, str) else value
    return re.sub(r"\s+", " ", redact(text)).strip()[:maximum]


def audit(event: str, level: int = logging.INFO, **fields: Any) -> None:
    setup_logging()
    safe_fields = {key: redact(value) for key, value in fields.items() if value is not None}
    logging.getLogger(AUDIT_LOGGER).log(
        level,
        event,
        extra={"event": event, "audit_fields": safe_fields},
    )
