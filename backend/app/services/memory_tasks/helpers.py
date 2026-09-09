"""Memory Tasks 共用的纯函数：响应解析、失败分类、规范化 key。"""

import hashlib
import re
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from backend.app.agent.provider import AgentConfigurationError

from .errors import MemoryFailureDetails, MemoryProcessingError


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts).strip()
    return ""


def _safe_upstream_value(value: Any) -> str | None:
    """限制上游 code/param 的字符和长度，避免异常对象夹带敏感正文。"""
    if value is None:
        return None
    text = str(value).strip()
    return text[:80] if re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", text) else None


def _failure_details(exc: Exception) -> MemoryFailureDetails:
    if isinstance(exc, MemoryProcessingError):
        return MemoryFailureDetails(exc.category, exc.retryable, str(exc))
    if isinstance(exc, AgentConfigurationError):
        return MemoryFailureDetails(
            "configuration_error", False, "记忆模型配置不可用"
        )

    message = str(exc).lower()
    upstream_status = getattr(exc, "status_code", None)
    upstream_status = upstream_status if isinstance(upstream_status, int) else None
    upstream_error_code = _safe_upstream_value(getattr(exc, "code", None))
    upstream_parameter = _safe_upstream_value(getattr(exc, "param", None))
    for known_parameter in ("max_completion_tokens", "tool_choice", "response_format"):
        if upstream_parameter is None and known_parameter in message:
            upstream_parameter = known_parameter
            break

    if upstream_parameter in {"max_completion_tokens", "response_format"}:
        return MemoryFailureDetails(
            "compatibility_error",
            False,
            "记忆模型请求参数与 DeepSeek Chat Completions 不兼容",
            upstream_status,
            upstream_error_code,
            upstream_parameter,
        )
    if upstream_parameter == "tool_choice" or (
        "thinking mode does not support this tool_choice" in message
    ):
        return MemoryFailureDetails(
            "compatibility_error",
            False,
            "记忆模型调用方式与模型不兼容",
            upstream_status,
            upstream_error_code,
            "tool_choice",
        )
    if isinstance(exc, SQLAlchemyError):
        return MemoryFailureDetails("database_error", True, "记忆数据库操作失败")
    error_name = type(exc).__name__.lower()
    retryable_status = upstream_status in {408, 429} or (
        upstream_status is not None and 500 <= upstream_status <= 599
    )
    if retryable_status or any(
        token in error_name for token in ("timeout", "connection", "ratelimit")
    ) or any(
        token in message
        for token in (
            "timed out",
            "connection",
            "rate limit",
            "http 408",
            "http 429",
            "http 500",
            "http 502",
            "http 503",
            "http 504",
        )
    ):
        return MemoryFailureDetails(
            "transport_error",
            True,
            "记忆模型服务暂时不可用",
            upstream_status,
            upstream_error_code,
            upstream_parameter,
        )
    return MemoryFailureDetails(
        "compatibility_error",
        False,
        "记忆整理过程发生不可重试错误",
        upstream_status,
        upstream_error_code,
        upstream_parameter,
    )


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _canonical_key(memory_type: str, key: str, content: str) -> str:
    normalized = " ".join(key.lower().split())[:120]
    if normalized:
        return normalized[:160]
    digest = hashlib.sha256(content.strip().lower().encode("utf-8")).hexdigest()[:24]
    return f"{memory_type}:{digest}"
