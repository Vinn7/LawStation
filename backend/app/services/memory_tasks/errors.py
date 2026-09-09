"""Memory Tasks 的异常与结果数据类型。"""

from dataclasses import dataclass
from typing import Any


class MemoryProcessingError(RuntimeError):
    def __init__(self, category: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.category = category
        self.retryable = retryable


@dataclass(frozen=True)
class MemoryPersistStats:
    created_count: int = 0
    replaced_count: int = 0
    rejected_count: int = 0

    @property
    def changed_count(self) -> int:
        return self.created_count + self.replaced_count


@dataclass(frozen=True)
class MemoryFailureDetails:
    """可安全写入审计日志和任务表的记忆失败分类。"""

    category: str
    retryable: bool
    safe_error: str
    upstream_status: int | None = None
    upstream_error_code: str | None = None
    upstream_parameter: str | None = None

    def audit_fields(self) -> dict[str, Any]:
        """只返回诊断所需的上游元数据，不返回请求正文或原始响应。"""
        return {
            "upstream_status": self.upstream_status,
            "upstream_error_code": self.upstream_error_code,
            "upstream_parameter": self.upstream_parameter,
        }
