"""LangSmith Trace 的可选导出层；Trace 不是业务状态或恢复事实源。"""

import asyncio
import contextlib
import hashlib
import hmac
import json
import re
import secrets
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from langchain_core.tracers.langchain import LangChainTracer
from langsmith import Client, trace, tracing_context
from langsmith.run_trees import RunTree

from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import audit, redact, summary
from backend.app.core.resource_budget import MonthlyResourceBudget, ResourceBudgetExceeded

_PRIVATE_KEYS = (
    "api_key",
    "authorization",
    "cookie",
    "database_url",
    "reasoning",
    "reasoning_content",
    "secret",
    "token",
)
_SECRET_TEXT = re.compile(r"(?i)(?:sk|bearer)[-_\s][A-Za-z0-9._-]{12,}")
_DATABASE_URL = re.compile(r"(?i)(?:sqlite|postgres(?:ql)?|mysql)://[^\s]+")


def _safe_payload(value: Any, *, capture_content: bool, depth: int = 0) -> Any:
    if not capture_content:
        return {}
    if depth >= 12:
        return "<max-depth>"
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(private in lowered for private in _PRIVATE_KEYS):
                result[key] = "<redacted>"
            else:
                result[key] = _safe_payload(item, capture_content=True, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        return [_safe_payload(item, capture_content=True, depth=depth + 1) for item in value]
    if isinstance(value, str):
        cleaned = _SECRET_TEXT.sub("<redacted-secret>", value)
        cleaned = _DATABASE_URL.sub("<redacted-database-url>", cleaned)
        return redact(cleaned)
    return value


def _client_for_settings(settings: Settings, error_callback=None) -> Client:
    kwargs = {}
    if error_callback is not None:
        kwargs["tracing_error_callback"] = error_callback
    return Client(
        api_url=settings.langsmith_endpoint,
        api_key=settings.langsmith_api_key,
        workspace_id=settings.langsmith_workspace_id or None,
        hide_inputs=lambda value: _safe_payload(
            value, capture_content=settings.langsmith_capture_content
        ),
        hide_outputs=lambda value: _safe_payload(
            value, capture_content=settings.langsmith_capture_content
        ),
        hide_metadata=lambda value: _safe_payload(value, capture_content=True),
        auto_batch_tracing=True,
        **kwargs,
    )


def validate_langsmith_startup(settings: Settings) -> None:
    """Fail-fast validation used only by the explicit all-trace CLI mode."""
    if not settings.langsmith_api_key:
        raise RuntimeError("缺少 LANGSMITH_API_KEY")
    if not settings.langsmith_id_hash_secret:
        raise RuntimeError("缺少 LANGSMITH_ID_HASH_SECRET")
    if not settings.langsmith_workspace_id:
        raise RuntimeError("缺少 LANGSMITH_WORKSPACE_ID")
    client = _client_for_settings(settings)
    try:
        next(client.list_projects(limit=1), None)
    finally:
        client.close()


class SessionTraceBudget:
    def __init__(self, limit: int) -> None:
        if limit <= 0:
            raise ValueError("LangSmith 会话 Trace 上限必须大于 0")
        self.limit = limit
        self._used = 0
        self._lock = threading.Lock()

    def acquire(self) -> bool:
        with self._lock:
            if self._used >= self.limit:
                return False
            self._used += 1
            return True

    def status(self) -> dict[str, int | bool]:
        with self._lock:
            return {
                "session_trace_limit": self.limit,
                "session_trace_used": self._used,
                "session_trace_remaining": max(0, self.limit - self._used),
                "session_trace_exhausted": self._used >= self.limit,
            }


@dataclass(frozen=True)
class TraceInvocation:
    enabled: bool
    trace_id: str | None
    config: dict[str, Any]


@dataclass
class RootTrace:
    """一次咨询或记忆任务的根 Trace 句柄，支持子 Span 与幂等结束。"""
    enabled: bool
    trace_id: str | None
    config: dict[str, Any]
    run_tree: RunTree | None = None
    client: Client | None = None
    _finished: bool = False

    def activate(self):
        """把根 RunTree 放入当前上下文，使 LangChain/LangGraph 子调用自动嵌套。"""
        if not self.enabled or self.run_tree is None:
            return contextlib.nullcontext()
        return tracing_context(parent=self.run_tree, client=self.client, enabled=True)

    def span(
        self,
        name: str,
        *,
        run_type: str = "chain",
        inputs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        if not self.enabled or self.run_tree is None:
            return contextlib.nullcontext()
        return trace(
            name,
            run_type=run_type,
            inputs=inputs or {},
            parent=self.run_tree,
            client=self.client,
            metadata=metadata,
        )

    async def finish(
        self,
        *,
        outputs: dict[str, Any] | None = None,
        error: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled or self.run_tree is None or self._finished:
            return
        self._finished = True
        self.run_tree.end(
            outputs=outputs or {},
            error=error or None,
            metadata=metadata,
        )
        await asyncio.to_thread(self.run_tree.patch)


class LangSmithObservability:
    """应用级、运行期 fail-open 的 LangSmith 集成。

    AgentRun/AgentRunEvent 和 LangGraph Checkpoint 才负责业务恢复；LangSmith 只负责
    调用链观察。导出异常会降级本组件，但不得中断回答、MCP 或记忆任务。
    """

    graph_version = "three-agent-v3-runtime-skills"
    prompt_version = "legal-consultation-v3-progressive-skills"
    app_version = "0.2.0"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client: Client | None = None
        self._status = "disabled"
        self._last_error = ""
        self._budget = MonthlyResourceBudget(self.settings.eval_resource_budget_path)
        self._session_budget = SessionTraceBudget(
            self.settings.langsmith_session_trace_limit
        )
        self._bridge_token = secrets.token_urlsafe(32)
        self._budget_exhausted_logged = False
        if (
            self.settings.langsmith_runtime_mode == "off"
            or not self.settings.langsmith_enabled
        ):
            return
        if not self.settings.langsmith_api_key or not self.settings.langsmith_id_hash_secret:
            self._status = "misconfigured"
            self._last_error = "缺少 LANGSMITH_API_KEY 或 LANGSMITH_ID_HASH_SECRET"
            audit("langsmith.configuration.invalid", status=self._status, error=self._last_error)
            return
        try:
            self._client = _client_for_settings(self.settings, self._on_export_error)
            self._status = "ready"
        except Exception as exc:  # noqa: BLE001 - observability must not block startup
            self._status = "failed"
            self._last_error = type(exc).__name__
            audit(
                "langsmith.initialization.failed",
                status="failed",
                error_type=type(exc).__name__,
                error=summary(str(exc)),
            )

    @property
    def enabled(self) -> bool:
        return self._client is not None and self._status == "ready"

    def status(self) -> dict[str, Any]:
        session = self._session_budget.status()
        return {
            "enabled": self.enabled,
            "export_status": self._status,
            "capture_content": bool(self.settings.langsmith_capture_content),
            "runtime_mode": self.settings.langsmith_runtime_mode,
            "strict_startup": bool(self.settings.langsmith_strict_startup),
            **session,
        }

    @property
    def client(self) -> Client | None:
        return self._client

    @property
    def bridge_token(self) -> str:
        return self._bridge_token

    def _on_export_error(self, exc: Exception) -> None:
        self._status = "degraded"
        self._last_error = type(exc).__name__
        audit(
            "langsmith.export.failed",
            status="degraded",
            error_type=type(exc).__name__,
            error=summary(str(exc)),
        )

    def _stable_hash(self, value: str) -> str:
        return hmac.new(
            self.settings.langsmith_id_hash_secret.encode("utf-8"),
            value.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    @staticmethod
    def _sampled(request_id: str, rate: float) -> bool:
        bounded = min(1.0, max(0.0, rate))
        if bounded <= 0:
            return False
        if bounded >= 1:
            return True
        value = int(hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:16], 16)
        return value / float(0xFFFFFFFFFFFFFFFF) < bounded

    def _project(self, pipeline: str) -> str:
        return f"{self.settings.langsmith_project}-{self.settings.langsmith_environment}-{pipeline}"

    def _production_slot(self) -> bool:
        if self.settings.langsmith_runtime_mode == "all":
            acquired = self._session_budget.acquire()
            if not acquired and not self._budget_exhausted_logged:
                self._budget_exhausted_logged = True
                audit(
                    "langsmith.session_budget.exhausted",
                    status="degraded",
                    resource="session_traces",
                    limit=self.settings.langsmith_session_trace_limit,
                )
            return acquired
        try:
            self._budget.reserve(
                "production_traces",
                1,
                self.settings.langsmith_monthly_production_trace_budget,
            )
            return True
        except ResourceBudgetExceeded as exc:
            if not self._budget_exhausted_logged:
                self._budget_exhausted_logged = True
                audit(
                    "langsmith.budget.exhausted",
                    status="degraded",
                    resource="production_traces",
                    error=summary(str(exc)),
                )
            return False

    def trace_headers(self) -> dict[str, str]:
        """Return signed distributed-trace headers for an active child call."""
        if not self.enabled:
            return {}
        from langsmith.run_helpers import get_current_run_tree

        current = get_current_run_tree()
        if current is None:
            return {}
        return {
            **current.to_headers(),
            "x-lawstation-trace-bridge": self._bridge_token,
        }

    def accepts_trace_headers(self, headers: dict[bytes, bytes]) -> bool:
        supplied = headers.get(b"x-lawstation-trace-bridge", b"").decode(
            "utf-8", errors="ignore"
        )
        return bool(
            self.enabled
            and b"langsmith-trace" in headers
            and hmac.compare_digest(supplied, self._bridge_token)
        )

    def propagated_context(self, headers: dict[bytes, bytes]):
        if not self.accepts_trace_headers(headers) or self._client is None:
            return contextlib.nullcontext()
        return tracing_context(
            parent=headers,
            client=self._client,
            enabled=True,
        )

    def _law_data_version(self) -> str:
        path = Path(self.settings.index_dir) / "law" / "manifest.json"
        try:
            value = json.loads(path.read_text("utf-8"))
            return str(value.get("fingerprint") or value.get("data_version") or "unknown")
        except (OSError, ValueError, TypeError):
            return "unknown"

    def ensure_startup_ready(self) -> None:
        if self.settings.langsmith_strict_startup and not self.enabled:
            raise RuntimeError(self._last_error or "LangSmith 初始化失败")

    def _should_trace(self, request_id: str) -> bool:
        if not self.enabled:
            return False
        if self.settings.langsmith_runtime_mode == "all":
            return self._production_slot()
        return self._sampled(
            request_id, self.settings.langsmith_trace_sample_rate
        ) and self._production_slot()

    def _trace_uuid(self, request_id: str) -> UUID:
        try:
            return UUID(request_id)
        except ValueError:
            return UUID(hashlib.md5(request_id.encode("utf-8")).hexdigest())

    def start_consultation(
        self,
        *,
        request_id: str,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        question: str,
        model_name: str,
    ) -> RootTrace:
        thread_id = self._stable_hash(conversation_id) if self.enabled else conversation_id
        base_config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
        if not self._should_trace(request_id) or self._client is None:
            return RootTrace(False, None, base_config)
        trace_uuid = self._trace_uuid(request_id)
        metadata = {
            "request_id": request_id,
            "tenant_hash": self._stable_hash(tenant_id),
            "user_hash": self._stable_hash(user_id),
            "conversation_hash": self._stable_hash(conversation_id),
            "app_environment": self.settings.langsmith_environment,
            "app_version": self.app_version,
            "graph_version": self.graph_version,
            "prompt_version": self.prompt_version,
            "model_name": model_name,
            "law_data_version": self._law_data_version(),
            "runtime_mode": self.settings.langsmith_runtime_mode,
        }
        root = RunTree(
            id=trace_uuid,
            name="lawstation.consultation",
            run_type="chain",
            inputs={"question": question},
            tags=[
                f"env:{self.settings.langsmith_environment}",
                "pipeline:consultation",
            ],
            extra={"metadata": metadata},
            project_name=self._project("agent"),
            ls_client=self._client,
        )
        try:
            root.post()
        except Exception as exc:  # noqa: BLE001 - running export is fail-open
            self._on_export_error(exc)
            return RootTrace(False, None, base_config)
        tracer = LangChainTracer(
            project_name=self._project("agent"),
            client=self._client,
            tags=[
                f"env:{self.settings.langsmith_environment}",
                "pipeline:consultation",
            ],
            metadata={key: str(value) for key, value in metadata.items()},
        )
        config = {
            **base_config,
            "callbacks": [tracer],
            "run_name": "lawstation-three-agent-graph",
            "tags": [
                f"env:{self.settings.langsmith_environment}",
                "pipeline:consultation",
            ],
            "metadata": metadata,
        }
        return RootTrace(True, str(trace_uuid), config, root, self._client)

    def start_memory_job(
        self,
        *,
        request_id: str,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        job_id: str,
        source_message_id: str,
        linked_consultation_trace_id: str = "",
        attempt: int = 1,
    ) -> RootTrace:
        if not self._should_trace(request_id) or self._client is None:
            return RootTrace(False, None, {})
        metadata = {
            "request_id": request_id,
            "job_id": job_id,
            "tenant_hash": self._stable_hash(tenant_id),
            "user_hash": self._stable_hash(user_id),
            "conversation_hash": self._stable_hash(conversation_id),
            "source_message_hash": self._stable_hash(source_message_id),
            "linked_consultation_trace_id": linked_consultation_trace_id,
            "retry_attempt": attempt,
            "app_environment": self.settings.langsmith_environment,
            "model_name": self.settings.memory_llm_model or self.settings.deepseek_model,
            "runtime_mode": self.settings.langsmith_runtime_mode,
        }
        root = RunTree(
            name="lawstation.memory",
            run_type="chain",
            inputs={"job_id": job_id, "source_message_id": source_message_id},
            tags=[
                f"env:{self.settings.langsmith_environment}",
                "pipeline:memory",
            ],
            extra={"metadata": metadata},
            project_name=self._project("memory"),
            ls_client=self._client,
        )
        try:
            root.post()
        except Exception as exc:  # noqa: BLE001 - running export is fail-open
            self._on_export_error(exc)
            return RootTrace(False, None, {})
        tracer = LangChainTracer(
            project_name=self._project("memory"),
            client=self._client,
            tags=[
                f"env:{self.settings.langsmith_environment}",
                "pipeline:memory",
            ],
            metadata={key: str(value) for key, value in metadata.items()},
        )
        return RootTrace(
            True,
            str(root.id),
            {
                "callbacks": [tracer],
                "run_name": "lawstation.memory",
                "tags": [
                    f"env:{self.settings.langsmith_environment}",
                    "pipeline:memory",
                ],
                "metadata": metadata,
            },
            root,
            self._client,
        )

    def consultation(
        self,
        *,
        request_id: str,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        model_name: str,
        memory_context_chars: int,
    ) -> TraceInvocation:
        thread_id = self._stable_hash(conversation_id) if self.enabled else conversation_id
        base_config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
        if not self.enabled or not self._sampled(
            request_id, self.settings.langsmith_trace_sample_rate
        ):
            return TraceInvocation(False, None, base_config)
        if not self._production_slot():
            return TraceInvocation(False, None, base_config)
        try:
            trace_uuid = UUID(request_id)
        except ValueError:
            trace_uuid = UUID(hashlib.md5(request_id.encode("utf-8")).hexdigest())
        metadata = {
            "request_id": request_id,
            "tenant_hash": self._stable_hash(tenant_id),
            "user_hash": self._stable_hash(user_id),
            "conversation_hash": self._stable_hash(conversation_id),
            "app_environment": self.settings.langsmith_environment,
            "app_version": self.app_version,
            "graph_version": self.graph_version,
            "prompt_version": self.prompt_version,
            "model_name": model_name,
            "law_data_version": self._law_data_version(),
            "memory_context_chars": memory_context_chars,
        }
        tracer = LangChainTracer(
            project_name=self._project("agent"),
            client=self._client,
            tags=[f"env:{self.settings.langsmith_environment}", "pipeline:consultation"],
            metadata={key: str(value) for key, value in metadata.items()},
        )
        config = {
            **base_config,
            "callbacks": [tracer],
            "run_name": "lawstation.consultation",
            "run_id": trace_uuid,
            "tags": [f"env:{self.settings.langsmith_environment}", "pipeline:consultation"],
            "metadata": metadata,
        }
        return TraceInvocation(True, str(trace_uuid), config)

    async def force_outcome_trace(
        self,
        *,
        request_id: str,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        question: str,
        outcome: dict[str, Any],
        error: str = "",
    ) -> str | None:
        """Create a summary trace when an unsampled request becomes important."""
        if (
            not self.enabled
            or self._client is None
            or not self.settings.langsmith_error_trace_enabled
        ):
            return None
        if not self._production_slot():
            return None
        try:
            run_id = UUID(request_id)
        except ValueError:
            run_id = uuid4()
        metadata = {
            "request_id": request_id,
            "tenant_hash": self._stable_hash(tenant_id),
            "user_hash": self._stable_hash(user_id),
            "conversation_hash": self._stable_hash(conversation_id),
            "app_environment": self.settings.langsmith_environment,
            "app_version": self.app_version,
            "graph_version": self.graph_version,
            "prompt_version": self.prompt_version,
            "forced_summary": True,
        }
        try:
            await asyncio.to_thread(
                self._client.create_run,
                "lawstation.consultation.summary",
                {"question": question},
                "chain",
                id=run_id,
                trace_id=run_id,
                project_name=self._project("agent"),
                outputs=outcome,
                error=error or None,
                end_time=datetime.now(UTC),
                tags=[
                    f"env:{self.settings.langsmith_environment}",
                    "pipeline:consultation",
                    "forced:important-outcome",
                ],
                extra={"metadata": metadata},
            )
            return str(run_id)
        except Exception as exc:  # noqa: BLE001 - tracing must remain fail-open
            self._on_export_error(exc)
            return None

    def memory(
        self,
        *,
        request_id: str,
        tenant_id: str,
        user_id: str,
        conversation_id: str,
        job_id: str,
    ) -> TraceInvocation:
        if not self.enabled or not self._sampled(
            request_id, self.settings.langsmith_trace_sample_rate
        ):
            return TraceInvocation(False, None, {})
        if not self._production_slot():
            return TraceInvocation(False, None, {})
        metadata = {
            "request_id": request_id,
            "job_id": job_id,
            "tenant_hash": self._stable_hash(tenant_id),
            "user_hash": self._stable_hash(user_id),
            "conversation_hash": self._stable_hash(conversation_id),
            "app_environment": self.settings.langsmith_environment,
            "model_name": self.settings.memory_llm_model or self.settings.deepseek_model,
        }
        tracer = LangChainTracer(
            project_name=self._project("memory"),
            client=self._client,
            tags=[f"env:{self.settings.langsmith_environment}", "pipeline:memory"],
            metadata={key: str(value) for key, value in metadata.items()},
        )
        return TraceInvocation(
            True,
            None,
            {
                "callbacks": [tracer],
                "run_name": "lawstation.memory",
                "tags": [f"env:{self.settings.langsmith_environment}", "pipeline:memory"],
                "metadata": metadata,
            },
        )

    async def create_user_feedback(
        self,
        *,
        trace_id: str,
        score: int,
        comment: str,
    ) -> bool:
        if not self.enabled or self._client is None or not trace_id:
            return False
        try:
            await asyncio.to_thread(
                self._client.create_feedback,
                trace_id=trace_id,
                key="user_rating",
                score=score,
                comment=comment or None,
                source_info={"source": "lawstation-ui"},
            )
            if score < 0 and self.settings.langsmith_annotation_queue_id:
                await asyncio.to_thread(
                    self._client.add_runs_to_annotation_queue,
                    self.settings.langsmith_annotation_queue_id,
                    run_ids=[trace_id],
                )
            return True
        except Exception as exc:  # noqa: BLE001 - feedback is durably stored locally
            self._on_export_error(exc)
            return False

    async def close(self) -> None:
        if self._client is None:
            return
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._client.flush),
                timeout=self.settings.langsmith_flush_timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001 - shutdown must continue
            self._on_export_error(exc)
        finally:
            try:
                await asyncio.to_thread(self._client.close)
            except Exception as exc:  # noqa: BLE001 - shutdown must continue
                self._on_export_error(exc)
