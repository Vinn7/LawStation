"""应用级 MCP 工具发现缓存；不缓存工具执行 Session 或法律查询结果。"""

import asyncio
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest

from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import audit, summary
from backend.app.observability.langsmith import LangSmithObservability


class MCPTraceContextInterceptor:
    """Propagate only an active, signed LangSmith parent context to MCP HTTP."""

    def __init__(self, observability: LangSmithObservability) -> None:
        self.observability = observability

    async def __call__(self, request: MCPToolCallRequest, handler):
        # 仅当当前咨询已经存在父 Trace 时，才把签名后的上下文附到 MCP HTTP。
        # handler 继续执行真实远程调用；Interceptor 不直接访问 RAG handler。
        headers = self.observability.trace_headers()
        if not headers:
            return await handler(request)
        return await handler(request.override(headers={**(request.headers or {}), **headers}))


@dataclass(frozen=True)
class MCPToolRegistryStatus:
    status: str
    version: int
    tool_names: tuple[str, ...]
    loaded_at: float | None
    last_error: str | None


class MCPToolRegistry:
    """缓存 MCP 工具元数据和 LangChain BaseTool Wrapper 的应用级注册表。

    ``MultiServerMCPClient`` 是客户端而非 MCP Server。缓存 Wrapper 只消除每轮
    get_tools 发现开销；每次 Tool Call 仍通过 Streamable HTTP 建立短期 MCP Session。
    """

    def __init__(
        self,
        settings: Settings | None = None,
        client: MultiServerMCPClient | None = None,
        observability: LangSmithObservability | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.observability = observability
        # Adapter 根据远端 name/description/input schema 构造 BaseTool，供
        # LangChain create_agent 使用。handle_tool_errors 让业务错误成为 ToolMessage。
        self.client = client or MultiServerMCPClient(
            {
                "law": {
                    "url": self.settings.mcp_law_server_url,
                    "transport": "streamable_http",
                }
            },
            tool_interceptors=(
                [MCPTraceContextInterceptor(observability)] if observability else []
            ),
            handle_tool_errors=True,
        )
        self._lock = asyncio.Lock()
        self._tools: tuple[BaseTool, ...] = ()
        self._tool_map: dict[str, BaseTool] = {}
        self._status = "uninitialized"
        self._version = 0
        self._loaded_at: float | None = None
        self._last_attempt = 0.0
        self._last_error: str | None = None

    async def get_tools(self, audit_context: dict[str, Any] | None = None) -> list[BaseTool]:
        """返回已发现工具；仅首次或明确失效后才访问 MCP Server。"""
        context = audit_context or {}
        if self._status == "ready":
            audit(
                "tool.discovery.cache_hit",
                level=logging.DEBUG,
                status="ready",
                tool_names=list(self._tool_map),
                **context,
            )
            return list(self._tools)
        if self._tools and self._status == "stale" and not self._retry_due():
            return list(self._tools)
        if self._status == "failed" and not self._retry_due():
            return []
        return await self.refresh("initial" if self._version == 0 else "stale", context)

    async def refresh(
        self,
        reason: str,
        audit_context: dict[str, Any] | None = None,
        *,
        force: bool = False,
    ) -> list[BaseTool]:
        """single-flight 刷新工具目录，失败时优先保留最后一份可用缓存。"""

        context = audit_context or {}
        # Lock 使二十个并发首请求也只执行一次 client.get_tools()。
        async with self._lock:
            if not force:
                if self._status == "ready":
                    return list(self._tools)
                if self._status in {"failed", "stale"} and not self._retry_due():
                    return list(self._tools)

            previous_tools = self._tools
            event = "tool.discovery.started" if self._version == 0 else "tool.discovery.refresh_started"
            self._status = "loading" if not previous_tools else "stale"
            self._last_attempt = time.monotonic()
            started = time.perf_counter()
            audit(event, status="started", reason=reason, **context)
            try:
                # get_tools 只发现工具，不执行 search_laws。wait_for 防止故障 MCP
                # 让首轮咨询无限等待。
                tools = await asyncio.wait_for(
                    self.client.get_tools(),
                    timeout=self.settings.mcp_tool_timeout_seconds,
                )
                if not tools:
                    raise RuntimeError("MCP Server 未发现任何工具")
                tool_map = {tool.name: tool for tool in tools}
                if len(tool_map) != len(tools):
                    raise RuntimeError("MCP Server 返回了重复工具名称")
                self._tools = tuple(tools)
                self._tool_map = tool_map
                self._status = "ready"
                # version 变化会进入 AgentRuntime 的 Graph Cache Key，从而原子重建
                # 绑定新工具 Schema 的 Research Agent。
                self._version += 1
                self._loaded_at = time.time()
                self._last_error = None
                completed_event = (
                    "tool.discovery.completed"
                    if self._version == 1
                    else "tool.discovery.refresh_completed"
                )
                audit(
                    completed_event,
                    status="success",
                    reason=reason,
                    tool_names=list(tool_map),
                    version=self._version,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    **context,
                )
            except Exception as exc:  # noqa: BLE001 - discovery failures drive registry degradation
                # 有旧工具时标记 stale 并继续服务；从未成功过才是 failed/无工具降级。
                self._status = "stale" if previous_tools else "failed"
                self._last_error = summary(str(exc))
                audit(
                    "tool.discovery.failed",
                    level=logging.ERROR,
                    status=self._status,
                    reason=reason,
                    error_type=type(exc).__name__,
                    error=self._last_error,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    **context,
                )
            return list(self._tools)

    def invalidate(self, reason: str, audit_context: dict[str, Any] | None = None) -> None:
        self._status = "stale" if self._tools else "failed"
        self._last_attempt = time.monotonic()
        self._last_error = summary(reason)
        audit(
            "tool.discovery.invalidated",
            level=logging.WARNING,
            status=self._status,
            reason=self._last_error,
            **(audit_context or {}),
        )

    def status(self) -> MCPToolRegistryStatus:
        return MCPToolRegistryStatus(
            status=self._status,
            version=self._version,
            tool_names=tuple(self._tool_map),
            loaded_at=self._loaded_at,
            last_error=self._last_error,
        )

    def status_dict(self) -> dict[str, Any]:
        return asdict(self.status())

    async def close(self) -> None:
        # No persistent MCP session is opened by MultiServerMCPClient.get_tools().
        self._tools = ()
        self._tool_map = {}

    def _retry_due(self) -> bool:
        """冷却期避免 MCP 故障时每个请求都重新冲击工具发现端点。"""
        return (
            time.monotonic() - self._last_attempt
            >= self.settings.mcp_tool_discovery_retry_seconds
        )
