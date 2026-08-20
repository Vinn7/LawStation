import asyncio
import json
import logging
import time
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain_core.messages import ToolMessage

from backend.app.agent.registry import MCPToolRegistry
from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import audit, redact, summary
from backend.app.db.models import RetrievalTrace, ToolCallRecord
from backend.app.db.session import SessionLocal


def result_metadata(result: str) -> dict[str, Any]:
    try:
        value = json.loads(result)
    except (TypeError, json.JSONDecodeError):
        return {"result_chars": len(result)}
    documents: list[dict[str, Any]] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            if "document_id" in item:
                documents.append(
                    {
                        key: item.get(key)
                        for key in ("document_id", "law_name", "article_number")
                    }
                )
            for nested in item.values():
                visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)

    visit(value)
    return {
        "result_count": len(documents),
        "documents": documents[:20],
        "result_chars": len(result),
    }


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


def _persist_tool_audit(
    context: Any,
    name: str,
    args: dict[str, Any],
    result: str,
    status: str,
    duration_ms: int,
) -> None:
    metadata = result_metadata(result)
    with SessionLocal() as db:
        db.add(
            ToolCallRecord(
                tenant_id=context.user.tenant_id,
                user_id=context.user.user_id,
                conversation_id=context.conversation_id,
                tool_name=name,
                arguments_json=json.dumps(
                    {"summary": summary(redact(args), 1000)}, ensure_ascii=False
                ),
                result_summary=json.dumps(metadata, ensure_ascii=False)[:4000],
                status=status,
                duration_ms=duration_ms,
            )
        )
        if name == "search_laws":
            db.add(
                RetrievalTrace(
                    tenant_id=context.user.tenant_id,
                    user_id=context.user.user_id,
                    conversation_id=context.conversation_id,
                    query=summary(args.get("query", ""), 500),
                    results_json=json.dumps(metadata, ensure_ascii=False)[:12000],
                )
            )
        db.commit()


class ToolAuditMiddleware(AgentMiddleware):
    def __init__(
        self,
        registry: MCPToolRegistry,
        settings: Settings | None = None,
    ) -> None:
        self.registry = registry
        self.settings = settings or get_settings()

    async def awrap_tool_call(self, request: ToolCallRequest, handler):
        context = request.runtime.context
        call = request.tool_call
        name = call["name"]
        args = call.get("args", {})
        context.tool_call_count += 1
        fields = {
            "request_id": context.user.request_id,
            "tenant_id": context.user.tenant_id,
            "user_id": context.user.user_id,
            "conversation_id": context.conversation_id,
        }
        request.runtime.stream_writer(
            {"event": "tool_call_start", "data": {"name": name, "status": "started"}}
        )
        audit(
            "tool.call.started",
            status="started",
            tool_name=name,
            arguments_summary=summary(args),
            **fields,
        )
        started = time.perf_counter()
        status = "success"
        business_error = False
        try:
            message = await asyncio.wait_for(
                handler(request), timeout=self.settings.mcp_tool_timeout_seconds
            )
            result = _message_text(message)
            if isinstance(message, ToolMessage) and message.status == "error":
                status = "error"
                business_error = True
        except TimeoutError:
            status = "error"
            result = "工具调用超时，请稍后重试。"
            message = ToolMessage(
                content=result,
                tool_call_id=call["id"],
                name=name,
                status="error",
            )
            audit(
                "tool.call.timeout",
                level=logging.WARNING,
                status="timeout",
                tool_name=name,
                duration_ms=int((time.perf_counter() - started) * 1000),
                **fields,
            )
        except Exception as exc:
            status = "error"
            result = f"工具调用失败：{summary(str(exc))}"
            message = ToolMessage(
                content=result,
                tool_call_id=call["id"],
                name=name,
                status="error",
            )
            self.registry.invalidate(f"{type(exc).__name__}: {exc}", fields)
            audit(
                "tool.call.failed",
                level=logging.ERROR,
                status="failed",
                tool_name=name,
                error_type=type(exc).__name__,
                error=summary(str(exc)),
                duration_ms=int((time.perf_counter() - started) * 1000),
                **fields,
            )
        duration_ms = int((time.perf_counter() - started) * 1000)
        try:
            await asyncio.to_thread(
                _persist_tool_audit, context, name, args, result, status, duration_ms
            )
        except Exception as exc:
            audit(
                "tool.call.audit_failed",
                level=logging.ERROR,
                status="failed",
                tool_name=name,
                error_type=type(exc).__name__,
                error=summary(str(exc)),
                **fields,
            )
        if status == "success":
            audit(
                "tool.call.completed",
                status="success",
                tool_name=name,
                duration_ms=duration_ms,
                **result_metadata(result),
                **fields,
            )
        elif business_error:
            audit(
                "tool.call.failed",
                level=logging.ERROR,
                status="failed",
                tool_name=name,
                error_type="MCPToolError",
                error=summary(result),
                duration_ms=duration_ms,
                **fields,
            )
        request.runtime.stream_writer(
            {"event": "tool_call_result", "data": {"name": name, "status": status}}
        )
        return message
