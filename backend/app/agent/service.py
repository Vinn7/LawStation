import asyncio
import json
import time
import logging
from collections.abc import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from sqlalchemy.orm import Session

from backend.app.core.config import get_settings
from backend.app.core.context import RequestUserContext
from backend.app.core.logging import audit, summary
from backend.app.db.models import RetrievalTrace, ToolCallRecord

SYSTEM_PROMPT = """你是一名谨慎的中国法律咨询助手。你可以自主决定是否调用法律检索工具以及工具参数。
涉及具体法律规则、法条编号、权利义务或法律结论时，应优先使用 search_laws 或 get_law_article 核验；
结果不足时可以修改查询再次调用。禁止虚构法条。工具不可用时要明确说明未能核验。
回答不是正式法律意见。不得向工具传递或猜测用户身份；记忆已由系统按当前用户隔离注入。"""


def result_metadata(result: str) -> dict:
    try:
        value = json.loads(result)
    except Exception:
        return {"result_chars": len(result)}
    documents = []
    def visit(item):
        if isinstance(item, dict):
            if "document_id" in item:
                documents.append({key: item.get(key) for key in ("document_id", "law_name", "article_number")})
            for nested in item.values():
                visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)
    visit(value)
    return {"result_count": len(documents), "documents": documents[:20], "result_chars": len(result)}


class AgentService:
    def __init__(self, db: Session, ctx: RequestUserContext, conversation_id: str):
        self.db, self.ctx, self.conversation_id = db, ctx, conversation_id
        self.settings = get_settings()
        self.tool_call_count = 0

    async def run(self, memory_context: str, history, question: str) -> AsyncIterator[dict]:
        if not self.settings.deepseek_api_key:
            yield {"event": "token", "data": "尚未配置 DEEPSEEK_API_KEY，消息已保存；配置后即可使用 Agent。"}
            return
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain_openai import ChatOpenAI

        client = MultiServerMCPClient(
            {"law": {"url": self.settings.mcp_law_server_url, "transport": "streamable_http"}}
        )
        audit("tool.discovery.started", request_id=self.ctx.request_id, tenant_id=self.ctx.tenant_id, user_id=self.ctx.user_id, conversation_id=self.conversation_id, status="started")
        discovery_started = time.perf_counter()
        try:
            tools = await asyncio.wait_for(client.get_tools(), self.settings.mcp_tool_timeout_seconds)
            audit("tool.discovery.completed", request_id=self.ctx.request_id, tenant_id=self.ctx.tenant_id, user_id=self.ctx.user_id, conversation_id=self.conversation_id, status="success", tool_names=[tool.name for tool in tools], duration_ms=int((time.perf_counter()-discovery_started)*1000))
        except Exception as exc:
            tools = []
            audit("tool.discovery.failed", level=logging.ERROR, request_id=self.ctx.request_id, tenant_id=self.ctx.tenant_id, user_id=self.ctx.user_id, conversation_id=self.conversation_id, status="failed", error_type=type(exc).__name__, error=summary(str(exc)), duration_ms=int((time.perf_counter()-discovery_started)*1000))
        tool_map = {tool.name: tool for tool in tools}
        llm = ChatOpenAI(
            model=self.settings.deepseek_model,
            api_key=self.settings.deepseek_api_key,
            base_url=self.settings.deepseek_base_url,
            streaming=True,
        )
        runnable = llm.bind_tools(tools) if tools else llm
        messages = [SystemMessage(content=SYSTEM_PROMPT + ("\n\n" + memory_context if memory_context else ""))]
        for message in history:
            messages.append(HumanMessage(content=message.content) if message.role == "user" else AIMessage(content=message.content))
        messages.append(HumanMessage(content=question))

        for _ in range(self.settings.agent_max_tool_calls + 1):
            full = None
            emitted = False
            async for chunk in runnable.astream(messages):
                full = chunk if full is None else full + chunk
                if chunk.content and isinstance(chunk.content, str):
                    emitted = True
                    yield {"event": "token", "data": chunk.content}
            if full is None or not full.tool_calls:
                if not emitted and full and full.content:
                    yield {"event": "token", "data": str(full.content)}
                return
            messages.append(full)
            for call in full.tool_calls:
                self.tool_call_count += 1
                name, args = call["name"], call.get("args", {})
                yield {"event": "tool_call_start", "data": {"name": name, "arguments": args}}
                started = time.perf_counter()
                status, result = "success", ""
                audit("tool.call.started", request_id=self.ctx.request_id, tenant_id=self.ctx.tenant_id, user_id=self.ctx.user_id, conversation_id=self.conversation_id, status="started", tool_name=name, arguments_summary=summary(args))
                try:
                    if name not in tool_map:
                        raise ValueError("未知或未授权工具")
                    result = await asyncio.wait_for(
                        tool_map[name].ainvoke(args), self.settings.mcp_tool_timeout_seconds
                    )
                    result = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
                except asyncio.TimeoutError as exc:
                    status, result = "error", f"工具调用超时：{exc}"
                    audit("tool.call.timeout", level=logging.WARNING, request_id=self.ctx.request_id, tenant_id=self.ctx.tenant_id, user_id=self.ctx.user_id, conversation_id=self.conversation_id, status="timeout", tool_name=name, duration_ms=int((time.perf_counter()-started)*1000))
                except Exception as exc:
                    status, result = "error", f"工具调用失败：{exc}"
                    audit("tool.call.failed", level=logging.ERROR, request_id=self.ctx.request_id, tenant_id=self.ctx.tenant_id, user_id=self.ctx.user_id, conversation_id=self.conversation_id, status="failed", tool_name=name, error_type=type(exc).__name__, error=summary(str(exc)), duration_ms=int((time.perf_counter()-started)*1000))
                self.db.add(
                    ToolCallRecord(
                        tenant_id=self.ctx.tenant_id,
                        user_id=self.ctx.user_id,
                        conversation_id=self.conversation_id,
                        tool_name=name,
                        arguments_json=json.dumps(args, ensure_ascii=False),
                        result_summary=result[:4000],
                        status=status,
                        duration_ms=int((time.perf_counter() - started) * 1000),
                    )
                )
                if name == "search_laws":
                    self.db.add(RetrievalTrace(tenant_id=self.ctx.tenant_id, user_id=self.ctx.user_id, conversation_id=self.conversation_id, query=str(args.get("query", "")), results_json=result[:12000]))
                self.db.commit()
                if status == "success":
                    audit("tool.call.completed", request_id=self.ctx.request_id, tenant_id=self.ctx.tenant_id, user_id=self.ctx.user_id, conversation_id=self.conversation_id, status="success", tool_name=name, duration_ms=int((time.perf_counter()-started)*1000), **result_metadata(result))
                messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
                yield {"event": "tool_call_result", "data": {"name": name, "status": status}}
        yield {"event": "token", "data": "工具调用次数已达上限，请缩小问题范围后重试。"}
