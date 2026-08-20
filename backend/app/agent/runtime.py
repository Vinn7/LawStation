import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, SystemMessage

from backend.app.agent.middleware import ToolAuditMiddleware
from backend.app.agent.provider import LLMProvider
from backend.app.agent.registry import MCPToolRegistry
from backend.app.core.config import Settings, get_settings
from backend.app.core.context import RequestUserContext

SYSTEM_PROMPT = """你是一名谨慎的中国法律咨询助手。你可以自主决定是否调用法律检索工具以及工具参数。
涉及具体法律规则、法条编号、权利义务或法律结论时，应优先使用 search_laws 或 get_law_article 核验；
结果不足时可以修改查询再次调用。禁止虚构法条。工具不可用时要明确说明未能核验。
回答不是正式法律意见。不得向工具传递或猜测用户身份；记忆已由系统按当前用户隔离注入。"""

DEGRADED_PROMPT = """本轮法律检索工具不可用。不得声称已经完成法规核验，不得虚构具体法条编号；
涉及具体法律结论时必须明确提示当前处于检索降级状态，并建议用户稍后重试或咨询专业律师。"""


@dataclass
class AgentInvocationContext:
    user: RequestUserContext
    conversation_id: str
    tool_call_count: int = 0


def _text_content(message: BaseMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    parts = []
    for block in message.content if isinstance(message.content, list) else []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
            parts.append(str(block.get("text", "")))
    return "".join(parts)


class AgentRuntime:
    def __init__(
        self,
        registry: MCPToolRegistry,
        provider: LLMProvider,
        settings: Settings | None = None,
    ) -> None:
        self.registry = registry
        self.provider = provider
        self.settings = settings or get_settings()
        self._compile_lock = asyncio.Lock()
        self._agent: Any = None
        self._agent_key: tuple[int, tuple[str, ...]] | None = None

    async def ensure_ready(self, context: AgentInvocationContext):
        model = self.provider.get_chat_model()
        audit_context = {
            "request_id": context.user.request_id,
            "tenant_id": context.user.tenant_id,
            "user_id": context.user.user_id,
            "conversation_id": context.conversation_id,
        }
        tools = await self.registry.get_tools(audit_context)
        registry_status = self.registry.status()
        key = (registry_status.version, tuple(tool.name for tool in tools))
        if self._agent is not None and self._agent_key == key:
            return self._agent, bool(tools)
        async with self._compile_lock:
            if self._agent is None or self._agent_key != key:
                self._agent = create_agent(
                    model=model,
                    tools=tools,
                    context_schema=AgentInvocationContext,
                    middleware=[
                        ToolCallLimitMiddleware(
                            run_limit=self.settings.agent_max_tool_calls,
                            exit_behavior="continue",
                        ),
                        ModelCallLimitMiddleware(
                            run_limit=self.settings.agent_max_model_calls,
                            exit_behavior="end",
                        ),
                        ToolAuditMiddleware(self.registry, self.settings),
                    ],
                    name="lawstation-legal-agent",
                )
                self._agent_key = key
        return self._agent, bool(tools)

    async def stream(
        self,
        context: AgentInvocationContext,
        messages: list[BaseMessage],
        memory_context: str,
    ) -> AsyncIterator[dict[str, Any]]:
        agent, tools_available = await self.ensure_ready(context)
        prompt = SYSTEM_PROMPT
        if not tools_available:
            prompt += "\n\n" + DEGRADED_PROMPT
        if memory_context:
            prompt += "\n\n" + memory_context
        input_messages = [SystemMessage(content=prompt), *messages]
        final_answer = ""
        emitted_tokens: list[str] = []
        async for part in agent.astream(
            {"messages": input_messages},
            context=context,
            stream_mode=["messages", "updates", "custom"],
            version="v2",
        ):
            part_type = part.get("type")
            data = part.get("data")
            if part_type == "custom" and isinstance(data, dict) and "event" in data:
                yield data
                continue
            if part_type == "messages" and isinstance(data, tuple):
                message, _metadata = data
                if isinstance(message, AIMessageChunk):
                    text = _text_content(message)
                    if text:
                        emitted_tokens.append(text)
                        yield {"event": "token", "data": text}
                continue
            if part_type == "updates" and isinstance(data, dict):
                for update in data.values():
                    update_messages = update.get("messages", []) if isinstance(update, dict) else []
                    if not update_messages:
                        continue
                    message = update_messages[-1]
                    if isinstance(message, AIMessage) and not message.tool_calls:
                        text = _text_content(message)
                        if text:
                            final_answer = text
        if final_answer and not "".join(emitted_tokens).endswith(final_answer):
            yield {"event": "token", "data": final_answer}
        yield {"event": "agent_final", "data": final_answer or "".join(emitted_tokens)}

    async def close(self) -> None:
        await self.registry.close()
