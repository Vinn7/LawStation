"""应用级 LangGraph Runtime：缓存编译图，并适配请求级输入、恢复与事件输出。

Runtime 本身不保存任何用户消息或案件状态；这些内容只存在于本次 State、Invocation
Context 和指定 thread 的 Checkpoint 中。工具目录或 Skill 内容版本变化时才重编译图。
"""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import BaseMessage

from backend.app.agent.graph import LegalConsultationGraph
from backend.app.agent.provider import LLMProvider
from backend.app.agent.registry import MCPToolRegistry
from backend.app.agent.skills import SkillRegistry
from backend.app.agent.state import AgentInvocationContext, LegalConsultationState
from backend.app.core.config import Settings, get_settings
from backend.app.observability import LangSmithObservability


class AgentRuntime:
    """在共享基础设施与彼此隔离的 Graph invocation 之间建立边界。"""

    def __init__(
        self,
        registry: MCPToolRegistry,
        provider: LLMProvider,
        settings: Settings | None = None,
        observability: LangSmithObservability | None = None,
        checkpointer: Any = None,
        skill_registry: SkillRegistry | None = None,
    ) -> None:
        self.registry = registry
        self.provider = provider
        self.settings = settings or get_settings()
        self.skill_registry = skill_registry or SkillRegistry(self.settings)
        self.observability = observability or LangSmithObservability(self.settings)
        self.checkpointer = checkpointer
        self._compile_lock = asyncio.Lock()
        self._graph: LegalConsultationGraph | None = None
        self._graph_key: tuple[int, tuple[str, ...], str] | None = None

    async def ensure_ready(self, context: AgentInvocationContext) -> LegalConsultationGraph:
        """懒加载 MCP 工具，并按工具/Skill 版本 single-flight 编译 Graph。"""

        model = self.provider.get_chat_model()
        # get_tools 是 MCP“工具发现”而不是法律查询。Registry 缓存 BaseTool 包装，
        # 真实 search_laws/get_law_article 仍会在每次 Tool Call 时通过 MCP 执行。
        tools = await self.registry.get_tools(context.audit_fields)
        registry_status = self.registry.status()
        key = (
            registry_status.version,
            tuple(tool.name for tool in tools),
            self.skill_registry.status().catalog_digest,
        )
        if self._graph is not None and self._graph_key == key:
            return self._graph
        # 多个首请求可能同时到达；Lock 保证相同版本只构造一次共享 Graph。
        async with self._compile_lock:
            if self._graph is None or self._graph_key != key:
                self._graph = LegalConsultationGraph(
                    model=model,
                    tools=tools,
                    registry=self.registry,
                    skill_registry=self.skill_registry,
                    settings=self.settings,
                    checkpointer=self.checkpointer,
                )
                self._graph_key = key
        return self._graph

    async def stream(
        self,
        context: AgentInvocationContext,
        messages: list[BaseMessage],
        memory_context: str,
        trace_config: dict[str, Any] | None = None,
        thread_id: str | None = None,
        resume: bool = False,
    ) -> AsyncIterator[dict[str, Any]]:
        """执行或恢复一次 Graph，并把 LangGraph stream 适配为业务 Agent Event。"""

        graph = await self.ensure_ready(context)
        # 新 Run 从空中间结果开始。State 会被节点返回的 dict 增量合并，并在启用
        # Checkpointer 时于 super-step 边界持久化。
        state: LegalConsultationState = {
            "messages": list(messages),
            "memory_context": memory_context,
            "case_analysis": None,
            "evidence_packet": None,
            "counsel_draft": None,
            "review_result": None,
            "retry_count": 0,
            "revision_count": 0,
            "final_answer": "",
            "citations": [],
            "errors": [],
            "current_fact_overrides": [],
            "active_skills": [],
            "skill_outputs": {},
            "model_call_count": 0,
            "tool_call_count": 0,
            "tool_trajectory": [],
        }
        final_state: dict[str, Any] = dict(state)
        config = dict(trace_config or context.trace_config or {})
        configurable = dict(config.get("configurable") or {})
        # 每个 AgentRun 使用独立 thread_id，避免把 Checkpoint 误用为跨轮记忆。
        # checkpoint_ns 用于隔离未来其他 Graph/版本写入的状态。
        configurable.update({
            "thread_id": thread_id or configurable.get("thread_id") or context.identity.request_id,
            "checkpoint_ns": "lawstation-consultation-v1",
        })
        config["configurable"] = configurable
        graph_input: LegalConsultationState | None = state
        if resume and self.checkpointer is not None:
            # aget_state 读取 LangGraph Checkpoint，不读取业务 messages 表。恢复时把
            # 持久化计数同步回 Context，防止重启后模型/工具上限被重置。
            snapshot = await graph.compiled.aget_state(config)
            if snapshot and snapshot.values:
                final_state.update(snapshot.values)
                context.metrics.model_call_count = int(snapshot.values.get("model_call_count", 0))
                context.metrics.tool_call_count = int(snapshot.values.get("tool_call_count", 0))
                context.metrics.tool_trajectory = list(
                    snapshot.values.get("tool_trajectory", [])
                )
                context.active_skills = list(snapshot.values.get("active_skills", []))
                # 对已有 thread 传 None 表示从最近 Checkpoint 继续，而非重新提交初始
                # State；LangGraph 会从下一个未完成 super-step 恢复。
                graph_input = None
        # CompiledStateGraph.astream 执行节点。updates 用于在服务端合并 State，custom
        # 是节点/中间件通过 stream_writer 产生的安全 UI 状态。
        async for part in graph.compiled.astream(
            graph_input,
            context=context,
            config=config,
            stream_mode=["updates", "custom"],
            version="v2",
        ):
            part_type = part.get("type")
            data = part.get("data")
            if part_type == "custom" and isinstance(data, dict) and "event" in data:
                # 身份快照由服务端补入事件，订阅方可据此阻止旧请求污染其他会话。
                event_data = data.get("data")
                if isinstance(event_data, dict):
                    event_data = {
                        **event_data,
                        "request_id": context.identity.request_id,
                        "user_id": context.identity.user_id,
                        "conversation_id": context.identity.conversation_id,
                    }
                yield {"event": data["event"], "data": event_data}
            elif part_type == "updates" and isinstance(data, dict):
                # updates 不直接发给前端；它可能含草稿和内部结构化状态，只用于得到
                # 最终 State 与评测摘要。
                for update in data.values():
                    if isinstance(update, dict):
                        final_state.update(update)
        citations = final_state.get("citations") or []
        # evaluation_output 是安全的运行结果摘要，供测试、LangSmith metadata 和
        # AgentService 读取；它不是下一轮对话记忆。
        context.evaluation_output = {
            "final_answer": str(final_state.get("final_answer") or ""),
            "case_analysis": (
                final_state["case_analysis"].model_dump()
                if final_state.get("case_analysis") else None
            ),
            "evidence_packet": (
                final_state["evidence_packet"].model_dump()
                if final_state.get("evidence_packet") else None
            ),
            "review_result": (
                final_state["review_result"].model_dump()
                if final_state.get("review_result") else None
            ),
            "citations": [
                item.model_dump() if hasattr(item, "model_dump") else item for item in citations
            ],
            "tool_call_count": context.metrics.tool_call_count,
            "model_call_count": context.metrics.model_call_count,
            "review_mode": context.metrics.review_mode,
            "tool_trajectory": list(context.metrics.tool_trajectory),
            "retry_count": int(final_state.get("retry_count") or 0),
            "revision_count": int(final_state.get("revision_count") or 0),
            "errors": [
                item.model_dump() if hasattr(item, "model_dump") else item
                for item in final_state.get("errors", [])
            ],
            "active_skills": list(final_state.get("active_skills", [])),
            "skill_outputs": dict(final_state.get("skill_outputs", {})),
        }
        analysis = context.evaluation_output.get("case_analysis") or {}
        evidence = context.evaluation_output.get("evidence_packet") or {}
        important_outcome = (
            analysis.get("risk_level") == "high"
            or evidence.get("retrieval_status") in {"tool_error", "tool_unavailable"}
        )
        if not context.langsmith_trace_id and important_outcome:
            question = ""
            if messages:
                content = messages[-1].content
                question = content if isinstance(content, str) else str(content)
            context.langsmith_trace_id = await self.observability.force_outcome_trace(
                request_id=context.identity.request_id,
                tenant_id=context.identity.tenant_id,
                user_id=context.identity.user_id,
                conversation_id=context.identity.conversation_id,
                question=question,
                outcome=context.evaluation_output,
            )
        if citations:
            # citations 只在 Finalize 后产生，因此不会暴露未复核候选证据。
            yield {
                "event": "citations",
                "data": [
                    item.model_dump() if hasattr(item, "model_dump") else item
                    for item in citations
                ],
            }
        final_answer = str(final_state.get("final_answer") or "")
        # Counsel/Reviewer 阶段不流出正文。Finalize 后才把已批准回答切成小块模拟
        # token 事件，避免内部草稿、工具参数或 reasoning 泄露。
        for start in range(0, len(final_answer), 24):
            yield {"event": "token", "data": final_answer[start : start + 24]}
            await asyncio.sleep(0)
        yield {"event": "agent_final", "data": final_answer}

    async def close(self) -> None:
        await self.registry.close()

    async def checkpoint_info(self, thread_id: str) -> dict[str, Any]:
        """读取最终 Checkpoint 标识和下一节点；不返回完整案件 State。"""
        if self.checkpointer is None:
            return {}
        graph = self._graph
        if graph is None:
            return {}
        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "lawstation-consultation-v1",
            }
        }
        # aget_state 返回 StateSnapshot；这里仅抽取持久化 bookkeeping 信息。
        snapshot = await graph.compiled.aget_state(config)
        configurable = (snapshot.config or {}).get("configurable", {}) if snapshot else {}
        return {
            "checkpoint_id": str(configurable.get("checkpoint_id") or ""),
            "next": list(snapshot.next) if snapshot else [],
        }

    async def scenario_outcome(self, thread_id: str) -> dict[str, Any]:
        """Return a deliberately small, non-reasoning summary for the test observer."""
        if self.checkpointer is None or self._graph is None:
            return {"checkpoint_available": False}
        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "lawstation-consultation-v1",
            }
        }
        # 场景观察接口只读取允许公开的状态摘要，不能返回 Prompt、memory_context、
        # 法条正文或模型内部推理。
        snapshot = await self._graph.compiled.aget_state(config)
        if snapshot is None or not snapshot.values:
            return {"checkpoint_available": False}
        values = snapshot.values
        evidence = values.get("evidence_packet")
        if hasattr(evidence, "model_dump"):
            evidence = evidence.model_dump()
        active_skills = values.get("active_skills") or []
        skill_ids = sorted({
            str(item.get("skill_id"))
            for item in active_skills
            if isinstance(item, dict) and item.get("skill_id")
        })
        citations = values.get("citations") or []
        return {
            "checkpoint_available": True,
            "retrieval_status": (
                evidence.get("retrieval_status")
                if isinstance(evidence, dict) else "unknown"
            ) or "unknown",
            "selected_skill_ids": skill_ids,
            "citation_count": len(citations),
        }

    async def delete_checkpoint_thread(self, thread_id: str) -> None:
        """清理测试会话等已授权 Run 的整个 LangGraph thread。"""
        if self.checkpointer is not None:
            await self.checkpointer.adelete_thread(thread_id)
