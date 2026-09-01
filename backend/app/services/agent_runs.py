"""持久化 Agent 任务、后台 Worker、租约及可重放事件日志。

这里是业务任务层：AgentRun 记录排队/运行/终态、所有权、取消和最终消息；
AgentRunEvent 保存按 sequence 排序的 SSE 事件。LangGraph AsyncSqliteSaver 是另一层，
只负责节点 State 和 super-step 恢复，不能替代本模块的任务管理和消息幂等。

正常生命周期：创建 queued Run -> Worker 获取并发配额 -> claim/lease -> 保存用户消息
并读取 MemorySnapshot -> 执行 Graph -> 记录安全事件 -> 幂等保存助手消息 -> 入队记忆
任务 -> 写 message_end。浏览器订阅断开不会终止这条后台链路。
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from backend.app.agent.concurrency import (
    AgentConcurrencyManager,
    ConcurrencyIdentity,
    ConversationBusyError,
)
from backend.app.agent.service import AgentService
from backend.app.core.config import Settings, get_settings
from backend.app.core.context import RequestUserContext
from backend.app.core.logging import audit, summary
from backend.app.db.models import AgentRun, AgentRunEvent, Conversation, Message
from backend.app.db.session import SessionLocal
from backend.app.services.memory import MemoryService

TERMINAL_STATUSES = {"completed", "interrupted", "failed"}
ACTIVE_STATUSES = {"queued", "running"}


def run_payload(run: AgentRun) -> dict:
    return {
        "id": run.id,
        "request_id": run.request_id,
        "conversation_id": run.conversation_id,
        "status": run.status,
        "current_stage": run.current_stage,
        "input_text": run.input_text,
        "attempt": run.attempt,
        "last_event_sequence": run.last_event_seq,
        "user_message_id": run.user_message_id,
        "assistant_message_id": run.assistant_message_id,
        "model_call_count": run.model_call_count,
        "tool_call_count": run.tool_call_count,
        "error_type": run.error_type,
        "error_summary": run.error_summary,
        "created_at": run.created_at,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
    }


class AgentRunConflict(RuntimeError):
    pass


class AgentRunManager:
    """围绕 LangGraph Checkpoint 提供持久任务所有权、恢复和 SSE 重放。"""

    def __init__(
        self,
        runtime,
        concurrency: AgentConcurrencyManager,
        memory_tasks,
        observability,
        settings: Settings | None = None,
    ) -> None:
        self.runtime = runtime
        self.concurrency = concurrency
        self.memory_tasks = memory_tasks
        self.observability = observability
        self.settings = settings or get_settings()
        self.worker_id = f"{socket.gethostname()}:{id(self)}"
        self._runner: asyncio.Task | None = None
        self._active: dict[str, asyncio.Task] = {}
        self._condition = asyncio.Condition()
        self._closing = False

    async def start(self) -> None:
        """恢复遗留任务/预占、清理过期事件，并启动后台轮询 Worker。"""

        # SQLAlchemy 是同步 API，放入线程避免阻塞 FastAPI 的 asyncio 事件循环。
        await asyncio.to_thread(self._recover_stale)
        for thread_id in await asyncio.to_thread(self._prune_expired_events):
            await self.runtime.delete_checkpoint_thread(thread_id)
        for identity in await asyncio.to_thread(self._active_identities):
            try:
                await self.concurrency.reserve(identity)
            except ConversationBusyError:
                audit(
                    "agent.run.reservation.recovered_duplicate",
                    level=logging.WARNING,
                    status="ignored",
                    request_id=identity.request_id,
                    tenant_id=identity.tenant_id,
                    user_id=identity.user_id,
                    conversation_id=identity.conversation_id,
                )
                continue
        # 独立 Task 使 Agent 执行不依赖某个 SSE 请求是否仍连接。
        self._runner = asyncio.create_task(self._loop(), name="agent-run-worker")

    async def close(self) -> None:
        """停止本进程 Worker；运行中任务保留可恢复状态而非伪装成用户取消。"""
        self._closing = True
        if self._runner:
            self._runner.cancel()
            await asyncio.gather(self._runner, return_exceptions=True)
        tasks = list(self._active.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def create(self, ctx: RequestUserContext, conversation_id: str, content: str) -> AgentRun:
        """在所有权校验后创建 queued Run 和最初两条持久事件。"""
        with SessionLocal() as db:
            conversation = db.scalar(select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.tenant_id == ctx.tenant_id,
                Conversation.user_id == ctx.user_id,
            ))
            if conversation is None:
                raise LookupError("会话不存在或无权访问")
            run_id = str(uuid4())
            # 每个 Run 使用独立 LangGraph thread。conversation_id 仍用于业务历史，
            # Checkpoint 不会跨多轮自动继承上一次 Graph 中间状态。
            run = AgentRun(
                id=run_id,
                request_id=ctx.request_id,
                tenant_id=ctx.tenant_id,
                user_id=ctx.user_id,
                conversation_id=conversation_id,
                input_text=content,
                status="queued",
                current_stage="queued",
                langgraph_thread_id=f"agent-run:{run_id}",
            )
            db.add(run)
            try:
                db.flush()
                self._append_event_in_session(
                    db,
                    run,
                    "message_start",
                    {"request_id": ctx.request_id, "run_id": run.id},
                )
                self._append_event_in_session(db, run, "agent_status", {
                    "request_id": ctx.request_id,
                    "user_id": ctx.user_id,
                    "conversation_id": conversation_id,
                    "agent": "coordinator",
                    "status": "queued",
                    "message": "咨询任务正在排队",
                })
                db.commit()
            except IntegrityError as exc:
                # 数据库部分唯一索引是跨协程的第二道防线；即使进程内 reservation
                # 出现竞态，同一会话也只能存在一个 queued/running Run。
                db.rollback()
                raise AgentRunConflict("该会话正在生成回答，请等待完成或先停止生成。") from exc
            db.refresh(run)
            audit("agent.run.created", status="queued", run_id=run.id, **ctx.__dict__, conversation_id=conversation_id)
            return run

    def owned(self, ctx: RequestUserContext, run_id: str) -> AgentRun | None:
        with SessionLocal() as db:
            return db.scalar(select(AgentRun).where(
                AgentRun.id == run_id,
                AgentRun.tenant_id == ctx.tenant_id,
                AgentRun.user_id == ctx.user_id,
            ))

    def active_for_conversation(
        self, ctx: RequestUserContext, conversation_id: str
    ) -> AgentRun | None:
        with SessionLocal() as db:
            return db.scalar(select(AgentRun).where(
                AgentRun.tenant_id == ctx.tenant_id,
                AgentRun.user_id == ctx.user_id,
                AgentRun.conversation_id == conversation_id,
                AgentRun.status.in_(ACTIVE_STATUSES),
            ).order_by(AgentRun.created_at.desc()))

    def events(
        self, ctx: RequestUserContext, run_id: str, after_sequence: int
    ) -> tuple[AgentRun | None, list[AgentRunEvent]]:
        """按所有权和 sequence 返回游标之后的事件，用于 SSE 断线重放。"""
        with SessionLocal() as db:
            run = db.scalar(select(AgentRun).where(
                AgentRun.id == run_id,
                AgentRun.tenant_id == ctx.tenant_id,
                AgentRun.user_id == ctx.user_id,
            ))
            if run is None:
                return None, []
            rows = list(db.scalars(select(AgentRunEvent).where(
                AgentRunEvent.run_id == run_id,
                AgentRunEvent.tenant_id == ctx.tenant_id,
                AgentRunEvent.user_id == ctx.user_id,
                AgentRunEvent.sequence > max(0, after_sequence),
            ).order_by(AgentRunEvent.sequence)))
            return run, rows

    async def scenario_outcome(
        self, ctx: RequestUserContext, run_id: str
    ) -> dict | None:
        run, events = await asyncio.to_thread(self.events, ctx, run_id, 0)
        if run is None:
            return None
        checkpoint = {"checkpoint_available": False}
        try:
            checkpoint = await self.runtime.scenario_outcome(run.langgraph_thread_id)
        except Exception:  # noqa: BLE001 - observer metadata must not affect the run
            checkpoint = {"checkpoint_available": False}
        observed_events = [event.event_type for event in events]
        citation_count = 0
        event_skill_ids: set[str] = set()
        for event in events:
            try:
                payload = json.loads(event.payload_json)
            except json.JSONDecodeError:
                continue
            if event.event_type == "citations" and isinstance(payload, list):
                citation_count = max(citation_count, len(payload))
            if event.event_type == "skill_status" and isinstance(payload, dict):
                skill_id = payload.get("skill_id")
                if skill_id:
                    event_skill_ids.add(str(skill_id))
        selected_skill_ids = checkpoint.get("selected_skill_ids") or sorted(event_skill_ids)
        return {
            "terminal_status": run.status,
            "observed_events": observed_events,
            "retrieval_status": checkpoint.get("retrieval_status", "unknown"),
            "selected_skill_ids": selected_skill_ids,
            "citation_count": max(citation_count, int(checkpoint.get("citation_count") or 0)),
            "model_call_count": run.model_call_count,
            "tool_call_count": run.tool_call_count,
            "last_event_sequence": run.last_event_seq,
            "checkpoint_available": bool(checkpoint.get("checkpoint_available")),
        }

    async def wait_for_events(self, timeout: float) -> None:
        try:
            async with self._condition:
                await asyncio.wait_for(self._condition.wait(), timeout=timeout)
        except TimeoutError:
            return

    async def cancel(self, ctx: RequestUserContext, run_id: str) -> AgentRun | None:
        """持久化取消意图，并只取消本进程中对应 Run 的 asyncio Task。"""
        run = await asyncio.to_thread(self._request_cancel, ctx, run_id)
        task = self._active.get(run_id)
        if task and not task.done():
            task.cancel()
        if self.concurrency is not None and run is not None and run.status in TERMINAL_STATUSES:
            await self.concurrency.release_reservation(ConcurrencyIdentity(
                run.request_id, run.tenant_id, run.user_id, run.conversation_id
            ))
        await self._notify()
        return run

    def _request_cancel(self, ctx: RequestUserContext, run_id: str) -> AgentRun | None:
        with SessionLocal() as db:
            run = db.scalar(select(AgentRun).where(
                AgentRun.id == run_id,
                AgentRun.tenant_id == ctx.tenant_id,
                AgentRun.user_id == ctx.user_id,
            ))
            if run is None or run.status in TERMINAL_STATUSES:
                return run
            run.cancel_requested = True
            if run.status == "queued":
                run.status = "interrupted"
                run.current_stage = "interrupted"
                run.completed_at = datetime.now(UTC)
                self._append_event_in_session(db, run, "agent_status", {
                    "agent": "coordinator", "status": "interrupted", "message": "任务已取消"
                })
                self._append_event_in_session(db, run, "message_end", {
                    "run_id": run.id, "status": "interrupted"
                })
            db.commit()
            db.refresh(run)
            return run

    def _recover_stale(self) -> None:
        """启动时把未超过恢复上限的 running Run 重新排队。"""
        with SessionLocal() as db:
            running = list(db.scalars(select(AgentRun).where(AgentRun.status == "running")))
            for run in running:
                if run.attempt >= self.settings.agent_run_recovery_max_attempts:
                    run.status = "failed"
                    run.current_stage = "failed"
                    run.completed_at = datetime.now(UTC)
                    run.error_type = "RecoveryLimitExceeded"
                    run.error_summary = "任务恢复次数达到上限"
                    self._append_event_in_session(db, run, "error", {
                        "message": "任务恢复次数达到上限，请重新发送问题"
                    })
                else:
                    run.status = "queued"
                    run.current_stage = "queued"
                    run.lease_owner = ""
                    run.lease_expires_at = None
            db.commit()

    def _prune_expired_events(self) -> list[str]:
        cutoff = datetime.now(UTC) - timedelta(
            days=max(1, self.settings.agent_run_event_retention_days)
        )
        with SessionLocal() as db:
            expired_runs = list(db.scalars(select(AgentRun).where(
                AgentRun.status.in_(TERMINAL_STATUSES),
                AgentRun.completed_at.is_not(None),
                AgentRun.completed_at < cutoff,
            )))
            run_ids = [run.id for run in expired_runs]
            if run_ids:
                db.query(AgentRunEvent).filter(AgentRunEvent.run_id.in_(run_ids)).delete(
                    synchronize_session=False
                )
                for run in expired_runs:
                    run.latest_checkpoint_id = ""
                db.commit()
            return [run.langgraph_thread_id for run in expired_runs]

    async def _loop(self) -> None:
        """轮询 queued Run，并为每个候选创建独立执行 Task。"""
        while not self._closing:
            run_ids = await asyncio.to_thread(self._queued_ids)
            for run_id in run_ids:
                if run_id not in self._active:
                    # create_task 只调度执行；真正的用户/全局容量在 _execute.acquire
                    # 获得，多个排队项不会在这里预占模型额度。
                    task = asyncio.create_task(self._execute(run_id), name=f"agent-run:{run_id}")
                    self._active[run_id] = task
                    task.add_done_callback(lambda _task, rid=run_id: self._active.pop(rid, None))
            await asyncio.sleep(max(0.05, self.settings.agent_run_worker_poll_seconds))

    def _queued_ids(self) -> list[str]:
        with SessionLocal() as db:
            return list(db.scalars(select(AgentRun.id).where(
                AgentRun.status == "queued"
            ).order_by(AgentRun.created_at).limit(self.settings.agent_global_concurrency * 4)))

    def _active_identities(self) -> list[ConcurrencyIdentity]:
        with SessionLocal() as db:
            return [
                ConcurrencyIdentity(
                    run.request_id, run.tenant_id, run.user_id, run.conversation_id
                )
                for run in db.scalars(select(AgentRun).where(
                    AgentRun.status.in_(ACTIVE_STATUSES)
                ))
            ]

    async def _execute(self, run_id: str) -> None:
        """执行一个 Run 的完整事务外 Agent 生命周期，并在 finally 释放所有配额。"""
        run = await asyncio.to_thread(self._load_run, run_id)
        if run is None or run.status != "queued":
            return
        identity = ConcurrencyIdentity(
            run.request_id, run.tenant_id, run.user_id, run.conversation_id
        )
        acquired = False
        lease_task: asyncio.Task | None = None
        trace = None
        try:
            # 等到用户和全局运行容量可用后才 claim；排队等待期间不调用模型/MCP。
            await self.concurrency.acquire(identity)
            acquired = True
            run = await asyncio.to_thread(self._claim, run_id)
            if run is None:
                return
            # Lease heartbeat 标明本进程仍负责该 Run。进程崩溃后 heartbeat 停止，
            # 下次启动会把 stale running Run 重新排队并从 Checkpoint 恢复。
            lease_task = asyncio.create_task(
                self._lease_heartbeat(run_id), name=f"agent-run-lease:{run_id}"
            )
            await self._notify()
            ctx = RequestUserContext(run.tenant_id, run.user_id, run.request_id)
            trace = self.observability.start_consultation(
                request_id=run.request_id,
                tenant_id=run.tenant_id,
                user_id=run.user_id,
                conversation_id=run.conversation_id,
                question=run.input_text,
                model_name=self.settings.deepseek_model,
            )
            # 获得运行配额后才持久化用户消息并读取不可变 MemorySnapshot，避免任务
            # 排队很久却使用过早的记忆快照。事务在模型调用前关闭。
            user_message_id, snapshot = await asyncio.to_thread(self._prepare, run, ctx, trace.trace_id)
            agent = AgentService(
                self.runtime,
                ctx,
                run.conversation_id,
                trace_config=trace.config,
                trace_id=trace.trace_id,
                run_id=run.id,
                resume=run.attempt > 1,
            )
            buffered_tokens: list[str] = []
            citations: list[dict] = []
            # AgentService 把 Graph custom/update 输出适配为安全业务事件。正文 token
            # 先缓冲，只有 Finalize 完成后才会出现，内部草稿不会写入 Event 表。
            async for item in agent.run(snapshot.context, snapshot.history, run.input_text):
                if await asyncio.to_thread(self._cancel_requested, run.id):
                    raise asyncio.CancelledError
                if item["event"] == "token":
                    buffered_tokens.append(str(item["data"]))
                elif item["event"] == "citations":
                    citations = list(item["data"] or [])
                else:
                    # 每个事件在独立短事务中递增 sequence；SSE 订阅者可用游标重放。
                    await asyncio.to_thread(self._append_event, run.id, item["event"], item["data"])
                    await self._notify()
            answer = agent.final_answer or "".join(buffered_tokens)
            # Graph 已结束后只读取最新 checkpoint_id，供任务状态/诊断关联；完整
            # Checkpoint State 不复制进业务表。
            checkpoint = await self.runtime.checkpoint_info(run.langgraph_thread_id)
            assistant_id = await asyncio.to_thread(
                self._complete,
                run.id,
                ctx,
                user_message_id,
                answer,
                citations,
                agent.langsmith_trace_id,
                agent.model_call_count,
                agent.tool_call_count,
                checkpoint.get("checkpoint_id", ""),
            )
            memory_payload = {"status": "failed", "message": "记忆整理任务未能入队"}
            try:
                # 记忆整理是回答后的独立后台任务。入队失败只影响 memory_status，
                # 不回滚已完成的法律回答。
                job = await asyncio.to_thread(
                    self.memory_tasks.enqueue, ctx, run.conversation_id, user_message_id
                )
                memory_job_id = job.id
                memory_payload = {"status": "pending", "job_id": memory_job_id}
            except Exception as exc:  # noqa: BLE001 - memory remains non-blocking
                audit("memory.enqueue.failed", level=logging.ERROR, status="failed", run_id=run.id, error_type=type(exc).__name__)
            await asyncio.to_thread(
                self._finish_completed, run.id, assistant_id, trace.trace_id, memory_payload
            )
            skill_ids = [item.get("skill_id") for item in agent.active_skills]
            skill_versions = {
                item.get("skill_id"): item.get("version")
                for item in agent.active_skills
            }
            await trace.finish(
                outputs={
                    "status": "success",
                    "final_answer": answer,
                    "citations": citations,
                    "model_call_count": agent.model_call_count,
                    "tool_call_count": agent.tool_call_count,
                    "skill_ids": skill_ids,
                    "skill_versions": skill_versions,
                },
                metadata={
                    "skill_ids": skill_ids,
                    "skill_versions": skill_versions,
                },
            )
            audit("agent.run.completed", status="completed", run_id=run.id, assistant_message_id=assistant_id, **ctx.__dict__, conversation_id=run.conversation_id)
            await self._notify()
        except asyncio.CancelledError:
            # 用户 cancel 与应用 shutdown 都会取消 Task：前者落 interrupted 终态，
            # 后者保留 running/lease，让下次启动恢复而不是伪造用户中断。
            if not self._closing:
                await asyncio.to_thread(self._interrupt, run_id)
                if trace is not None:
                    await trace.finish(outputs={"status": "interrupted"}, error="interrupted")
            await self._notify()
        except Exception as exc:  # noqa: BLE001 - persist terminal failure
            await asyncio.to_thread(self._fail, run_id, exc)
            if trace is not None:
                await trace.finish(
                    outputs={"status": "failed", "error_type": type(exc).__name__},
                    error=type(exc).__name__,
                )
            await self._notify()
        finally:
            # 所有异常、取消和早退路径都必须释放 lease task、运行配额和会话预占。
            if lease_task is not None:
                lease_task.cancel()
                await asyncio.gather(lease_task, return_exceptions=True)
            if acquired:
                await self.concurrency.release(identity)
            await self.concurrency.release_reservation(identity)

    async def _lease_heartbeat(self, run_id: str) -> None:
        """按租约时长三分之一续期，降低正常长模型调用被误回收的风险。"""
        interval = max(1.0, self.settings.agent_run_lease_seconds / 3)
        while True:
            await asyncio.sleep(interval)
            renewed = await asyncio.to_thread(self._renew_lease, run_id)
            if not renewed:
                return

    def _renew_lease(self, run_id: str) -> bool:
        with SessionLocal() as db:
            run = db.scalar(select(AgentRun).where(
                AgentRun.id == run_id,
                AgentRun.status == "running",
                AgentRun.lease_owner == self.worker_id,
            ))
            if run is None:
                return False
            run.lease_expires_at = datetime.now(UTC) + timedelta(
                seconds=self.settings.agent_run_lease_seconds
            )
            db.commit()
            return True

    def _load_run(self, run_id: str) -> AgentRun | None:
        with SessionLocal() as db:
            return db.get(AgentRun, run_id)

    def _claim(self, run_id: str) -> AgentRun | None:
        with SessionLocal() as db:
            run = db.get(AgentRun, run_id)
            if run is None or run.status != "queued" or run.cancel_requested:
                return None
            run.status = "running"
            run.current_stage = "analyzing"
            run.attempt += 1
            run.started_at = run.started_at or datetime.now(UTC)
            run.lease_owner = self.worker_id
            run.lease_expires_at = datetime.now(UTC) + timedelta(seconds=self.settings.agent_run_lease_seconds)
            self._append_event_in_session(db, run, "agent_status", {
                "agent": "case_analyst", "status": "analyzing", "message": "正在启动案情分析"
            })
            db.commit()
            db.refresh(run)
            return run

    def _prepare(self, run: AgentRun, ctx: RequestUserContext, trace_id: str | None):
        """幂等保存用户消息并在同一短事务中读取本轮 MemorySnapshot。"""
        with SessionLocal() as db:
            current = db.get(AgentRun, run.id)
            if current is None:
                raise RuntimeError("Agent Run不存在")
            if current.user_message_id:
                user_message = db.get(Message, current.user_message_id)
            else:
                user_message = Message(
                    tenant_id=ctx.tenant_id,
                    user_id=ctx.user_id,
                    conversation_id=run.conversation_id,
                    role="user",
                    content=run.input_text,
                    langsmith_trace_id=trace_id,
                )
                db.add(user_message)
                db.flush()
                current.user_message_id = user_message.id
                db.commit()
            snapshot = MemoryService(db, ctx).snapshot(
                run.conversation_id, run.input_text, user_message.id
            )
            return user_message.id, snapshot

    def _complete(
        self,
        run_id: str,
        ctx: RequestUserContext,
        user_message_id: str,
        answer: str,
        citations: list[dict],
        trace_id: str | None,
        model_calls: int,
        tool_calls: int,
        checkpoint_id: str,
    ) -> str:
        """幂等保存最终助手消息、引用事件和 Graph 调用统计。"""
        del user_message_id
        with SessionLocal() as db:
            run = db.get(AgentRun, run_id)
            if run is None:
                raise RuntimeError("Agent Run不存在")
            # 恢复/重试可能再次到达此处；已有终态消息时直接返回原 ID。
            if run.status == "completed" and run.assistant_message_id:
                return run.assistant_message_id
            assistant = db.get(Message, run.assistant_message_id) if run.assistant_message_id else None
            if assistant is None:
                assistant = Message(
                    tenant_id=ctx.tenant_id,
                    user_id=ctx.user_id,
                    conversation_id=run.conversation_id,
                    role="assistant",
                    content=answer,
                    status="complete",
                    langsmith_trace_id=trace_id,
                )
                db.add(assistant)
                db.flush()
                run.assistant_message_id = assistant.id
            has_tokens = db.scalar(select(AgentRunEvent.id).where(
                AgentRunEvent.run_id == run.id,
                AgentRunEvent.event_type == "token",
            ).limit(1))
            # Runtime 只在复核后输出批准正文；此处按 24 字符写可重放 token 事件。
            # 已存在 token 时跳过，避免崩溃重试产生重复 SSE 正文。
            if not has_tokens:
                for start in range(0, len(answer), 24):
                    self._append_event_in_session(db, run, "token", answer[start : start + 24])
                if citations:
                    self._append_event_in_session(db, run, "citations", citations)
            run.model_call_count = model_calls
            run.tool_call_count = tool_calls
            run.latest_checkpoint_id = checkpoint_id
            db.commit()
            return assistant.id

    def _finish_completed(
        self,
        run_id: str,
        assistant_id: str,
        trace_id: str | None,
        memory_payload: dict,
    ) -> None:
        with SessionLocal() as db:
            run = db.get(AgentRun, run_id)
            if run is None or run.status == "completed":
                return
            self._append_event_in_session(db, run, "memory_status", memory_payload)
            run.status = "completed"
            run.current_stage = "completed"
            run.completed_at = datetime.now(UTC)
            run.lease_owner = ""
            run.lease_expires_at = None
            self._append_event_in_session(db, run, "message_end", {
                "run_id": run.id,
                "message_id": assistant_id,
                "status": "completed",
                "trace_available": bool(trace_id),
            })
            db.commit()

    def _interrupt(self, run_id: str) -> None:
        with SessionLocal() as db:
            run = db.get(AgentRun, run_id)
            if run is None:
                return
            run.status = "interrupted"
            run.current_stage = "interrupted"
            run.completed_at = datetime.now(UTC)
            run.lease_owner = ""
            run.lease_expires_at = None
            self._append_event_in_session(db, run, "agent_status", {
                "agent": "coordinator", "status": "interrupted", "message": "任务已停止"
            })
            self._append_event_in_session(db, run, "message_end", {
                "run_id": run.id, "status": "interrupted"
            })
            db.commit()

    def _fail(self, run_id: str, exc: Exception) -> None:
        with SessionLocal() as db:
            run = db.get(AgentRun, run_id)
            if run is None or run.status in TERMINAL_STATUSES:
                return
            run.status = "failed"
            run.current_stage = "failed"
            run.completed_at = datetime.now(UTC)
            run.error_type = type(exc).__name__
            run.error_summary = summary(str(exc))
            run.lease_owner = ""
            run.lease_expires_at = None
            self._append_event_in_session(db, run, "error", {"message": "回答生成失败，请稍后重试"})
            self._append_event_in_session(db, run, "message_end", {
                "run_id": run.id, "status": "failed"
            })
            db.commit()
            audit("agent.run.failed", level=logging.ERROR, status="failed", run_id=run.id, error_type=type(exc).__name__)

    def _cancel_requested(self, run_id: str) -> bool:
        with SessionLocal() as db:
            value = db.scalar(select(AgentRun.cancel_requested).where(AgentRun.id == run_id))
            return bool(value)

    def _append_event(self, run_id: str, event_type: str, payload) -> None:
        with SessionLocal() as db:
            run = db.get(AgentRun, run_id)
            if run is None or run.status in TERMINAL_STATUSES:
                return
            self._append_event_in_session(db, run, event_type, payload)
            if event_type == "agent_status" and isinstance(payload, dict):
                run.current_stage = str(payload.get("status") or run.current_stage)
            db.commit()

    @staticmethod
    def _append_event_in_session(db, run: AgentRun, event_type: str, payload) -> AgentRunEvent:
        """在调用方事务中原子递增 Run 游标并追加一条可重放事件。"""
        run.last_event_seq += 1
        event = AgentRunEvent(
            run_id=run.id,
            tenant_id=run.tenant_id,
            user_id=run.user_id,
            conversation_id=run.conversation_id,
            sequence=run.last_event_seq,
            event_type=event_type,
            payload_json=json.dumps(payload, ensure_ascii=False, default=str),
        )
        db.add(event)
        return event

    async def _notify(self) -> None:
        async with self._condition:
            self._condition.notify_all()
