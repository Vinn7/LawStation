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
    """Durable task ownership and SSE replay around LangGraph checkpoints."""

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
        self._runner = asyncio.create_task(self._loop(), name="agent-run-worker")

    async def close(self) -> None:
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
        with SessionLocal() as db:
            conversation = db.scalar(select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.tenant_id == ctx.tenant_id,
                Conversation.user_id == ctx.user_id,
            ))
            if conversation is None:
                raise LookupError("会话不存在或无权访问")
            run_id = str(uuid4())
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

    async def wait_for_events(self, timeout: float) -> None:
        try:
            async with self._condition:
                await asyncio.wait_for(self._condition.wait(), timeout=timeout)
        except TimeoutError:
            return

    async def cancel(self, ctx: RequestUserContext, run_id: str) -> AgentRun | None:
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
        while not self._closing:
            run_ids = await asyncio.to_thread(self._queued_ids)
            for run_id in run_ids:
                if run_id not in self._active:
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
            await self.concurrency.acquire(identity)
            acquired = True
            run = await asyncio.to_thread(self._claim, run_id)
            if run is None:
                return
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
            async for item in agent.run(snapshot.context, snapshot.history, run.input_text):
                if await asyncio.to_thread(self._cancel_requested, run.id):
                    raise asyncio.CancelledError
                if item["event"] == "token":
                    buffered_tokens.append(str(item["data"]))
                elif item["event"] == "citations":
                    citations = list(item["data"] or [])
                else:
                    await asyncio.to_thread(self._append_event, run.id, item["event"], item["data"])
                    await self._notify()
            answer = agent.final_answer or "".join(buffered_tokens)
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
            if lease_task is not None:
                lease_task.cancel()
                await asyncio.gather(lease_task, return_exceptions=True)
            if acquired:
                await self.concurrency.release(identity)
            await self.concurrency.release_reservation(identity)

    async def _lease_heartbeat(self, run_id: str) -> None:
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
        del user_message_id
        with SessionLocal() as db:
            run = db.get(AgentRun, run_id)
            if run is None:
                raise RuntimeError("Agent Run不存在")
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
