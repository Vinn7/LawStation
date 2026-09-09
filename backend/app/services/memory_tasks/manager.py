"""Memory Tasks 包的编排入口：后台队列生命周期与单任务处理流程。

本模块只负责队列生命周期（start/close/enqueue/_run/_claim_next）和单任务
编排（_process：加载 -> 抽取 -> 持久化 -> 摘要 -> 完成/失败），具体的模型调用、
记忆持久化和摘要生成分别在 invocation.py/persistence.py/summary.py 里按职责
拆分，通过多继承组合进本文件的 MemoryTaskManager。
"""

import asyncio
import logging
import time

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from backend.app.agent.provider import LLMProvider
from backend.app.core.config import Settings, get_settings
from backend.app.core.context import RequestUserContext
from backend.app.core.logging import audit
from backend.app.db.models import MemoryJob
from backend.app.db.session import SessionLocal
from backend.app.observability import LangSmithObservability
from backend.app.services.memory_schemas import MemoryExtractionResult

from .helpers import _failure_details
from .invocation import InvocationMixin
from .persistence import PersistenceMixin
from .prompts import EXTRACTION_SYSTEM
from .summary import SummaryMixin


class MemoryTaskManager(PersistenceMixin, SummaryMixin, InvocationMixin):
    """管理记忆抽取后台任务队列，持有可复用的 Provider/Settings/Observability。

    类实例可以由多个请求共享；`enqueue` 只做去重和入队，实际处理由内部
    worker 任务串行执行。
    """

    def __init__(
        self,
        provider: LLMProvider,
        settings: Settings | None = None,
        observability: LangSmithObservability | None = None,
    ) -> None:
        self.provider = provider
        self.settings = settings or get_settings()
        self.observability = observability or LangSmithObservability(self.settings)
        self._wake = asyncio.Event()
        self._worker: asyncio.Task | None = None
        self._stopping = False

    async def start(self) -> None:
        with SessionLocal() as db:
            db.execute(
                update(MemoryJob)
                .where(MemoryJob.status == "running")
                .values(status="pending", last_error="服务重启后恢复")
            )
            db.commit()
        self._worker = asyncio.create_task(self._run(), name="lawstation-memory-worker")
        self._wake.set()

    async def close(self) -> None:
        self._stopping = True
        self._wake.set()
        if self._worker:
            await self._worker

    def enqueue(self, ctx: RequestUserContext, conversation_id: str, source_message_id: str) -> MemoryJob:
        with SessionLocal() as db:
            existing = db.scalar(select(MemoryJob).where(
                MemoryJob.tenant_id == ctx.tenant_id,
                MemoryJob.user_id == ctx.user_id,
                MemoryJob.source_message_id == source_message_id,
            ))
            if existing:
                return existing
            job = MemoryJob(
                tenant_id=ctx.tenant_id,
                user_id=ctx.user_id,
                conversation_id=conversation_id,
                source_message_id=source_message_id,
            )
            db.add(job)
            try:
                db.commit()
            except IntegrityError:
                db.rollback()
                existing = db.scalar(select(MemoryJob).where(
                    MemoryJob.tenant_id == ctx.tenant_id,
                    MemoryJob.user_id == ctx.user_id,
                    MemoryJob.source_message_id == source_message_id,
                ))
                if existing is None:
                    raise RuntimeError("记忆任务并发入队失败")
                return existing
            db.refresh(job)
        self._wake.set()
        return job

    async def _run(self) -> None:
        while not self._stopping:
            job_id = self._claim_next()
            if job_id:
                await self._process(job_id)
                continue
            self._wake.clear()
            try:
                await asyncio.wait_for(
                    self._wake.wait(), timeout=self.settings.memory_worker_poll_seconds
                )
            except TimeoutError:
                pass

    def _claim_next(self) -> str | None:
        with SessionLocal() as db:
            job = db.scalar(
                select(MemoryJob)
                .where(MemoryJob.status == "pending")
                .order_by(MemoryJob.created_at)
                .limit(1)
            )
            if not job:
                return None
            job.status = "running"
            job.attempts += 1
            db.commit()
            return job.id

    async def _process(self, job_id: str) -> None:
        started = time.perf_counter()
        phase = "load"
        root_trace = None
        try:
            job_data = self._load_job(job_id)
        except Exception as exc:  # noqa: BLE001 - durable worker boundary
            details = _failure_details(exc)
            status, attempt = self._fail(
                job_id, details.safe_error, retryable=details.retryable
            )
            audit(
                "memory.extraction.failed",
                level=logging.ERROR,
                status=status,
                request_id=f"memory-job:{job_id}",
                error_type=type(exc).__name__,
                error_category=details.category,
                memory_phase=phase,
                memory_model=self._memory_model_name,
                attempt=attempt,
                error=details.safe_error,
                **details.audit_fields(),
            )
            return
        if job_data is None:
            return
        root_trace = self.observability.start_memory_job(
            request_id=job_data["audit"]["request_id"],
            tenant_id=job_data["tenant_id"],
            user_id=job_data["user_id"],
            conversation_id=job_data["conversation_id"],
            job_id=job_data["id"],
            source_message_id=job_data["source_message_id"],
            linked_consultation_trace_id=job_data["linked_consultation_trace_id"],
            attempt=job_data["attempt"],
        )
        audit(
            "memory.extraction.started",
            status="started",
            memory_model=self._memory_model_name,
            **job_data["audit"],
        )
        try:
            with root_trace.activate():
                phase = "extraction"
                model = self.provider.get_memory_model()
                extraction_trace_config = (
                    {**root_trace.config, "run_name": "memory.extraction"}
                    if root_trace.enabled else None
                )
                extracted = await self._invoke_structured_json(
                    model,
                    MemoryExtractionResult,
                    EXTRACTION_SYSTEM,
                    {
                        "source_message": job_data["source_content"],
                        "existing_memories": job_data["existing_memories"],
                    },
                    {"memories": []},
                    trace_config=extraction_trace_config,
                )
                phase = "replacement_or_create"
                with root_trace.span(
                    "memory.replacement_or_create",
                    inputs={"candidate_count": len(extracted.memories)},
                ) as replacement_span:
                    persist_stats = self._persist_candidates(job_data, extracted)
                    if replacement_span is not None:
                        replacement_span.end(outputs={
                            "created_count": persist_stats.created_count,
                            "replaced_count": persist_stats.replaced_count,
                            "rejected_count": persist_stats.rejected_count,
                        })
                try:
                    phase = "summary"
                    summary_updated = await self._update_summary(
                        model,
                        job_data,
                        trace_config=root_trace.config if root_trace.enabled else None,
                    )
                except Exception as exc:
                    details = _failure_details(exc)
                    audit(
                        "memory.summary.failed",
                        level=logging.ERROR,
                        status="failed",
                        error_type=type(exc).__name__,
                        error_category=details.category,
                        memory_phase=phase,
                        memory_model=self._memory_model_name,
                        attempt=job_data["attempt"],
                        error=details.safe_error,
                        **details.audit_fields(),
                        **job_data["audit"],
                    )
                    raise
                phase = "persistence"
                with root_trace.span("memory.persist") as persist_span:
                    self._complete_job(
                        job_id, persist_stats.changed_count, summary_updated
                    )
                    if persist_span is not None:
                        persist_span.end(outputs={
                            "candidate_count": persist_stats.changed_count,
                            "summary_updated": summary_updated,
                        })
                phase = "completion"
            audit(
                "memory.extraction.completed",
                status="success",
                memory_phase=phase,
                attempt=job_data["attempt"],
                candidate_count=persist_stats.changed_count,
                created_count=persist_stats.created_count,
                replaced_count=persist_stats.replaced_count,
                rejected_count=persist_stats.rejected_count,
                summary_updated=summary_updated,
                duration_ms=int((time.perf_counter() - started) * 1000),
                **job_data["audit"],
            )
            await root_trace.finish(outputs={
                "status": "completed",
                "candidate_count": persist_stats.changed_count,
                "created_count": persist_stats.created_count,
                "replaced_count": persist_stats.replaced_count,
                "rejected_count": persist_stats.rejected_count,
                "summary_updated": summary_updated,
                "attempt": job_data["attempt"],
            })
        except Exception as exc:  # noqa: BLE001 - durable worker boundary
            details = _failure_details(exc)
            status, attempt = self._fail(
                job_id, details.safe_error, retryable=details.retryable
            )
            audit(
                "memory.extraction.failed",
                level=logging.ERROR,
                status=status,
                error_type=type(exc).__name__,
                error_category=details.category,
                memory_phase=phase,
                memory_model=self._memory_model_name,
                attempt=attempt,
                error=details.safe_error,
                duration_ms=int((time.perf_counter() - started) * 1000),
                **details.audit_fields(),
                **job_data["audit"],
            )
            if root_trace is not None:
                await root_trace.finish(
                    outputs={"status": status, "phase": phase, "attempt": attempt},
                    error=f"{details.category}: {details.safe_error}",
                )
