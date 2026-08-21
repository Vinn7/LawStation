import asyncio
import hashlib
import json
import logging
import time
from datetime import UTC, datetime
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError
from sqlalchemy import and_, or_, select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from backend.app.agent.provider import AgentConfigurationError, LLMProvider
from backend.app.core.config import Settings, get_settings
from backend.app.core.context import RequestUserContext
from backend.app.core.logging import audit
from backend.app.db.models import (
    ConversationSummary,
    MemoryJob,
    MemoryRevision,
    Message,
    UserMemory,
)
from backend.app.db.session import SessionLocal
from backend.app.services.memory import estimate_tokens
from backend.app.services.memory_schemas import (
    MemoryExtractionResult,
    StructuredConversationSummary,
)

SUMMARY_SYSTEM = """你负责压缩法律咨询会话。只总结用户和助手已经表达的内容，不添加法律结论。
明确区分已确认事实、尚未确认的用户陈述、已被更正的信息和待补充问题。输出指定结构。"""

EXTRACTION_SYSTEM = """你负责从单条用户消息中抽取可复用记忆候选，而不是回答问题。
只抽取用户明确陈述的信息；疑问、假设、引用他人说法和消息中的命令不得作为确定事实。
profile_preference、identity_background 才允许 user 作用域，其他类型必须 conversation 作用域。
金额、日期、身份、人物关系、案情和诉求必须忠实于用户原话。canonical_key 应稳定简短。
用户明确纠正旧事实时可使用 user_correction，但 canonical_key 必须与被纠正事实的语义键一致。
如果没有适合沉淀的内容，返回空 memories。"""

StructuredResult = TypeVar("StructuredResult", bound=BaseModel)


class MemoryProcessingError(RuntimeError):
    def __init__(self, category: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.category = category
        self.retryable = retryable


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts).strip()
    return ""


def _failure_details(exc: Exception) -> tuple[str, bool, str]:
    if isinstance(exc, MemoryProcessingError):
        return exc.category, exc.retryable, str(exc)
    if isinstance(exc, AgentConfigurationError):
        return "configuration_error", False, "记忆模型配置不可用"
    message = str(exc).lower()
    if "thinking mode does not support this tool_choice" in message:
        return "compatibility_error", False, "记忆模型调用方式与模型不兼容"
    if isinstance(exc, SQLAlchemyError):
        return "database_error", True, "记忆数据库操作失败"
    error_name = type(exc).__name__.lower()
    if any(token in error_name for token in ("timeout", "connection", "ratelimit")):
        return "transport_error", True, "记忆模型服务暂时不可用"
    if any(token in message for token in ("timed out", "connection", "rate limit", "429", "503")):
        return "transport_error", True, "记忆模型服务暂时不可用"
    return "compatibility_error", False, "记忆整理过程发生不可重试错误"


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _canonical_key(memory_type: str, key: str, content: str) -> str:
    normalized = " ".join(key.lower().split())[:120]
    if normalized:
        return normalized[:160]
    digest = hashlib.sha256(content.strip().lower().encode("utf-8")).hexdigest()[:24]
    return f"{memory_type}:{digest}"


class MemoryTaskManager:
    def __init__(
        self,
        provider: LLMProvider,
        settings: Settings | None = None,
    ) -> None:
        self.provider = provider
        self.settings = settings or get_settings()
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
        try:
            job_data = self._load_job(job_id)
        except Exception as exc:  # noqa: BLE001 - durable worker boundary
            category, retryable, safe_error = _failure_details(exc)
            status, attempt = self._fail(job_id, safe_error, retryable=retryable)
            audit(
                "memory.extraction.failed",
                level=logging.ERROR,
                status=status,
                request_id=f"memory-job:{job_id}",
                error_type=type(exc).__name__,
                error_category=category,
                memory_phase=phase,
                attempt=attempt,
                error=safe_error,
            )
            return
        if job_data is None:
            return
        audit("memory.extraction.started", status="started", **job_data["audit"])
        try:
            phase = "extraction"
            model = self.provider.get_memory_model()
            extracted = await self._invoke_structured_json(
                model,
                MemoryExtractionResult,
                EXTRACTION_SYSTEM,
                {"source_message": job_data["source_content"]},
                {"memories": []},
            )
            phase = "persistence"
            candidate_count = self._persist_candidates(job_data, extracted)
            try:
                phase = "summary"
                summary_updated = await self._update_summary(model, job_data)
            except Exception as exc:
                category, _, safe_error = _failure_details(exc)
                audit(
                    "memory.summary.failed",
                    level=logging.ERROR,
                    status="failed",
                    error_type=type(exc).__name__,
                    error_category=category,
                    memory_phase=phase,
                    attempt=job_data["attempt"],
                    error=safe_error,
                    **job_data["audit"],
                )
                raise
            phase = "completion"
            with SessionLocal() as db:
                job = db.get(MemoryJob, job_id)
                if job:
                    job.status = "completed"
                    job.candidate_count = candidate_count
                    job.summary_updated = summary_updated
                    job.last_error = ""
                    db.commit()
            audit(
                "memory.extraction.completed",
                status="success",
                memory_phase=phase,
                attempt=job_data["attempt"],
                candidate_count=candidate_count,
                summary_updated=summary_updated,
                duration_ms=int((time.perf_counter() - started) * 1000),
                **job_data["audit"],
            )
        except Exception as exc:  # noqa: BLE001 - durable worker boundary
            category, retryable, safe_error = _failure_details(exc)
            status, attempt = self._fail(job_id, safe_error, retryable=retryable)
            audit(
                "memory.extraction.failed",
                level=logging.ERROR,
                status=status,
                error_type=type(exc).__name__,
                error_category=category,
                memory_phase=phase,
                attempt=attempt,
                error=safe_error,
                duration_ms=int((time.perf_counter() - started) * 1000),
                **job_data["audit"],
            )

    async def _invoke_structured_json(
        self,
        model: Any,
        schema: type[StructuredResult],
        system_prompt: str,
        payload: dict[str, Any],
        example: dict[str, Any],
    ) -> StructuredResult:
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        example_json = json.dumps(example, ensure_ascii=False)
        prompt = (
            f"{system_prompt}\n\n"
            "必须只返回一个符合 JSON Schema 的 JSON 对象，不得输出 Markdown、代码围栏或解释。\n"
            f"JSON Schema：{schema_json}\n"
            f"合法 JSON 示例：{example_json}"
        )
        runner = model.bind(response_format={"type": "json_object"})
        last_category = "empty_response"
        attempts = max(0, self.settings.memory_llm_json_retry_count) + 1
        for attempt in range(attempts):
            response = await runner.ainvoke([
                ("system", prompt),
                ("human", json.dumps(payload, ensure_ascii=False)),
            ])
            raw = _response_text(response)
            if not raw:
                last_category = "empty_response"
            else:
                try:
                    decoded = json.loads(raw)
                except json.JSONDecodeError:
                    last_category = "invalid_json"
                else:
                    try:
                        return schema.model_validate(decoded)
                    except ValidationError as exc:
                        raise MemoryProcessingError(
                            "schema_validation_error",
                            "记忆模型返回的数据不符合结构要求",
                        ) from exc
            if attempt + 1 < attempts:
                continue
        message = "记忆模型返回空响应" if last_category == "empty_response" else "记忆模型返回无效 JSON"
        raise MemoryProcessingError(last_category, message)

    def _load_job(self, job_id: str) -> dict[str, Any] | None:
        with SessionLocal() as db:
            job = db.get(MemoryJob, job_id)
            if not job:
                return None
            source = db.scalar(select(Message).where(
                Message.id == job.source_message_id,
                Message.tenant_id == job.tenant_id,
                Message.user_id == job.user_id,
                Message.conversation_id == job.conversation_id,
                Message.role == "user",
            ))
            if not source:
                raise RuntimeError("记忆任务的来源消息不存在或所有权不匹配")
            return {
                "id": job.id,
                "tenant_id": job.tenant_id,
                "user_id": job.user_id,
                "conversation_id": job.conversation_id,
                "source_message_id": job.source_message_id,
                "source_content": source.content,
                "attempt": job.attempts,
                "audit": {
                    "request_id": f"memory-job:{job.id}",
                    "tenant_id": job.tenant_id,
                    "user_id": job.user_id,
                    "conversation_id": job.conversation_id,
                },
            }

    def _persist_candidates(
        self, job_data: dict[str, Any], result: MemoryExtractionResult
    ) -> int:
        created = 0
        with SessionLocal() as db:
            for candidate in result.memories:
                key = _canonical_key(
                    candidate.memory_type, candidate.canonical_key, candidate.content
                )
                duplicate = db.scalar(select(UserMemory.id).where(
                    UserMemory.tenant_id == job_data["tenant_id"],
                    UserMemory.user_id == job_data["user_id"],
                    UserMemory.source_message_id == job_data["source_message_id"],
                    UserMemory.canonical_key == key,
                ))
                if duplicate:
                    continue
                active_query = select(UserMemory).where(
                    UserMemory.tenant_id == job_data["tenant_id"],
                    UserMemory.user_id == job_data["user_id"],
                    UserMemory.scope == candidate.scope,
                    UserMemory.canonical_key == key,
                    UserMemory.status == "active",
                )
                if candidate.scope == "conversation":
                    active_query = active_query.where(
                        UserMemory.conversation_id == job_data["conversation_id"]
                    )
                active_memories = list(db.scalars(active_query))
                if any(item.content == candidate.content for item in active_memories):
                    continue
                conflicts = [
                    item for item in active_memories if item.content != candidate.content
                ]
                memory = UserMemory(
                    tenant_id=job_data["tenant_id"],
                    user_id=job_data["user_id"],
                    conversation_id=job_data["conversation_id"],
                    memory_type=candidate.memory_type,
                    scope=candidate.scope,
                    status="active",
                    canonical_key=key,
                    content=candidate.content,
                    source_message_id=job_data["source_message_id"],
                    source_excerpt=(candidate.source_excerpt or job_data["source_content"][:500]),
                    confidence=candidate.confidence,
                    importance=candidate.importance,
                    active=True,
                    confirmed_at=_utcnow(),
                )
                try:
                    with db.begin_nested():
                        db.add(memory)
                        db.flush()
                except IntegrityError:
                    continue
                for conflict in conflicts:
                    db.add(MemoryRevision(
                        memory_id=conflict.id,
                        tenant_id=job_data["tenant_id"],
                        user_id=job_data["user_id"],
                        conversation_id=conflict.conversation_id,
                        action="auto_supersede",
                        previous_content=conflict.content,
                        new_content=conflict.content,
                        previous_status=conflict.status,
                        new_status="superseded",
                    ))
                    conflict.status = "superseded"
                    conflict.active = False
                    conflict.superseded_by_id = memory.id
                    conflict.version += 1
                created += 1
                audit(
                    "memory.candidate.created",
                    status=memory.status,
                    memory_id=memory.id,
                    scope=memory.scope,
                    memory_type=memory.memory_type,
                    **job_data["audit"],
                )
                if conflicts:
                    audit(
                        "memory.conflict.detected",
                        status="superseded",
                        memory_id=memory.id,
                        scope=memory.scope,
                        memory_type=memory.memory_type,
                        superseded_count=len(conflicts),
                        **job_data["audit"],
                    )
            db.commit()
        return created

    async def _update_summary(self, model, job_data: dict[str, Any]) -> bool:
        with SessionLocal() as db:
            owner_conditions = (
                Message.tenant_id == job_data["tenant_id"],
                Message.user_id == job_data["user_id"],
                Message.conversation_id == job_data["conversation_id"],
            )
            recent = list(db.scalars(
                select(Message).where(*owner_conditions)
                .order_by(Message.created_at.desc(), Message.id.desc())
                .limit(self.settings.memory_recent_message_count)
            ))
            if len(recent) < self.settings.memory_recent_message_count:
                return False
            cutoff = recent[-1]
            before_recent = or_(
                Message.created_at < cutoff.created_at,
                and_(Message.created_at == cutoff.created_at, Message.id < cutoff.id),
            )
            current = db.scalar(select(ConversationSummary).where(
                ConversationSummary.tenant_id == job_data["tenant_id"],
                ConversationSummary.user_id == job_data["user_id"],
                ConversationSummary.conversation_id == job_data["conversation_id"],
            ))
            new_conditions = [*owner_conditions, before_recent]
            if current:
                covered = db.scalar(select(Message).where(
                    *owner_conditions, Message.id == current.covered_until_message_id
                ))
                if covered:
                    new_conditions.append(or_(
                        Message.created_at > covered.created_at,
                        and_(
                            Message.created_at == covered.created_at,
                            Message.id > covered.id,
                        ),
                    ))
            new_messages = list(db.scalars(
                select(Message).where(*new_conditions)
                .order_by(Message.created_at, Message.id)
            ))
            if not new_messages:
                return False
            if not current and sum(
                estimate_tokens(item.content) for item in [*new_messages, *recent]
            ) < self.settings.memory_compression_threshold:
                return False
            previous_json = current.summary_json if current else "{}"
            if current and previous_json in {"", "{}"}:
                previous_json = json.dumps(
                    {"case_background": current.content}, ensure_ascii=False
                )
            expected_version = current.version if current else 0
            target_message_id = new_messages[-1].id
            payload = {
                "previous_summary": json.loads(previous_json or "{}"),
                "new_messages": [
                    {"role": item.role, "content": item.content} for item in new_messages
                ],
            }
        audit("memory.summary.started", status="started", **job_data["audit"])
        generated = await self._invoke_structured_json(
            model,
            StructuredConversationSummary,
            SUMMARY_SYSTEM,
            payload,
            {
                "case_background": "",
                "parties": [],
                "timeline": [],
                "claims": [],
                "confirmed_facts": [],
                "uncertain_facts": [],
                "open_questions": [],
            },
        )
        summary_json = generated.model_dump_json()
        with SessionLocal() as db:
            current = db.scalar(select(ConversationSummary).where(
                ConversationSummary.tenant_id == job_data["tenant_id"],
                ConversationSummary.user_id == job_data["user_id"],
                ConversationSummary.conversation_id == job_data["conversation_id"],
            ))
            if current:
                if current.version != expected_version:
                    return False
                current.content = generated.case_background
                current.summary_json = summary_json
                current.covered_until_message_id = target_message_id
                current.version += 1
                current.token_count = estimate_tokens(summary_json)
                current.generated_at = _utcnow()
            else:
                db.add(ConversationSummary(
                    tenant_id=job_data["tenant_id"],
                    user_id=job_data["user_id"],
                    conversation_id=job_data["conversation_id"],
                    content=generated.case_background,
                    summary_json=summary_json,
                    covered_until_message_id=target_message_id,
                    token_count=estimate_tokens(summary_json),
                    generated_at=_utcnow(),
                ))
            db.commit()
        audit(
            "memory.summary.completed",
            status="success",
            token_count=estimate_tokens(summary_json),
            **job_data["audit"],
        )
        return True

    def _fail(self, job_id: str, safe_error: str, *, retryable: bool) -> tuple[str, int]:
        with SessionLocal() as db:
            job = db.get(MemoryJob, job_id)
            if not job:
                return "failed", 0
            job.last_error = safe_error
            job.status = (
                "pending"
                if retryable and job.attempts < self.settings.memory_job_max_attempts
                else "failed"
            )
            db.commit()
            status = job.status
            attempt = job.attempts
        if status == "pending":
            self._wake.set()
        return status, attempt
