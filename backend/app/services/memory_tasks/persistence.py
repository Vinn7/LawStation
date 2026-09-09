"""Memory Tasks 的任务状态 I/O 与乐观锁记忆持久化。"""

import logging
from typing import Any

from sqlalchemy import and_, or_, select, update
from sqlalchemy.exc import IntegrityError

from backend.app.core.logging import audit
from backend.app.db.models import MemoryJob, MemoryRevision, Message, UserMemory
from backend.app.db.session import SessionLocal
from backend.app.services.memory_schemas import MemoryExtractionResult

from .errors import MemoryPersistStats
from .helpers import _canonical_key, _utcnow


class PersistenceMixin:

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
            existing = list(db.scalars(
                select(UserMemory).where(
                    UserMemory.tenant_id == job.tenant_id,
                    UserMemory.user_id == job.user_id,
                    UserMemory.status == "active",
                    or_(
                        UserMemory.scope == "user",
                        and_(
                            UserMemory.scope == "conversation",
                            UserMemory.conversation_id == job.conversation_id,
                        ),
                    ),
                ).order_by(UserMemory.importance.desc(), UserMemory.updated_at.desc())
            ))
            return {
                "id": job.id,
                "tenant_id": job.tenant_id,
                "user_id": job.user_id,
                "conversation_id": job.conversation_id,
                "source_message_id": job.source_message_id,
                "source_content": source.content,
                "linked_consultation_trace_id": source.langsmith_trace_id or "",
                "existing_memories": [
                    {
                        "memory_id": memory.id,
                        "scope": memory.scope,
                        "memory_type": memory.memory_type,
                        "canonical_key": memory.canonical_key,
                        "content": memory.content,
                    }
                    for memory in existing
                ],
                "attempt": job.attempts,
                "audit": {
                    "request_id": f"memory-job:{job.id}",
                    "tenant_id": job.tenant_id,
                    "user_id": job.user_id,
                    "conversation_id": job.conversation_id,
                },
            }

    @staticmethod
    def _complete_job(
        job_id: str,
        candidate_count: int,
        summary_updated: bool,
    ) -> None:
        with SessionLocal() as db:
            job = db.get(MemoryJob, job_id)
            if job:
                job.status = "completed"
                job.candidate_count = candidate_count
                job.summary_updated = summary_updated
                job.last_error = ""
                db.commit()

    def _persist_candidates(
        self, job_data: dict[str, Any], result: MemoryExtractionResult
    ) -> MemoryPersistStats:
        created_count = 0
        replaced_count = 0
        rejected_count = 0
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
                target_query = select(UserMemory).where(
                    UserMemory.tenant_id == job_data["tenant_id"],
                    UserMemory.user_id == job_data["user_id"],
                    UserMemory.scope == candidate.scope,
                    UserMemory.status == "active",
                )
                if candidate.scope == "conversation":
                    target_query = target_query.where(
                        UserMemory.conversation_id == job_data["conversation_id"]
                    )
                replacement_source = "model" if candidate.replaces_memory_id else "canonical_key"
                if candidate.replaces_memory_id:
                    target_query = target_query.where(
                        UserMemory.id == candidate.replaces_memory_id
                    )
                else:
                    target_query = target_query.where(UserMemory.canonical_key == key)
                target_query = target_query.order_by(
                    UserMemory.updated_at.desc(), UserMemory.id.desc()
                ).limit(1)
                target = db.scalar(target_query)
                if candidate.replaces_memory_id and target is None:
                    rejected_count += 1
                    audit(
                        "memory.replacement.rejected",
                        level=logging.WARNING,
                        status="rejected",
                        reason="invalid_or_unowned_target",
                        scope=candidate.scope,
                        memory_type=candidate.memory_type,
                        replacement_source=replacement_source,
                        **job_data["audit"],
                    )
                    continue
                if target is not None:
                    if target.content == candidate.content:
                        continue
                    audit(
                        "memory.replacement.detected",
                        status="detected",
                        memory_id=target.id,
                        scope=target.scope,
                        memory_type=candidate.memory_type,
                        replacement_source=replacement_source,
                        **job_data["audit"],
                    )
                    replacement_status = self._replace_memory(
                        db, target_query, candidate, job_data
                    )
                    if replacement_status == "updated":
                        replaced_count += 1
                        audit(
                            "memory.replacement.completed",
                            status="success",
                            memory_id=target.id,
                            scope=target.scope,
                            memory_type=candidate.memory_type,
                            replacement_source=replacement_source,
                            **job_data["audit"],
                        )
                    elif replacement_status != "unchanged":
                        rejected_count += 1
                        audit(
                            "memory.replacement.rejected",
                            level=logging.WARNING,
                            status="rejected",
                            reason=replacement_status,
                            scope=candidate.scope,
                            memory_type=candidate.memory_type,
                            replacement_source=replacement_source,
                            **job_data["audit"],
                        )
                    continue
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
                created_count += 1
                audit(
                    "memory.candidate.created",
                    status=memory.status,
                    memory_id=memory.id,
                    scope=memory.scope,
                    memory_type=memory.memory_type,
                    **job_data["audit"],
                )
            db.commit()
        return MemoryPersistStats(created_count, replaced_count, rejected_count)

    def _replace_memory(
        self,
        db,
        target_query,
        candidate,
        job_data: dict[str, Any],
    ) -> str:
        for _attempt in range(2):
            target = db.scalar(target_query.execution_options(populate_existing=True))
            if target is None:
                return "invalid_or_unowned_target"
            if target.content == candidate.content:
                return "unchanged"
            previous_content = target.content
            previous_status = target.status
            expected_version = target.version
            try:
                with db.begin_nested():
                    result = db.execute(
                        update(UserMemory)
                        .where(
                            UserMemory.id == target.id,
                            UserMemory.tenant_id == job_data["tenant_id"],
                            UserMemory.user_id == job_data["user_id"],
                            UserMemory.status == "active",
                            UserMemory.version == expected_version,
                        )
                        .values(
                            conversation_id=job_data["conversation_id"],
                            memory_type=candidate.memory_type,
                            content=candidate.content,
                            source_message_id=job_data["source_message_id"],
                            source_excerpt=(
                                candidate.source_excerpt
                                or job_data["source_content"][:500]
                            ),
                            confidence=candidate.confidence,
                            importance=candidate.importance,
                            confirmed_at=_utcnow(),
                            superseded_by_id=None,
                            version=expected_version + 1,
                        )
                    )
                    if not result.rowcount:
                        continue
                    db.add(MemoryRevision(
                        memory_id=target.id,
                        tenant_id=job_data["tenant_id"],
                        user_id=job_data["user_id"],
                        conversation_id=target.conversation_id,
                        action="auto_replace",
                        previous_content=previous_content,
                        new_content=candidate.content,
                        previous_status=previous_status,
                        new_status="active",
                    ))
                    db.flush()
                    return "updated"
            except IntegrityError:
                db.expire_all()
                return "integrity_conflict"
            finally:
                db.expire_all()
        return "version_conflict"

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
