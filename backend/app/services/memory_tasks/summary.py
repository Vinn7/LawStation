"""Memory Tasks 的增量结构化摘要生成。"""

import json
from typing import Any

from sqlalchemy import and_, or_, select

from backend.app.core.logging import audit
from backend.app.db.models import ConversationSummary, Message
from backend.app.db.session import SessionLocal
from backend.app.services.memory import estimate_tokens
from backend.app.services.memory_schemas import StructuredConversationSummary

from .helpers import _utcnow
from .prompts import SUMMARY_SYSTEM


class SummaryMixin:
    async def _update_summary(
        self,
        model,
        job_data: dict[str, Any],
        *,
        trace_config: dict[str, Any] | None = None,
    ) -> bool:
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
            trace_config=(
                {**trace_config, "run_name": "memory.summary"}
                if trace_config else None
            ),
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
