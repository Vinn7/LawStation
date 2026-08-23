from dataclasses import dataclass, field
from typing import Any, TypedDict

from langchain_core.messages import BaseMessage

from backend.app.agent.schemas import (
    AgentError,
    CaseAnalysis,
    Citation,
    CounselDraft,
    EvidencePacket,
    ReviewResult,
)


class LegalConsultationState(TypedDict):
    messages: list[BaseMessage]
    memory_context: str
    case_analysis: CaseAnalysis | None
    evidence_packet: EvidencePacket | None
    counsel_draft: CounselDraft | None
    review_result: ReviewResult | None
    retry_count: int
    revision_count: int
    final_answer: str
    citations: list[Citation]
    errors: list[AgentError]


@dataclass(frozen=True)
class AgentInvocationIdentity:
    request_id: str
    tenant_id: str
    user_id: str
    conversation_id: str


@dataclass
class AgentInvocationMetrics:
    tool_call_count: int = 0
    model_call_count: int = 0
    tool_trajectory: list[dict[str, Any]] = field(default_factory=list)
    review_mode: str = "not_applicable"


@dataclass
class AgentInvocationContext:
    identity: AgentInvocationIdentity
    metrics: AgentInvocationMetrics = field(default_factory=AgentInvocationMetrics)
    langsmith_trace_id: str | None = None
    evaluation_output: dict[str, Any] = field(default_factory=dict)
    persist_tool_audit: bool = True
    evaluation_case_analysis: dict[str, Any] | None = None

    @property
    def audit_fields(self) -> dict[str, Any]:
        return {
            "request_id": self.identity.request_id,
            "tenant_id": self.identity.tenant_id,
            "user_id": self.identity.user_id,
            "conversation_id": self.identity.conversation_id,
        }
