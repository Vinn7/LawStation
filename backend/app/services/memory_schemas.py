from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from backend.app.db.models import Message

MemoryScope = Literal["user", "conversation"]
MemoryStatus = Literal["pending", "active", "superseded", "rejected", "expired"]
MemoryType = Literal[
    "profile_preference", "identity_background", "case_fact", "timeline_event",
    "party_relationship", "claim_or_goal", "evidence_status", "user_correction",
]


class StructuredConversationSummary(BaseModel):
    case_background: str = ""
    parties: list[str] = Field(default_factory=list)
    timeline: list[str] = Field(default_factory=list)
    claims: list[str] = Field(default_factory=list)
    confirmed_facts: list[str] = Field(default_factory=list)
    uncertain_facts: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class ExtractedMemory(BaseModel):
    memory_type: MemoryType
    scope: MemoryScope
    canonical_key: str = Field(min_length=1, max_length=160)
    content: str = Field(min_length=1, max_length=2000)
    source_excerpt: str = Field(default="", max_length=500)
    confidence: float = Field(ge=0, le=1)
    importance: int = Field(default=50, ge=1, le=100)

    @field_validator("scope")
    @classmethod
    def restrict_user_scope(cls, value: str, info):
        if value == "user" and info.data.get("memory_type") not in {
            "profile_preference", "identity_background"
        }:
            return "conversation"
        return value


class MemoryExtractionResult(BaseModel):
    memories: list[ExtractedMemory] = Field(default_factory=list, max_length=12)


@dataclass(frozen=True)
class MemorySnapshot:
    context: str
    history: list[Message]
    estimated_tokens: int
    selected_case_memories: int
    selected_profile_memories: int
    truncated: bool
