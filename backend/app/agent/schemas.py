from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ResearchTask(BaseModel):
    issue_id: str
    query: str
    purpose: str = ""

    @field_validator("issue_id", mode="before")
    @classmethod
    def normalize_issue_id(cls, value):
        return str(value)


class CaseAnalysis(BaseModel):
    request_type: Literal["casual_chat", "legal_consultation", "insufficient_information"]
    case_summary: str = ""
    jurisdiction: str = "中国大陆"
    legal_domain: str = "其他"
    key_facts: list[str] = Field(default_factory=list)
    missing_facts: list[str] = Field(default_factory=list)
    legal_issues: list[str] = Field(default_factory=list)
    research_tasks: list[ResearchTask] = Field(default_factory=list)
    risk_level: Literal["low", "medium", "high"] = "medium"
    next_action: Literal["direct_answer", "ask_clarification", "research"] = "research"
    direct_answer: str = ""
    clarification_questions: list[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    document_id: str
    chunk_id: str = ""
    law_name: str = ""
    article_number: str = ""
    content: str = ""
    supports_issue_ids: list[str] = Field(default_factory=list)
    retrieval_sources: list[str] = Field(default_factory=list)
    verification_status: Literal["retrieved", "exact_article_verified"] = "retrieved"
    data_version: str = ""


class UnresolvedIssue(BaseModel):
    issue_id: str
    description: str

    @field_validator("issue_id", mode="before")
    @classmethod
    def normalize_issue_id(cls, value):
        return str(value)


class EvidencePacket(BaseModel):
    retrieval_status: Literal["matched", "no_match", "tool_unavailable", "tool_error"] = "no_match"
    research_tasks: list[ResearchTask] = Field(default_factory=list)
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    unresolved_issues: list[UnresolvedIssue] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    research_summary: str = ""

    @field_validator("unresolved_issues", mode="before")
    @classmethod
    def normalize_unresolved_issues(cls, value):
        normalized = []
        for index, item in enumerate(value or []):
            if isinstance(item, str):
                normalized.append({"issue_id": f"issue-{index + 1}", "description": item})
            elif isinstance(item, dict):
                description = item.get("description") or item.get("issue") or item.get("reason") or "待进一步核验"
                normalized.append({
                    "issue_id": item.get("issue_id", f"issue-{index + 1}"),
                    "description": description,
                })
            else:
                normalized.append({"issue_id": f"issue-{index + 1}", "description": str(item)})
        return normalized

    @model_validator(mode="after")
    def align_status_with_evidence(self):
        if self.evidence_items:
            self.retrieval_status = "matched"
        elif self.retrieval_status == "matched":
            self.retrieval_status = "no_match"
        return self


class CounselClaim(BaseModel):
    claim: str
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    # Backward-compatible input for older prompts/evaluation fixtures. New
    # generations must use evidence_chunk_ids.
    evidence_document_ids: list[str] = Field(default_factory=list)


class CounselDraft(BaseModel):
    answer: str
    claims: list[CounselClaim] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "medium"
    limitations: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)


class ReviewResult(BaseModel):
    approved: bool = False
    unsupported_claims: list[str] = Field(default_factory=list)
    missing_issue_ids: list[str] = Field(default_factory=list)
    citation_errors: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    revision_instruction: str = ""
    next_action: Literal["finalize", "research_again", "revise_draft"] = "finalize"


class Citation(BaseModel):
    document_id: str
    chunk_id: str = ""
    law_name: str = ""
    article_number: str = ""
    quoted_excerpt: str = ""
    data_version: str = ""


class AgentError(BaseModel):
    agent: str
    message: str
