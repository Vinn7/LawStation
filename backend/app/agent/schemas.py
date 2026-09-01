"""各 Agent 结构化输出协议及证据链数据模型。"""

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

    @field_validator("query", "purpose", mode="before")
    @classmethod
    def normalize_task_text(cls, value):
        return "" if value is None else str(value)


class CurrentFactOverride(BaseModel):
    canonical_key: str = Field(min_length=1, max_length=160)
    new_value: str = Field(min_length=1, max_length=1000)
    old_value: str = Field(default="", max_length=1000)
    replaced_memory_id: str | None = None
    confidence: float = Field(default=0.8, ge=0, le=1)


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
    current_fact_overrides: list[CurrentFactOverride] = Field(default_factory=list)
    requested_skill_ids: list[str] = Field(default_factory=list)

    @field_validator("case_summary", "direct_answer", mode="before")
    @classmethod
    def normalize_optional_text(cls, value):
        return "" if value is None else str(value)

    @field_validator("jurisdiction", mode="before")
    @classmethod
    def normalize_jurisdiction(cls, value):
        return "中国大陆" if value is None else str(value)

    @field_validator("legal_domain", mode="before")
    @classmethod
    def normalize_legal_domain(cls, value):
        return "其他" if value is None else str(value)

    @field_validator(
        "key_facts", "missing_facts", "legal_issues", "research_tasks",
        "clarification_questions", "current_fact_overrides", "requested_skill_ids", mode="before",
    )
    @classmethod
    def normalize_optional_lists(cls, value):
        return [] if value is None else value


class EvidenceItem(BaseModel):
    """经代码映射回真实 MCP 候选后，允许进入 EvidencePacket 的 chunk 证据。"""
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


class CandidateRejection(BaseModel):
    chunk_id: str
    reason: str = "与当前争议点不直接相关"


class EvidenceSelectionResult(BaseModel):
    accepted_chunk_ids: list[str] = Field(default_factory=list)
    rejected_candidates: list[CandidateRejection] = Field(default_factory=list)


class EvidencePacket(BaseModel):
    """Research 向 Counsel/Reviewer 交付的唯一法规证据边界。

    candidate_status 描述召回候选，evidence_status 描述候选是否被采纳，最终
    retrieval_status 再区分 matched、正常 no_match 与工具不可用/异常。
    """
    retrieval_status: Literal["matched", "no_match", "tool_unavailable", "tool_error"] = "no_match"
    candidate_status: Literal["matched", "no_match"] = "no_match"
    evidence_status: Literal["accepted", "rejected", "unavailable", "error"] = "rejected"
    research_tasks: list[ResearchTask] = Field(default_factory=list)
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    accepted_chunk_ids: list[str] = Field(default_factory=list)
    rejected_candidates: list[CandidateRejection] = Field(default_factory=list)
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
            self.candidate_status = "matched"
            self.evidence_status = "accepted"
        elif self.retrieval_status == "matched":
            self.retrieval_status = "no_match"
        return self


class CounselClaim(BaseModel):
    """草稿中的单项论证及其证据 ID；Finalize 据此筛选实际使用的 Citation。"""
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
    skill_outputs: dict[str, dict] = Field(default_factory=dict)


class ReviewResult(BaseModel):
    approved: bool = False
    unsupported_claims: list[str] = Field(default_factory=list)
    missing_issue_ids: list[str] = Field(default_factory=list)
    citation_errors: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    revision_instruction: str = ""
    next_action: Literal["finalize", "research_again", "revise_draft"] = "finalize"


class Citation(BaseModel):
    """最终用户可见引用；元数据必须来自 EvidencePacket 中真实 chunk。"""
    document_id: str
    chunk_id: str = ""
    law_name: str = ""
    article_number: str = ""
    quoted_excerpt: str = ""
    data_version: str = ""


class AgentError(BaseModel):
    agent: str
    message: str
