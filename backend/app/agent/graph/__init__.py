"""Agent Graph 包的对外门面：只做 re-export，不放任何逻辑。"""

from .evidence import _authoritative_evidence, _citation_errors, _no_match_violations
from .orchestrator import LegalConsultationGraph
from .prompts import (
    ANALYST_PROMPT,
    COUNSEL_PROMPT,
    EVIDENCE_SELECTOR_PROMPT,
    RESEARCH_PROMPT,
    REVIEW_PROMPT,
)

__all__ = [
    "ANALYST_PROMPT",
    "COUNSEL_PROMPT",
    "EVIDENCE_SELECTOR_PROMPT",
    "RESEARCH_PROMPT",
    "REVIEW_PROMPT",
    "LegalConsultationGraph",
    "_authoritative_evidence",
    "_citation_errors",
    "_no_match_violations",
]
