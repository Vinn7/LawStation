from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar


@dataclass(frozen=True)
class RetrievalGateDecision:
    matched: bool
    confidence: float
    reason: str | None
    features: dict[str, float]
    gate_version: str


class RetrievalConfidenceGate:
    """Deterministic, versioned candidate/no-match decision.

    The gate only decides whether retrieval produced usable candidates. Evidence
    acceptance remains the Research Agent's responsibility.
    """

    DEFAULTS: ClassVar[dict[str, Any]] = {
        "version": "retrieval-gate-v1",
        "threshold": 0.25,
        "weights": {
            "rerank": 0.30,
            "dense": 0.15,
            "bm25": 0.15,
            "rrf": 0.10,
            "source_agreement": 0.15,
            "query_coverage": 0.10,
            "top_margin": 0.05,
        },
    }

    def __init__(self, path: str | Path, *, required: bool = False) -> None:
        self.path = Path(path)
        self.config = dict(self.DEFAULTS)
        if self.path.is_file():
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            self.config.update({key: value for key, value in loaded.items() if key != "weights"})
            self.config["weights"] = {**self.DEFAULTS["weights"], **loaded.get("weights", {})}
        elif required:
            raise FileNotFoundError(f"检索置信度配置不存在：{self.path}")

    @staticmethod
    def _normalized_scores(result: dict[str, Any]) -> dict[str, float]:
        scores = result.get("retrieval_scores") or {}
        bm25 = 1.0 - math.exp(-max(0.0, float(scores.get("bm25", 0.0))) / 6.0)
        dense = max(0.0, min(1.0, float(scores.get("dense", 0.0))))
        rrf = max(0.0, min(1.0, float(scores.get("rrf", 0.0)) / (2.0 / 61.0)))
        rerank = max(0.0, min(1.0, float(scores.get("rerank", 0.0))))
        return {"bm25": bm25, "dense": dense, "rrf": rrf, "rerank": rerank}

    def evaluate(self, query: str, results: list[dict[str, Any]], query_tokens: list[str]) -> RetrievalGateDecision:
        version = str(self.config["version"])
        if not results:
            return RetrievalGateDecision(False, 0.0, "no_candidates_above_retrieval_thresholds", {}, version)
        top = results[0]
        features = self._normalized_scores(top)
        sources = set(top.get("retrieval_sources") or [])
        features["source_agreement"] = float({"bm25", "dense"}.issubset(sources))
        meaningful = {token for token in query_tokens if token.strip() and len(token.strip()) > 1}
        content = str(top.get("content", ""))
        features["query_coverage"] = (
            sum(token in content for token in meaningful) / len(meaningful) if meaningful else 0.0
        )
        top_signal = features["rerank"] or features["rrf"]
        second = self._normalized_scores(results[1]) if len(results) > 1 else {}
        second_signal = second.get("rerank", 0.0) or second.get("rrf", 0.0)
        features["top_margin"] = max(0.0, min(1.0, top_signal - second_signal))
        weights = self.config["weights"]
        confidence = sum(float(weights.get(name, 0.0)) * value for name, value in features.items())
        confidence = max(0.0, min(1.0, confidence))
        threshold = float(self.config["threshold"])
        return RetrievalGateDecision(
            matched=confidence >= threshold,
            confidence=round(confidence, 6),
            reason=None if confidence >= threshold else "candidate_confidence_below_gate",
            features={key: round(value, 6) for key, value in features.items()},
            gate_version=version,
        )
