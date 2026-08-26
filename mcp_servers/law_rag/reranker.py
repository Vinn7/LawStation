"""Local legal passage reranking providers.

TEI is the default provider and serves BGE Cross-Encoder scores through its
native ``/rerank`` endpoint. The previous Ollama Qwen provider remains an
explicit rollback option.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from typing import Protocol

import httpx
from langsmith import trace

from backend.app.core.logging import audit
from backend.app.core.ollama import get_ollama_runtime_info
from backend.app.core.tei import (
    get_tei_reranker_runtime_info,
    resolve_cached_model_revision,
    validate_tei_reranker_info,
)

RERANK_INSTRUCTION_VERSION = "legal-rerank-v1"
TEI_RERANK_VERSION = "bge-cross-encoder-v1"
RERANK_INSTRUCTION = (
    "Given a legal consultation query, determine whether the legal article "
    "directly supports answering the query."
)


class RerankerUnavailable(RuntimeError):
    """Raised when reranking should fail open to the first-stage ranking."""


@dataclass(frozen=True)
class RerankerDescriptor:
    provider: str
    model: str
    digest: str
    instruction_version: str = RERANK_INSTRUCTION_VERSION

    @property
    def ranking_version(self) -> str:
        return f"{self.model}@{self.digest or 'unknown'}"


@dataclass(frozen=True)
class RerankCandidate:
    index: int
    chunk_id: str
    text: str
    rrf_score: float


@dataclass(frozen=True)
class RerankResult:
    index: int
    chunk_id: str
    score: float
    rrf_score: float


@dataclass(frozen=True)
class RerankBatch:
    results: list[RerankResult]
    duration_ms: float
    prompt_tokens: int


class Reranker(Protocol):
    async def prepare(self) -> RerankerDescriptor: ...

    async def rerank(
        self, query: str, candidates: list[RerankCandidate]
    ) -> RerankBatch: ...

    async def close(self) -> None: ...

    def status(self) -> dict: ...


def build_rerank_prompt(query: str, document: str) -> str:
    """Build the raw Qwen3-Reranker prompt without exposing hidden reasoning."""

    system = (
        'Judge whether the Document meets the requirements based on the Query and '
        'the Instruct provided. Note that the answer can only be "yes" or "no".'
    )
    user = (
        f"<Instruct>: {RERANK_INSTRUCTION}\n\n"
        f"<Query>: {query}\n\n"
        f"<Document>: {document}"
    )
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def _normalized_token(value: object) -> str:
    return str(value or "").strip().casefold()


def extract_yes_no_score(payload: dict) -> float:
    """Convert the first generated token distribution to P(yes | yes,no)."""

    logprobs = payload.get("logprobs")
    if not isinstance(logprobs, list) or not logprobs or not isinstance(logprobs[0], dict):
        raise ValueError("Ollama Reranker 未返回首 token 的 logprobs")
    alternatives = list(logprobs[0].get("top_logprobs") or [])
    alternatives.append(logprobs[0])
    scores: dict[str, float] = {}
    for item in alternatives:
        if not isinstance(item, dict):
            continue
        token = _normalized_token(item.get("token"))
        value = item.get("logprob")
        if token in {"yes", "no"} and isinstance(value, (int, float)) and math.isfinite(value):
            scores[token] = max(scores.get(token, -math.inf), float(value))
    if set(scores) != {"yes", "no"}:
        raise ValueError("Ollama Reranker 的候选 token 未同时包含 yes/no")
    maximum = max(scores.values())
    yes = math.exp(scores["yes"] - maximum)
    no = math.exp(scores["no"] - maximum)
    return yes / (yes + no)


def parse_tei_ranks(payload: object, expected_count: int) -> list[dict[str, float | int]]:
    """Validate a complete TEI HTTP rerank response without accepting partial scores."""

    ranks = payload.get("ranks") if isinstance(payload, dict) else payload
    if not isinstance(ranks, list) or len(ranks) != expected_count:
        raise ValueError("TEI Reranker 返回的候选数量不完整")
    parsed: dict[int, float] = {}
    for item in ranks:
        if not isinstance(item, dict):
            raise TypeError("TEI Reranker 返回了无效候选")
        index = item.get("index")
        score = item.get("score")
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("TEI Reranker 返回了无效索引")
        if index < 0 or index >= expected_count or index in parsed:
            raise ValueError("TEI Reranker 返回了重复或越界索引")
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(score)
            or not 0 <= float(score) <= 1
        ):
            raise ValueError("TEI Reranker 返回了无效归一化分数")
        parsed[index] = float(score)
    if set(parsed) != set(range(expected_count)):
        raise ValueError("TEI Reranker 返回的候选索引不完整")
    return [{"index": index, "score": score} for index, score in parsed.items()]


class OllamaReranker:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.client: httpx.AsyncClient | None = None
        self.descriptor: RerankerDescriptor | None = None
        self._semaphore = asyncio.Semaphore(max(1, settings.rag_rerank_concurrency))
        self._status = "checking"
        self._message = "正在检查 Ollama 法条精排模型"
        self._cooldown_until = 0.0
        self._managed = False

    async def _client(self) -> httpx.AsyncClient:
        if self.client is None:
            self.client = httpx.AsyncClient(
                timeout=self.settings.rag_rerank_request_timeout_seconds
            )
        return self.client

    async def prepare(self) -> RerankerDescriptor:
        audit(
            "reranker.check.started",
            status="checking",
            provider="ollama",
            model=self.settings.rag_rerank_model,
        )
        self._status = "checking"
        try:
            digest = ""
            runtime = get_ollama_runtime_info()
            if runtime is not None and runtime.reranker_model == self.settings.rag_rerank_model:
                if not runtime.reranker_available:
                    raise RuntimeError(runtime.reranker_error or "Ollama Reranker 预热失败")
                digest = runtime.reranker_digest
                self._managed = runtime.managed
            else:
                response = await (await self._client()).get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                models = response.json().get("models")
                if not isinstance(models, list):
                    raise TypeError("Ollama /api/tags 未返回模型列表")
                for item in models:
                    if not isinstance(item, dict):
                        continue
                    name = str(item.get("name") or item.get("model") or "")
                    if name == self.settings.rag_rerank_model:
                        digest = str(item.get("digest") or "")
                        break
            if not digest:
                raise RuntimeError(
                    f"未找到 {self.settings.rag_rerank_model}，请先执行："
                    f"ollama pull {self.settings.rag_rerank_model}"
                )
            self.descriptor = RerankerDescriptor(
                provider="ollama",
                model=self.settings.rag_rerank_model,
                digest=digest,
            )
            self._status = "ready"
            self._message = "Ollama 法条精排已就绪"
            audit(
                "reranker.check.completed",
                status="ready",
                provider="ollama",
                model=self.descriptor.model,
                model_digest=self.descriptor.digest,
            )
            return self.descriptor
        except Exception as exc:
            self._status = "degraded"
            self._message = f"精排不可用，当前使用 RRF：{exc}"
            audit(
                "reranker.check.failed",
                level=logging.WARNING,
                status="degraded",
                provider="ollama",
                model=self.settings.rag_rerank_model,
                error_type=type(exc).__name__,
                message=str(exc),
            )
            raise

    def status(self) -> dict:
        if self._cooldown_until > time.monotonic():
            status = "cooldown"
        else:
            status = self._status
        return {
            "status": status,
            "message": self._message,
            "provider": "ollama",
            "model": self.settings.rag_rerank_model,
            "model_digest": self.descriptor.digest if self.descriptor else "",
            "ranking_version": self.descriptor.ranking_version if self.descriptor else "",
            "managed": self._managed,
        }

    async def _score(self, query: str, candidate: RerankCandidate) -> tuple[RerankResult, int]:
        payload = {
            "model": self.settings.rag_rerank_model,
            "prompt": build_rerank_prompt(query, candidate.text),
            "raw": True,
            "stream": False,
            "think": False,
            "logprobs": True,
            "top_logprobs": self.settings.rag_rerank_top_logprobs,
            "keep_alive": self.settings.rag_rerank_keep_alive,
            "options": {
                "temperature": 0,
                "seed": 42,
                "num_predict": 1,
                "num_ctx": 2048,
            },
        }
        async with self._semaphore:
            response = await (await self._client()).post(
                f"{self.base_url}/api/generate", json=payload
            )
            response.raise_for_status()
            body = response.json()
        if not isinstance(body, dict):
            raise TypeError("Ollama /api/generate 未返回对象")
        score = extract_yes_no_score(body)
        return (
            RerankResult(
                index=candidate.index,
                chunk_id=candidate.chunk_id,
                score=score,
                rrf_score=candidate.rrf_score,
            ),
            int(body.get("prompt_eval_count") or 0),
        )

    def _degrade(self, exc: Exception, event: str) -> None:
        self._status = "degraded"
        self._cooldown_until = time.monotonic() + self.settings.rag_rerank_retry_seconds
        self._message = f"精排暂时降级，当前使用 RRF：{type(exc).__name__}"
        audit(
            event,
            level=logging.WARNING,
            status="degraded",
            provider="ollama",
            model=self.settings.rag_rerank_model,
            error_type=type(exc).__name__,
            message=str(exc),
            cooldown_seconds=self.settings.rag_rerank_retry_seconds,
        )
        audit(
            "reranker.cooldown.started",
            level=logging.WARNING,
            status="cooldown",
            retry_seconds=self.settings.rag_rerank_retry_seconds,
        )

    async def rerank(
        self, query: str, candidates: list[RerankCandidate]
    ) -> RerankBatch:
        if not candidates:
            return RerankBatch([], 0.0, 0)
        if self._cooldown_until > time.monotonic():
            raise RerankerUnavailable("Ollama Reranker 正处于冷却期")
        if self.descriptor is None:
            raise RerankerUnavailable("Ollama Reranker 尚未就绪")
        if self._status == "degraded":
            self._status = "ready"
            self._message = "Ollama 法条精排正在重试"
        if self._status != "ready":
            raise RerankerUnavailable("Ollama Reranker 尚未就绪")
        started = time.perf_counter()
        async with trace(
            "law_rag.rerank",
            run_type="retriever",
            inputs={
                "provider": "ollama",
                "model": self.descriptor.model,
                "model_digest": self.descriptor.digest,
                "candidate_count": len(candidates),
            },
        ) as run:
            try:
                async with asyncio.timeout(self.settings.rag_rerank_stage_timeout_seconds):
                    scored = await asyncio.gather(
                        *(self._score(query, candidate) for candidate in candidates)
                    )
            except TimeoutError as exc:
                self._degrade(exc, "reranker.call.timeout")
                if run is not None:
                    run.end(error="Reranker stage timeout")
                raise RerankerUnavailable("Ollama Reranker 阶段超时") from exc
            except Exception as exc:
                self._degrade(exc, "reranker.call.failed")
                if run is not None:
                    run.end(error=type(exc).__name__)
                raise RerankerUnavailable("Ollama Reranker 调用失败") from exc
            duration_ms = round((time.perf_counter() - started) * 1000, 3)
            results = [item[0] for item in scored]
            prompt_tokens = sum(item[1] for item in scored)
            results.sort(key=lambda item: (-item.score, -item.rrf_score, item.index))
            self._status = "ready"
            self._message = "Ollama 法条精排已就绪"
            audit(
                "reranker.call.completed",
                status="success",
                candidate_count=len(candidates),
                prompt_tokens=prompt_tokens,
                duration_ms=duration_ms,
            )
            if run is not None:
                run.end(outputs={
                    "candidate_count": len(candidates),
                    "prompt_tokens": prompt_tokens,
                    "duration_ms": duration_ms,
                    "matches": [
                        {
                            "chunk_id": item.chunk_id,
                            "rrf": round(item.rrf_score, 8),
                            "rerank": round(item.score, 8),
                        }
                        for item in results
                    ],
                })
            return RerankBatch(results, duration_ms, prompt_tokens)

    async def close(self) -> None:
        if self.client is not None:
            await self.client.aclose()
            self.client = None


class TEIReranker:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.base_url = settings.rag_rerank_base_url.rstrip("/")
        self.client: httpx.AsyncClient | None = None
        self.descriptor: RerankerDescriptor | None = None
        self._status = "checking"
        self._message = "正在检查 TEI BGE 法条精排模型"
        self._cooldown_until = 0.0
        self._managed = False

    async def _client(self) -> httpx.AsyncClient:
        if self.client is None:
            self.client = httpx.AsyncClient(
                timeout=self.settings.rag_rerank_request_timeout_seconds
            )
        return self.client

    async def prepare(self) -> RerankerDescriptor:
        audit(
            "reranker.check.started",
            status="checking",
            provider="tei",
            model=self.settings.rag_rerank_model,
        )
        self._status = "checking"
        try:
            runtime = get_tei_reranker_runtime_info()
            if runtime is not None and runtime.model == self.settings.rag_rerank_model:
                if not runtime.available:
                    raise RuntimeError("TEI Reranker 预热失败")
                model = runtime.model
                model_sha = runtime.model_sha
                self._managed = runtime.managed
            else:
                client = await self._client()
                health = await client.get(f"{self.base_url}/health")
                health.raise_for_status()
                info = await client.get(f"{self.base_url}/info")
                info.raise_for_status()
                fallback = resolve_cached_model_revision(
                    getattr(self.settings, "rag_rerank_model_cache", ""),
                    self.settings.rag_rerank_model,
                    getattr(self.settings, "rag_rerank_model_revision", ""),
                )
                model, model_sha = validate_tei_reranker_info(
                    info.json(), self.settings.rag_rerank_model, fallback
                )
            self.descriptor = RerankerDescriptor(
                provider="tei",
                model=model,
                digest=model_sha,
                instruction_version=TEI_RERANK_VERSION,
            )
            self._status = "ready"
            self._message = "TEI BGE 法条精排已就绪"
            audit(
                "reranker.check.completed",
                status="ready",
                provider="tei",
                model=model,
                model_digest=model_sha,
                managed=self._managed,
            )
            return self.descriptor
        except Exception as exc:
            self._status = "degraded"
            self._message = f"精排不可用，当前使用 RRF：{exc}"
            audit(
                "reranker.check.failed",
                level=logging.WARNING,
                status="degraded",
                provider="tei",
                model=self.settings.rag_rerank_model,
                error_type=type(exc).__name__,
                message=str(exc),
            )
            raise

    def status(self) -> dict:
        status = (
            "cooldown" if self._cooldown_until > time.monotonic() else self._status
        )
        return {
            "status": status,
            "message": self._message,
            "provider": "tei",
            "model": self.settings.rag_rerank_model,
            "model_digest": self.descriptor.digest if self.descriptor else "",
            "ranking_version": (
                self.descriptor.ranking_version if self.descriptor else ""
            ),
            "managed": self._managed,
        }

    def _degrade(self, exc: Exception, event: str) -> None:
        self._status = "degraded"
        self._cooldown_until = (
            time.monotonic() + self.settings.rag_rerank_retry_seconds
        )
        self._message = f"精排暂时降级，当前使用 RRF：{type(exc).__name__}"
        audit(
            event,
            level=logging.WARNING,
            status="degraded",
            provider="tei",
            model=self.settings.rag_rerank_model,
            error_type=type(exc).__name__,
            message=str(exc),
            cooldown_seconds=self.settings.rag_rerank_retry_seconds,
        )
        audit(
            "reranker.cooldown.started",
            level=logging.WARNING,
            status="cooldown",
            provider="tei",
            retry_seconds=self.settings.rag_rerank_retry_seconds,
        )

    @staticmethod
    def _token_count(response: httpx.Response, body: object) -> int:
        for header in ("x-compute-tokens", "x-prompt-tokens"):
            value = response.headers.get(header)
            if value and value.isdigit():
                return int(value)
        if isinstance(body, dict) and isinstance(body.get("metadata"), dict):
            value = body["metadata"].get("compute_tokens")
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        return 0

    async def rerank(
        self, query: str, candidates: list[RerankCandidate]
    ) -> RerankBatch:
        if not candidates:
            return RerankBatch([], 0.0, 0)
        if self._cooldown_until > time.monotonic():
            raise RerankerUnavailable("TEI Reranker 正处于冷却期")
        if self.descriptor is None:
            raise RerankerUnavailable("TEI Reranker 尚未就绪")
        if self._status == "degraded":
            self._status = "ready"
            self._message = "TEI BGE 法条精排正在重试"
        if self._status != "ready":
            raise RerankerUnavailable("TEI Reranker 尚未就绪")

        started = time.perf_counter()
        async with trace(
            "law_rag.rerank",
            run_type="retriever",
            inputs={
                "provider": "tei",
                "model": self.descriptor.model,
                "model_sha": self.descriptor.digest,
                "candidate_count": len(candidates),
            },
        ) as run:
            try:
                async with asyncio.timeout(
                    self.settings.rag_rerank_stage_timeout_seconds
                ):
                    response = await (await self._client()).post(
                        f"{self.base_url}/rerank",
                        json={
                            "query": query,
                            "texts": [candidate.text for candidate in candidates],
                            "truncate": True,
                            "raw_scores": False,
                            "return_text": False,
                        },
                    )
                    response.raise_for_status()
                    body = response.json()
                    ranks = parse_tei_ranks(body, len(candidates))
            except TimeoutError as exc:
                self._degrade(exc, "reranker.call.timeout")
                if run is not None:
                    run.end(error="Reranker stage timeout")
                raise RerankerUnavailable("TEI Reranker 阶段超时") from exc
            except Exception as exc:
                self._degrade(exc, "reranker.call.failed")
                if run is not None:
                    run.end(error=type(exc).__name__)
                raise RerankerUnavailable("TEI Reranker 调用失败") from exc

            duration_ms = round((time.perf_counter() - started) * 1000, 3)
            prompt_tokens = self._token_count(response, body)
            results = []
            for rank in ranks:
                candidate = candidates[int(rank["index"])]
                results.append(RerankResult(
                    index=candidate.index,
                    chunk_id=candidate.chunk_id,
                    score=float(rank["score"]),
                    rrf_score=candidate.rrf_score,
                ))
            results.sort(key=lambda item: (-item.score, -item.rrf_score, item.index))
            self._status = "ready"
            self._message = "TEI BGE 法条精排已就绪"
            audit(
                "reranker.call.completed",
                status="success",
                provider="tei",
                candidate_count=len(candidates),
                prompt_tokens=prompt_tokens,
                duration_ms=duration_ms,
            )
            if run is not None:
                run.end(outputs={
                    "candidate_count": len(candidates),
                    "prompt_tokens": prompt_tokens,
                    "duration_ms": duration_ms,
                    "matches": [
                        {
                            "chunk_id": item.chunk_id,
                            "rrf": round(item.rrf_score, 8),
                            "rerank": round(item.score, 8),
                        }
                        for item in results
                    ],
                })
            return RerankBatch(results, duration_ms, prompt_tokens)

    async def close(self) -> None:
        if self.client is not None:
            await self.client.aclose()
            self.client = None


def create_reranker(settings) -> Reranker | None:
    if not getattr(settings, "rag_rerank_enabled", False):
        return None
    provider = getattr(settings, "rag_rerank_provider", "tei")
    if provider == "tei":
        return TEIReranker(settings)
    if provider == "ollama":
        return OllamaReranker(settings)
    raise ValueError(f"不支持 RAG_RERANK_PROVIDER={provider}")
