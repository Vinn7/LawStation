import asyncio
import hashlib
import json
import logging
import os
import random
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import jieba
import numpy as np
from filelock import FileLock, Timeout
from langsmith import trace
from rank_bm25 import BM25Okapi

from backend.app.core.config import Settings, get_settings
from backend.app.core.logging import audit
from backend.app.core.ollama import ollama_runtime_status
from mcp_servers.law_rag.embeddings import create_embedding_provider
from mcp_servers.law_rag.reranker import (
    RerankCandidate,
    RerankerUnavailable,
    create_reranker,
)

CHUNKER_VERSION = "law-article-v1"
QUERY_INSTRUCTION_VERSION = "legal-query-v1"


def tokens(text: str) -> list[str]:
    return [item for item in jieba.lcut(text.lower()) if item.strip()]


def normalize_law_name(value: str) -> str:
    normalized = re.sub(r"[\s《》]", "", value or "").lower()
    return normalized.removeprefix("中华人民共和国")


def normalize_article_number(value: str) -> str:
    return re.sub(r"\s+", "", value or "")


def split_text(text: str, maximum: int, overlap: int) -> list[str]:
    if len(text) <= maximum:
        return [text]
    parts, start = [], 0
    while start < len(text):
        end = min(start + maximum, len(text))
        if end < len(text):
            candidates = [text.rfind(mark, start + maximum // 2, end) for mark in ("\n", "。", "；")]
            boundary = max(candidates)
            if boundary > start:
                end = boundary + 1
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [part for part in parts if part]


def load_chunks(path: Path, maximum: int, overlap: int) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    chunks = []
    for name, content in records.items():
        match = re.search(r"(第[零一二三四五六七八九十百千万0-9]+条)$", name)
        article = match.group(1) if match else ""
        law_name = name[: -len(article)] if article else name
        source_id = hashlib.sha1(name.encode()).hexdigest()
        for index, part in enumerate(split_text(content, maximum, overlap)):
            content_hash = hashlib.sha256(part.encode()).hexdigest()
            chunk_id = hashlib.sha1(f"{source_id}:{index}:{content_hash}".encode()).hexdigest()
            chunks.append({
                "document_id": source_id,
                "chunk_id": chunk_id,
                "chunk_index": index,
                "law_name": law_name,
                "article_number": article,
                "content": part,
                "text": f"{name} {part}",
            })
    return chunks


class LawSearchEngine:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.path = Path(self.settings.law_data_path)
        self.index_root = Path(self.settings.index_dir)
        self.final_dir = self.index_root / "law"
        self.source_sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.docs = load_chunks(
            self.path,
            self.settings.index_chunk_max_chars,
            self.settings.index_chunk_overlap_chars,
        )
        self._law_name_indices: dict[str, list[int]] = {}
        self._article_map: dict[tuple[str, str], list[dict]] = {}
        for index, document in enumerate(self.docs):
            normalized_name = normalize_law_name(document["law_name"])
            self._law_name_indices.setdefault(normalized_name, []).append(index)
            article_key = (normalized_name, normalize_article_number(document["article_number"]))
            self._article_map.setdefault(article_key, []).append(document)
        self.bm25 = BM25Okapi([tokens(document["text"]) for document in self.docs])
        self.faiss = None
        self._dense_lock = asyncio.Lock()
        self._build_task: asyncio.Task | None = None
        self.embedding_provider = create_embedding_provider(self.settings)
        self.embedding_descriptor = None
        self.reranker = create_reranker(self.settings)
        self.reranker_descriptor = None
        self.fingerprint = self._fingerprint()
        runtime = ollama_runtime_status()
        self._state = {
            "status": "checking",
            "source_file": self.path.name,
            "fingerprint": self.fingerprint,
            "source_documents": len(json.loads(self.path.read_text(encoding="utf-8"))),
            "processed_chunks": 0,
            "total_chunks": len(self.docs),
            "progress": 0.0,
            "dense_enabled": False,
            "embedding_provider": self.settings.embedding_provider,
            "embedding_model": self.settings.embedding_model,
            "embedding_model_digest": "",
            "ollama_available": runtime["available"],
            "ollama_managed": runtime["managed"],
            "reranker_enabled": bool(self.reranker),
            "reranker_status": "checking" if self.reranker else "disabled",
            "reranker_provider": getattr(self.settings, "rag_rerank_provider", "tei"),
            "reranker_model": getattr(self.settings, "rag_rerank_model", ""),
            "reranker_model_digest": "",
            "reranker_managed": False,
            "reranker_candidate_count": getattr(
                self.settings, "rag_rerank_candidate_count", 0
            ),
            "reranker_message": (
                "正在检查本地法条精排模型" if self.reranker else "法条精排已关闭"
            ),
            "message": "正在检查法律索引",
        }

    def _fingerprint(self) -> str:
        payload = {
            "source_sha256": self.source_sha256,
            "chunker_version": CHUNKER_VERSION,
            "max_chars": self.settings.index_chunk_max_chars,
            "overlap_chars": self.settings.index_chunk_overlap_chars,
            "embedding_provider": self.settings.embedding_provider,
            "embedding_model": self.settings.embedding_model,
            "embedding_model_digest": (
                self.embedding_descriptor.digest if self.embedding_descriptor else "unprepared"
            ),
            "embedding_dimension": self.settings.embedding_dimension,
            "query_instruction_version": QUERY_INSTRUCTION_VERSION,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def status(self) -> dict:
        self._sync_reranker_state()
        return dict(self._state)

    def _set_state(self, **values) -> None:
        self._state.update(values)

    def _sync_reranker_state(self) -> None:
        if self.reranker is None:
            return
        reranker = self.reranker.status()
        self._state.update({
            "reranker_status": reranker["status"],
            "reranker_provider": reranker["provider"],
            "reranker_model": reranker["model"],
            "reranker_model_digest": reranker["model_digest"],
            "reranker_managed": bool(reranker.get("managed", False)),
            "reranker_message": reranker["message"],
        })

    async def _initialize_reranker(self) -> None:
        if self.reranker is None:
            return
        try:
            self.reranker_descriptor = await self.reranker.prepare()
        except Exception:
            self._sync_reranker_state()
            if getattr(self.settings, "rag_rerank_required", False):
                raise
        else:
            self._sync_reranker_state()

    def _read_valid_index(self, directory: Path):
        manifest_path = directory / "manifest.json"
        chunks_path = directory / "chunks.jsonl"
        embeddings_path = directory / "embeddings.npy"
        faiss_path = directory / "law.faiss"
        if not all(path.is_file() for path in (manifest_path, chunks_path, embeddings_path, faiss_path)):
            return None
        try:
            import faiss

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            with chunks_path.open(encoding="utf-8") as chunk_file:
                metadata_count = sum(1 for line in chunk_file if line.strip())
            embeddings = np.load(embeddings_path, mmap_mode="r")
            index = faiss.read_index(str(faiss_path))
            valid = (
                manifest.get("fingerprint") == self.fingerprint
                and manifest.get("chunk_count") == len(self.docs) == metadata_count
                and embeddings.shape == (len(self.docs), self.settings.embedding_dimension)
                and index.ntotal == len(self.docs)
                and index.d == self.settings.embedding_dimension
            )
            if not valid:
                return None
            return index
        except Exception:  # noqa: BLE001 - any corrupt artifact must trigger a safe rebuild
            return None

    def _validate_and_load(self) -> bool:
        index = self._read_valid_index(self.final_dir)
        if index is None:
            return False
        self.faiss = index
        return True

    async def initialize_index(self, wait: bool = False, force: bool = False) -> None:
        await self._initialize_reranker()
        try:
            self.embedding_descriptor = await self.embedding_provider.prepare()
        except Exception as exc:  # noqa: BLE001 - direct MCP/debug mode must retain BM25
            runtime = ollama_runtime_status()
            self._set_state(
                status="degraded",
                dense_enabled=False,
                ollama_available=runtime["available"],
                ollama_managed=runtime["managed"],
                message=f"Embedding 服务不可用，当前仅使用 BM25：{exc}",
            )
            audit(
                "index.build.failed",
                level=logging.WARNING,
                status="degraded",
                error_type=type(exc).__name__,
                message=str(exc),
            )
            return
        self.fingerprint = self._fingerprint()
        runtime = ollama_runtime_status()
        self._set_state(
            fingerprint=self.fingerprint,
            embedding_provider=self.embedding_descriptor.provider,
            embedding_model=self.embedding_descriptor.model,
            embedding_model_digest=self.embedding_descriptor.digest,
            ollama_available=(self.embedding_descriptor.provider == "ollama"),
            ollama_managed=(runtime["managed"] if self.embedding_descriptor.provider == "ollama" else False),
        )
        audit("index.check.started", fingerprint=self.fingerprint, source_file=self.path.name)
        valid = False if force else await asyncio.to_thread(self._validate_and_load)
        if valid:
            self._set_state(status="ready", processed_chunks=len(self.docs), progress=1.0, dense_enabled=True, message="法律向量索引已就绪")
            audit("index.check.skipped", status="ready", fingerprint=self.fingerprint, chunk_count=len(self.docs))
            audit("index.loaded", status="ready", chunk_count=len(self.docs))
            return
        if not self.settings.index_auto_build:
            self._set_state(status="degraded", message="自动建库已关闭，当前仅使用 BM25")
            return
        provider_name = "Ollama 本地" if self.embedding_descriptor.provider == "ollama" else "远程"
        message = f"正在生成{provider_name}法律向量，当前使用 BM25"
        self._set_state(status="building", message=message)
        if self._build_task is None or self._build_task.done():
            self._build_task = asyncio.create_task(self._build(force=force), name="law-index-build")
        if wait:
            await self._build_task

    async def close(self) -> None:
        if self._build_task and not self._build_task.done():
            self._build_task.cancel()
            try:
                await self._build_task
            except asyncio.CancelledError:
                pass
        await self.embedding_provider.close()
        if self.reranker is not None:
            await self.reranker.close()

    async def _embed(self, texts: list[str]) -> np.ndarray:
        return await self.embedding_provider.embed_documents(texts)

    async def _embed_query(self, query: str) -> np.ndarray:
        async with trace(
            "law_rag.query_embedding",
            run_type="chain",
            inputs={
                "query": query,
                "provider": self.settings.embedding_provider,
                "model": self.settings.embedding_model,
            },
        ) as run:
            vector = await self.embedding_provider.embed_query(query)
            if run is not None:
                run.end(outputs={
                    "vector_count": int(vector.shape[0]) if vector.ndim > 1 else 1,
                    "dimension": int(vector.shape[-1]),
                    "provider": self.settings.embedding_provider,
                    "model": self.settings.embedding_model,
                })
            return vector

    async def _embed_batch(self, texts: list[str], batch_start: int) -> np.ndarray:
        import httpx

        retries = max(0, self.settings.index_embedding_max_retries)
        for attempt in range(retries + 1):
            try:
                return await self._embed(texts)
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                retryable = status_code in {408, 429} or status_code >= 500
                if not retryable or attempt >= retries:
                    raise
                error_type = f"HTTP{status_code}"
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt >= retries:
                    raise
                error_type = type(exc).__name__
            delay = min(
                self.settings.index_embedding_retry_base_seconds * (2**attempt),
                self.settings.index_embedding_retry_max_seconds,
            )
            delay += random.uniform(0, min(1.0, delay * 0.25))
            audit(
                "index.embedding.retry",
                level=logging.WARNING,
                status="retrying",
                batch_start=batch_start,
                batch_size=len(texts),
                attempt=attempt + 1,
                error_type=error_type,
                retry_delay_seconds=round(delay, 3),
            )
            await asyncio.sleep(delay)

        raise RuntimeError("Embedding 重试流程异常结束")

    def _load_batch(self, path: Path, expected_rows: int) -> np.ndarray | None:
        try:
            vectors = np.load(path, allow_pickle=False)
            if vectors.dtype != np.float32:
                return None
            if vectors.shape != (expected_rows, self.settings.embedding_dimension):
                return None
            if not np.isfinite(vectors).all():
                return None
            return vectors
        except Exception:  # noqa: BLE001 - a corrupt checkpoint batch must be regenerated
            return None

    def _chunks_file_is_valid(self, path: Path) -> bool:
        try:
            with path.open(encoding="utf-8") as chunk_file:
                for expected, line in zip(self.docs, chunk_file, strict=True):
                    if json.loads(line).get("chunk_id") != expected["chunk_id"]:
                        return False
            return True
        except (OSError, ValueError, json.JSONDecodeError):
            return False

    async def _build(self, force: bool = False) -> None:
        self.index_root.mkdir(parents=True, exist_ok=True)
        # Acquisition runs in a worker thread; disable thread-local ownership so
        # the event-loop thread can reliably release the same process lock.
        lock = FileLock(str(self.index_root / ".build.lock"), thread_local=False)
        try:
            await asyncio.to_thread(lock.acquire, timeout=0)
        except Timeout:
            audit("index.build.resumed", status="waiting", message="另一个进程正在构建索引")
            for _ in range(120):
                await asyncio.sleep(1)
                if await asyncio.to_thread(self._validate_and_load):
                    self._set_state(status="ready", processed_chunks=len(self.docs), progress=1.0, dense_enabled=True, message="法律向量索引已就绪")
                    audit("index.loaded", status="ready", fingerprint=self.fingerprint)
                    return
            self._set_state(status="failed", message="等待其他索引构建进程超时")
            audit("index.build.failed", status="failed", error_type="LockTimeout", message="等待其他索引构建进程超时")
            return
        staging = self.index_root / f".staging-{self.fingerprint}"
        try:
            if not force and await asyncio.to_thread(self._validate_and_load):
                self._set_state(status="ready", processed_chunks=len(self.docs), progress=1.0, dense_enabled=True, message="法律向量索引已就绪")
                audit("index.check.skipped", status="ready", fingerprint=self.fingerprint)
                return
            if force and staging.exists():
                await asyncio.to_thread(shutil.rmtree, staging)
            staging.mkdir(parents=True, exist_ok=True)
            chunks_path = staging / "chunks.jsonl"
            if not await asyncio.to_thread(self._chunks_file_is_valid, chunks_path):
                chunks_tmp = staging / f".chunks-{uuid4().hex}.tmp"
                chunks_tmp.write_text("".join(json.dumps(doc, ensure_ascii=False) + "\n" for doc in self.docs), encoding="utf-8")
                os.replace(chunks_tmp, chunks_path)
            provider_limit = (
                self.settings.ollama_embedding_batch_size
                if self.embedding_descriptor.provider == "ollama"
                else 20
            )
            batch_size = max(1, min(self.settings.index_build_batch_size, provider_limit, 20))
            processed = 0
            resumed = 0
            last_logged_percent = -5
            audit("index.build.started", fingerprint=self.fingerprint, chunk_count=len(self.docs))
            for start in range(0, len(self.docs), batch_size):
                batch_path = staging / f"batch-{start:08d}.npy"
                expected = min(batch_size, len(self.docs) - start)
                vector_batch = self._load_batch(batch_path, expected) if batch_path.is_file() else None
                if vector_batch is None and batch_path.exists():
                    batch_path.unlink()
                if vector_batch is None:
                    vector_batch = await self._embed_batch(
                        [doc["text"] for doc in self.docs[start : start + batch_size]],
                        batch_start=start,
                    )
                    temporary = staging / f".{batch_path.name}-{uuid4().hex}.tmp.npy"
                    np.save(temporary, vector_batch)
                    os.replace(temporary, batch_path)
                else:
                    resumed += len(vector_batch)
                    if resumed == len(vector_batch):
                        audit(
                            "index.build.resumed",
                            status="resumed",
                            processed_chunks=start + len(vector_batch),
                            total_chunks=len(self.docs),
                        )
                processed += len(vector_batch)
                progress = processed / len(self.docs)
                self._set_state(processed_chunks=processed, progress=progress)
                checkpoint_tmp = staging / ".checkpoint.tmp"
                checkpoint_tmp.write_text(json.dumps({"fingerprint": self.fingerprint, "processed_chunks": processed}), encoding="utf-8")
                os.replace(checkpoint_tmp, staging / "checkpoint.json")
                progress_percent = int(progress * 100)
                if processed == len(self.docs) or progress_percent >= last_logged_percent + 5:
                    audit("index.build.progress", status="building", processed_chunks=processed, total_chunks=len(self.docs), progress=round(progress, 4))
                    last_logged_percent = progress_percent
            import faiss

            index = faiss.IndexFlatIP(self.settings.embedding_dimension)
            embeddings_path = staging / "embeddings.npy"
            embeddings = np.lib.format.open_memmap(
                embeddings_path,
                mode="w+",
                dtype="float32",
                shape=(len(self.docs), self.settings.embedding_dimension),
            )
            for start in range(0, len(self.docs), batch_size):
                expected = min(batch_size, len(self.docs) - start)
                batch_path = staging / f"batch-{start:08d}.npy"
                vector_batch = self._load_batch(batch_path, expected)
                if vector_batch is None:
                    raise ValueError(f"Embedding 批次校验失败: {start}")
                normalized = vector_batch.copy()
                faiss.normalize_L2(normalized)
                embeddings[start : start + expected] = normalized
                index.add(normalized)
            embeddings.flush()
            del embeddings
            faiss.write_index(index, str(staging / "law.faiss"))
            manifest = {
                "fingerprint": self.fingerprint,
                "source_sha256": self.source_sha256,
                "chunker_version": CHUNKER_VERSION,
                "max_chars": self.settings.index_chunk_max_chars,
                "overlap_chars": self.settings.index_chunk_overlap_chars,
                "embedding_model": self.settings.embedding_model,
                "embedding_provider": self.embedding_descriptor.provider,
                "embedding_model_digest": self.embedding_descriptor.digest,
                "embedding_dimension": self.settings.embedding_dimension,
                "query_instruction_version": QUERY_INSTRUCTION_VERSION,
                "source_document_count": self._state["source_documents"],
                "chunk_count": len(self.docs),
                "built_at": datetime.now(UTC).isoformat(),
                "status": "ready",
            }
            (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            validated_index = await asyncio.to_thread(self._read_valid_index, staging)
            if validated_index is None:
                raise ValueError("生成的法律向量索引未通过完整性校验")
            for path in staging.glob("batch-*.npy"):
                path.unlink()
            (staging / "checkpoint.json").unlink(missing_ok=True)
            backup = self.index_root / ".law-backup"
            if backup.exists():
                shutil.rmtree(backup)
            if self.final_dir.exists():
                os.replace(self.final_dir, backup)
            try:
                os.replace(staging, self.final_dir)
            except Exception:
                if backup.exists():
                    os.replace(backup, self.final_dir)
                raise
            if backup.exists():
                shutil.rmtree(backup)
            async with self._dense_lock:
                self.faiss = validated_index
            self._set_state(status="ready", processed_chunks=len(self.docs), progress=1.0, dense_enabled=True, message="法律向量索引已就绪")
            self._record_manifest()
            audit("index.build.completed", status="ready", fingerprint=self.fingerprint, chunk_count=len(self.docs))
            audit("index.switched", status="ready", fingerprint=self.fingerprint)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - background failures must degrade, not crash the app
            self._set_state(status="failed", dense_enabled=self.faiss is not None, message="向量建库失败，当前使用可用的降级检索")
            audit("index.build.failed", status="failed", error_type=type(exc).__name__, message=str(exc))
        finally:
            lock.release()

    def _record_manifest(self) -> None:
        try:
            from sqlalchemy import select

            from backend.app.db.models import IndexManifest
            from backend.app.db.session import SessionLocal

            with SessionLocal() as db:
                if not db.scalar(select(IndexManifest).where(IndexManifest.data_version == self.fingerprint)):
                    db.add(IndexManifest(
                        data_version=self.fingerprint,
                        embedding_model=self.settings.embedding_model,
                        embedding_dimension=self.settings.embedding_dimension,
                        document_count=len(self.docs),
                    ))
                    db.commit()
        except Exception as exc:  # noqa: BLE001 - audit persistence cannot invalidate a built index
            audit("index.manifest.persist_failed", status="failed", error_type=type(exc).__name__, message=str(exc))

    def _filtered_indices(self, filters: dict | None) -> list[int] | None:
        with trace("law_rag.filter", inputs={"filters": filters or {}}) as run:
            if filters is None:
                result = None
            else:
                if not isinstance(filters, dict):
                    raise TypeError("filters 必须是对象")
                unknown = set(filters) - {"law_name"}
                if unknown:
                    raise ValueError(f"不支持的过滤字段：{', '.join(sorted(unknown))}")
                law_name = filters.get("law_name")
                if law_name in (None, ""):
                    result = None
                else:
                    if not isinstance(law_name, str):
                        raise TypeError("law_name 过滤条件必须是字符串")
                    normalized = normalize_law_name(law_name)
                    result = None if not normalized else [
                        index
                        for stored_name, indexes in self._law_name_indices.items()
                        if normalized in stored_name or stored_name in normalized
                        for index in indexes
                    ]
            if run is not None:
                run.end(outputs={
                    "candidate_count": len(self.docs) if result is None else len(result),
                    "filter_applied": result is not None,
                })
            return result

    def _lexical(
        self, query: str, pool: int, eligible_indices: list[int] | None = None
    ) -> list[tuple[int, float]]:
        with trace(
            "law_rag.bm25",
            run_type="retriever",
            inputs={"query": query, "pool": pool},
        ) as run:
            scores = self.bm25.get_scores(tokens(query))
            candidates = eligible_indices if eligible_indices is not None else range(len(self.docs))
            ranked = sorted(
                ((int(index), float(scores[index])) for index in candidates),
                key=lambda item: item[1],
                reverse=True,
            )
            minimum = float(getattr(self.settings, "rag_bm25_min_score", 0.01))
            result = [item for item in ranked[:pool] if item[1] >= minimum]
            if run is not None:
                run.end(outputs={
                    "matches": [
                        {
                            "document_id": self.docs[index]["document_id"],
                            "chunk_id": self.docs[index]["chunk_id"],
                            "score": score,
                        }
                        for index, score in result
                    ],
                    "minimum_score": minimum,
                })
            return result

    async def _faiss_search(self, vector: np.ndarray, pool: int):
        async with trace(
            "law_rag.faiss",
            run_type="retriever",
            inputs={"pool": pool, "dimension": self.settings.embedding_dimension},
        ) as run:
            async with self._dense_lock:
                dense_scores, dense = await asyncio.to_thread(
                    self.faiss.search, vector, pool
                )
            if run is not None:
                run.end(outputs={
                    "candidate_count": int(sum(index >= 0 for index in dense[0])),
                    "matches": [
                        {
                            "document_id": self.docs[int(index)]["document_id"],
                            "chunk_id": self.docs[int(index)]["chunk_id"],
                            "score": float(score),
                        }
                        for index, score in zip(dense[0], dense_scores[0], strict=True)
                        if index >= 0
                    ][:100],
                    "matches_truncated": int(sum(index >= 0 for index in dense[0])) > 100,
                })
            return dense_scores, dense

    def _rrf_order(self, ranks: dict[int, float], top_k: int) -> list[int]:
        with trace(
            "law_rag.rrf",
            inputs={"candidate_count": len(ranks), "top_k": top_k},
        ) as run:
            minimum = float(getattr(self.settings, "rag_rrf_min_score", 0.01))
            ordered = [
                index
                for index, _ in sorted(ranks.items(), key=lambda item: item[1], reverse=True)
                if ranks[index] >= minimum
            ][:top_k]
            if run is not None:
                run.end(outputs={
                    "minimum_score": minimum,
                    "matches": [
                        {
                            "document_id": self.docs[index]["document_id"],
                            "chunk_id": self.docs[index]["chunk_id"],
                            "rrf": round(ranks[index], 8),
                        }
                        for index in ordered
                    ],
                })
            return ordered

    async def search(self, query, top_k=8, filters=None, retrieval_mode: str | None = None):
        mode = retrieval_mode or getattr(self.settings, "rag_retrieval_mode", "hybrid")
        async with trace(
            "law_rag.search_laws",
            run_type="retriever",
            inputs={"query": query, "top_k": top_k, "filters": filters, "mode": mode},
        ) as run:
            if mode not in {"bm25", "hybrid"}:
                raise ValueError("retrieval_mode 必须是 bm25 或 hybrid")
            top_k = max(1, min(int(top_k), 20))
            eligible_indices = self._filtered_indices(filters)
            if eligible_indices == []:
                if run is not None:
                    run.end(outputs={"documents": [], "retrieval_status": "no_match"})
                return []
            eligible = set(eligible_indices) if eligible_indices is not None else None
            candidate_count = len(eligible_indices) if eligible_indices is not None else len(self.docs)
            pool = min(candidate_count, max(30, top_k * 4))
            ranks: dict[int, float] = {}
            sources: dict[int, list[str]] = {}
            raw_scores: dict[int, dict[str, float]] = {}
            lexical = await asyncio.to_thread(self._lexical, query, pool, eligible_indices)
            for rank, (index, score) in enumerate(lexical):
                ranks[index] = ranks.get(index, 0) + 1 / (61 + rank)
                sources.setdefault(index, []).append("bm25")
                raw_scores.setdefault(index, {})["bm25"] = score
            if mode == "hybrid" and self.faiss is not None and self.embedding_descriptor is not None:
                vector = await self._embed_query(query)
                import faiss

                faiss.normalize_L2(vector)
                dense_pool = len(self.docs) if eligible is not None else pool
                dense_scores, dense = await self._faiss_search(vector, dense_pool)
                minimum_dense = float(getattr(self.settings, "rag_dense_min_score", 0.20))
                accepted_rank = 0
                for index, score in zip(dense[0], dense_scores[0], strict=True):
                    if index < 0:
                        continue
                    index = int(index)
                    score = float(score)
                    if eligible is not None and index not in eligible:
                        continue
                    if score < minimum_dense:
                        continue
                    ranks[index] = ranks.get(index, 0) + 1 / (61 + accepted_rank)
                    sources.setdefault(index, []).append("dense")
                    raw_scores.setdefault(index, {})["dense"] = score
                    accepted_rank += 1
                    if accepted_rank >= pool:
                        break
            rerank_limit = max(
                top_k,
                int(getattr(self.settings, "rag_rerank_candidate_count", top_k)),
            )
            candidate_order = self._rrf_order(ranks, rerank_limit)
            ordered = candidate_order[:top_k]
            rerank_scores: dict[int, float] = {}
            rerank_applied = False
            rerank_duration_ms = 0.0
            rerank_prompt_tokens = 0
            if self.reranker is not None and candidate_order:
                try:
                    batch = await self.reranker.rerank(
                        query,
                        [
                            RerankCandidate(
                                index=index,
                                chunk_id=self.docs[index]["chunk_id"],
                                text=self.docs[index]["text"],
                                rrf_score=ranks[index],
                            )
                            for index in candidate_order
                        ],
                    )
                except RerankerUnavailable:
                    self._sync_reranker_state()
                else:
                    minimum_rerank = float(
                        getattr(self.settings, "rag_rerank_min_score", 0.0)
                    )
                    rerank_scores = {
                        item.index: item.score
                        for item in batch.results
                        if item.score >= minimum_rerank
                    }
                    ordered = [
                        item.index
                        for item in batch.results
                        if item.score >= minimum_rerank
                    ][:top_k]
                    rerank_applied = True
                    rerank_duration_ms = batch.duration_ms
                    rerank_prompt_tokens = batch.prompt_tokens
                    self._sync_reranker_state()
            state = self.status()
            results = [{
                **{key: value for key, value in self.docs[index].items() if key != "text"},
                "retrieval_sources": sources[index],
                "rank": rank + 1,
                "retrieval_scores": {
                    **raw_scores.get(index, {}),
                    "rrf": round(ranks[index], 8),
                    **(
                        {"rerank": round(rerank_scores[index], 8)}
                        if index in rerank_scores else {}
                    ),
                },
                "data_version": self.fingerprint,
                "retrieval_mode": mode,
                "rerank_applied": rerank_applied,
                "rerank_provider": (
                    self.reranker_descriptor.provider
                    if rerank_applied and self.reranker_descriptor else None
                ),
                "ranking_version": (
                    self.reranker_descriptor.ranking_version
                    if rerank_applied and self.reranker_descriptor else "rrf-v1"
                ),
                "rerank_duration_ms": rerank_duration_ms,
                "rerank_prompt_tokens": rerank_prompt_tokens,
                "rerank_candidate_count": len(candidate_order) if rerank_applied else 0,
                "dense_enabled": state["dense_enabled"],
                "index_status": state["status"],
            } for rank, index in enumerate(ordered)]
            if run is not None:
                run.end(outputs={
                    "documents": results,
                    "retrieval_status": "matched" if results else "no_match",
                })
            return results

    def get(self, law_name, article_number):
        with trace(
            "law_rag.get_article",
            run_type="retriever",
            inputs={"law_name": law_name, "article_number": article_number},
        ) as run:
            normalized_name = normalize_law_name(law_name)
            normalized_article = normalize_article_number(article_number)
            matches = self._article_map.get((normalized_name, normalized_article), [])
            if not matches:
                matches = [
                    document
                    for (stored_name, stored_article), documents in self._article_map.items()
                    if normalized_article == stored_article
                    and (normalized_name in stored_name or stored_name in normalized_name)
                    for document in documents
                ]
            if len(matches) == 1:
                result = {key: value for key, value in matches[0].items() if key != "text"}
            elif matches:
                result = {
                    "law_name": matches[0]["law_name"],
                    "article_number": matches[0]["article_number"],
                    "chunks": [
                        {key: value for key, value in document.items() if key != "text"}
                        for document in matches
                    ],
                }
            else:
                result = None
            if run is not None:
                run.end(outputs={"result": result, "chunk_count": len(matches)})
            return result
