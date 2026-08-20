import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import jieba
import numpy as np
from filelock import FileLock, Timeout
from rank_bm25 import BM25Okapi

from backend.app.core.config import get_settings
from backend.app.core.logging import audit

CHUNKER_VERSION = "law-article-v1"


def tokens(text: str) -> list[str]:
    return [item for item in jieba.lcut(text.lower()) if item.strip()]


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
    def __init__(self):
        self.settings = get_settings()
        self.path = Path(self.settings.law_data_path)
        self.index_root = Path(self.settings.index_dir)
        self.final_dir = self.index_root / "law"
        self.source_sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.docs = load_chunks(
            self.path,
            self.settings.index_chunk_max_chars,
            self.settings.index_chunk_overlap_chars,
        )
        self.bm25 = BM25Okapi([tokens(document["text"]) for document in self.docs])
        self.faiss = None
        self._dense_lock = asyncio.Lock()
        self._build_task: asyncio.Task | None = None
        self.fingerprint = self._fingerprint()
        self._state = {
            "status": "checking",
            "source_file": self.path.name,
            "fingerprint": self.fingerprint,
            "source_documents": len(json.loads(self.path.read_text(encoding="utf-8"))),
            "processed_chunks": 0,
            "total_chunks": len(self.docs),
            "progress": 0.0,
            "dense_enabled": False,
            "message": "正在检查法律索引",
        }

    def _fingerprint(self) -> str:
        payload = {
            "source_sha256": self.source_sha256,
            "chunker_version": CHUNKER_VERSION,
            "max_chars": self.settings.index_chunk_max_chars,
            "overlap_chars": self.settings.index_chunk_overlap_chars,
            "embedding_model": self.settings.embedding_model,
            "embedding_dimension": self.settings.embedding_dimension,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def status(self) -> dict:
        return dict(self._state)

    def _set_state(self, **values) -> None:
        self._state.update(values)

    def _validate_and_load(self) -> bool:
        manifest_path = self.final_dir / "manifest.json"
        chunks_path = self.final_dir / "chunks.jsonl"
        embeddings_path = self.final_dir / "embeddings.npy"
        faiss_path = self.final_dir / "law.faiss"
        if not all(path.is_file() for path in (manifest_path, chunks_path, embeddings_path, faiss_path)):
            return False
        try:
            import faiss

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            metadata_count = sum(1 for line in chunks_path.read_text(encoding="utf-8").splitlines() if line)
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
                return False
            self.faiss = index
            return True
        except Exception:  # noqa: BLE001 - any corrupt artifact must trigger a safe rebuild
            return False

    async def initialize_index(self, wait: bool = False, force: bool = False) -> None:
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
        if not self.settings.dashscope_api_key:
            self._set_state(status="degraded", message="缺少 DASHSCOPE_API_KEY，当前仅使用 BM25")
            audit("index.build.failed", level=logging.WARNING, status="degraded", error_type="MissingCredential", message="缺少 DASHSCOPE_API_KEY")
            return
        self._set_state(status="building", message="正在生成法律向量")
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

    async def _embed(self, texts: list[str]) -> np.ndarray:
        import httpx

        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                self.settings.dashscope_base_url + "/embeddings",
                headers={"Authorization": f"Bearer {self.settings.dashscope_api_key}"},
                json={
                    "model": self.settings.embedding_model,
                    "input": texts,
                    "dimensions": self.settings.embedding_dimension,
                },
            )
            response.raise_for_status()
            return np.asarray([item["embedding"] for item in response.json()["data"]], dtype="float32")

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
            staging.mkdir(parents=True, exist_ok=True)
            chunks_path = staging / "chunks.jsonl"
            if not chunks_path.exists():
                chunks_path.write_text("".join(json.dumps(doc, ensure_ascii=False) + "\n" for doc in self.docs), encoding="utf-8")
            batch_size = max(1, min(self.settings.index_build_batch_size, 20))
            batches = []
            processed = 0
            audit("index.build.started", fingerprint=self.fingerprint, chunk_count=len(self.docs))
            for start in range(0, len(self.docs), batch_size):
                batch_path = staging / f"batch-{start:08d}.npy"
                if batch_path.is_file():
                    vector_batch = np.load(batch_path)
                    expected = min(batch_size, len(self.docs) - start)
                    if vector_batch.shape != (expected, self.settings.embedding_dimension):
                        batch_path.unlink()
                        vector_batch = None
                    else:
                        audit("index.build.resumed", processed_chunks=start + len(vector_batch), total_chunks=len(self.docs))
                else:
                    vector_batch = None
                if vector_batch is None:
                    vector_batch = await self._embed([doc["text"] for doc in self.docs[start : start + batch_size]])
                    temporary = staging / f".{batch_path.name}-{uuid4().hex}.tmp.npy"
                    np.save(temporary, vector_batch)
                    os.replace(temporary, batch_path)
                batches.append(vector_batch)
                processed += len(vector_batch)
                progress = processed / len(self.docs)
                self._set_state(processed_chunks=processed, progress=progress)
                checkpoint_tmp = staging / ".checkpoint.tmp"
                checkpoint_tmp.write_text(json.dumps({"fingerprint": self.fingerprint, "processed_chunks": processed}), encoding="utf-8")
                os.replace(checkpoint_tmp, staging / "checkpoint.json")
                if processed == len(self.docs) or processed % max(5, len(self.docs) // 20 or 1) < batch_size:
                    audit("index.build.progress", status="building", processed_chunks=processed, total_chunks=len(self.docs), progress=round(progress, 4))
            embeddings = np.vstack(batches).astype("float32")
            import faiss

            faiss.normalize_L2(embeddings)
            index = faiss.IndexFlatIP(self.settings.embedding_dimension)
            index.add(embeddings)
            np.save(staging / "embeddings.npy", embeddings)
            faiss.write_index(index, str(staging / "law.faiss"))
            manifest = {
                "fingerprint": self.fingerprint,
                "source_sha256": self.source_sha256,
                "chunker_version": CHUNKER_VERSION,
                "max_chars": self.settings.index_chunk_max_chars,
                "overlap_chars": self.settings.index_chunk_overlap_chars,
                "embedding_model": self.settings.embedding_model,
                "embedding_dimension": self.settings.embedding_dimension,
                "source_document_count": self._state["source_documents"],
                "chunk_count": len(self.docs),
                "built_at": datetime.now(UTC).isoformat(),
                "status": "ready",
            }
            (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
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
                self.faiss = index
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

    def _lexical(self, query: str, pool: int) -> list[int]:
        scores = self.bm25.get_scores(tokens(query))
        return [int(index) for index in np.argsort(scores)[-pool:][::-1]]

    async def search(self, query, top_k=8, filters=None):
        top_k = max(1, min(int(top_k), 20))
        pool = min(len(self.docs), max(30, top_k * 4))
        ranks, sources = {}, {}
        for rank, index in enumerate(await asyncio.to_thread(self._lexical, query, pool)):
            ranks[index] = ranks.get(index, 0) + 1 / (61 + rank)
            sources.setdefault(index, []).append("bm25")
        if self.faiss is not None and self.settings.dashscope_api_key:
            vector = await self._embed([query])
            import faiss

            faiss.normalize_L2(vector)
            async with self._dense_lock:
                _, dense = await asyncio.to_thread(self.faiss.search, vector, pool)
            for rank, index in enumerate(dense[0]):
                if index >= 0:
                    index = int(index)
                    ranks[index] = ranks.get(index, 0) + 1 / (61 + rank)
                    sources.setdefault(index, []).append("dense")
        law_filter = (filters or {}).get("law_name") if isinstance(filters, dict) else None
        ordered = [
            index for index, _ in sorted(ranks.items(), key=lambda item: item[1], reverse=True)
            if not law_filter or law_filter in self.docs[index]["law_name"]
        ][:top_k]
        state = self.status()
        return [{
            **{key: value for key, value in self.docs[index].items() if key != "text"},
            "retrieval_sources": sources[index],
            "rank": rank + 1,
            "data_version": self.fingerprint,
            "dense_enabled": state["dense_enabled"],
            "index_status": state["status"],
        } for rank, index in enumerate(ordered)]

    def get(self, law_name, article_number):
        for document in self.docs:
            if law_name in document["law_name"] and article_number in document["article_number"]:
                return {key: value for key, value in document.items() if key != "text"}
        return None
