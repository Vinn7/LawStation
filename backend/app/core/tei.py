"""Lifecycle management for a local Hugging Face TEI reranker service."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx
from filelock import FileLock


@dataclass(frozen=True)
class TEIRerankerRuntimeInfo:
    available: bool
    managed: bool
    model: str
    model_sha: str


_runtime_info: TEIRerankerRuntimeInfo | None = None


def get_tei_reranker_runtime_info() -> TEIRerankerRuntimeInfo | None:
    return _runtime_info


def tei_reranker_runtime_status() -> dict:
    return asdict(_runtime_info) if _runtime_info else {
        "available": False,
        "managed": False,
        "model": "",
        "model_sha": "",
    }


def resolve_cached_model_revision(
    cache_root: str | Path,
    model_id: str,
    configured_revision: str = "",
) -> str:
    """Resolve the pinned Hugging Face revision without hashing model weights."""

    revision = configured_revision.strip()
    if revision and revision != "main":
        return revision
    if not cache_root:
        return ""
    ref_name = revision or "main"
    repository = "models--" + model_id.replace("/", "--")
    refs_root = (Path(cache_root).resolve() / repository / "refs").resolve()
    ref_path = (refs_root / ref_name).resolve()
    if refs_root not in ref_path.parents or not ref_path.is_file():
        return ""
    try:
        return ref_path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def validate_tei_reranker_info(
    payload: object,
    expected_model: str,
    fallback_model_sha: str = "",
) -> tuple[str, str]:
    if not isinstance(payload, dict):
        raise TypeError("TEI /info 返回了无效响应")
    model = str(payload.get("model_id") or "")
    if model != expected_model:
        raise RuntimeError(f"TEI 当前加载模型为 {model or 'unknown'}，期望 {expected_model}")
    model_type = payload.get("model_type")
    is_reranker = (
        model_type == "reranker"
        or model_type == 2
        or (isinstance(model_type, dict) and "reranker" in model_type)
    )
    if not is_reranker:
        raise RuntimeError("TEI 当前模型不是 Reranker")
    model_sha = str(payload.get("model_sha") or fallback_model_sha).strip()
    if not model_sha:
        raise RuntimeError(
            "TEI /info 未返回 model_sha，且未配置或缓存可验证的模型 revision"
        )
    return model, model_sha


class TEIRerankerProcessManager:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.base_url = settings.rag_rerank_base_url.rstrip("/")
        self.process: subprocess.Popen | None = None
        self._log_file = None
        self._owned_process = False

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.close()

    def _get_json(self, path: str, timeout: float = 2.0) -> object:
        response = httpx.get(f"{self.base_url}{path}", timeout=timeout)
        response.raise_for_status()
        return response.json()

    def _post_json(self, path: str, payload: dict) -> object:
        response = httpx.post(
            f"{self.base_url}{path}",
            json=payload,
            timeout=self.settings.rag_rerank_request_timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def _is_available(self) -> bool:
        try:
            response = httpx.get(f"{self.base_url}/health", timeout=2.0)
            response.raise_for_status()
            return True
        except httpx.HTTPError:
            return False

    def _local_host(self) -> tuple[str, int]:
        parsed = urlparse(self.base_url)
        host = parsed.hostname or ""
        if parsed.scheme != "http" or host not in {"127.0.0.1", "localhost", "::1"}:
            raise RuntimeError("RAG_RERANK_AUTO_START 仅支持本机 HTTP 回环地址")
        return host, parsed.port or 80

    def _child_environment(self) -> dict[str, str]:
        allowed = {
            "HOME",
            "LANG",
            "LC_ALL",
            "PATH",
            "SSL_CERT_DIR",
            "SSL_CERT_FILE",
            "TMPDIR",
        }
        return {key: value for key, value in os.environ.items() if key in allowed}

    def _command(self) -> list[str]:
        host, port = self._local_host()
        command = shutil.which(self.settings.rag_rerank_command)
        if command is None:
            raise RuntimeError(
                "未找到 TEI 命令 `text-embeddings-router`，"
                "请先执行：brew install text-embeddings-inference"
            )
        cache = Path(self.settings.rag_rerank_model_cache).resolve()
        cache.mkdir(parents=True, exist_ok=True)
        result = [
            command,
            "--model-id",
            self.settings.rag_rerank_model,
            "--port",
            str(port),
            "--hostname",
            host,
            "--huggingface-hub-cache",
            str(cache),
            "--max-client-batch-size",
            str(self.settings.rag_rerank_max_client_batch_size),
            "--max-batch-requests",
            str(self.settings.rag_rerank_max_batch_requests),
        ]
        revision = self.settings.rag_rerank_model_revision.strip()
        if revision:
            result.extend(["--revision", revision])
        return result

    def _start_process(self) -> None:
        command = self._command()
        log_path = Path(self.settings.rag_rerank_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = log_path.open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            command,
            env=self._child_environment(),
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._owned_process = True

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.settings.rag_rerank_startup_timeout_seconds
        while time.monotonic() < deadline:
            if self._is_available():
                if self.process is not None and self.process.poll() is not None:
                    self._owned_process = False
                return
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"TEI 启动进程提前退出，详情见 {self.settings.rag_rerank_log_path}"
                )
            time.sleep(0.5)
        raise RuntimeError(
            "等待 TEI Reranker 启动超过 "
            f"{self.settings.rag_rerank_startup_timeout_seconds:g} 秒"
        )

    def _model_info(self) -> tuple[str, str]:
        fallback = resolve_cached_model_revision(
            self.settings.rag_rerank_model_cache,
            self.settings.rag_rerank_model,
            self.settings.rag_rerank_model_revision,
        )
        return validate_tei_reranker_info(
            self._get_json("/info"),
            self.settings.rag_rerank_model,
            fallback,
        )

    def _warmup(self) -> None:
        from mcp_servers.law_rag.reranker import parse_tei_ranks

        payload = self._post_json("/rerank", {
            "query": "用人单位未签书面劳动合同",
            "texts": [
                "劳动合同法 第八十二条 未订立书面劳动合同应支付二倍工资",
                "海商法 船舶碰撞损害赔偿",
            ],
            "truncate": True,
            "raw_scores": False,
            "return_text": False,
        })
        ranks = parse_tei_ranks(payload, expected_count=2)
        by_index = {item["index"]: item["score"] for item in ranks}
        if by_index[0] <= by_index[1]:
            raise RuntimeError("TEI Reranker 预热相关性校验未通过")

    def ensure_ready(self) -> TEIRerankerRuntimeInfo:
        global _runtime_info

        try:
            if not self._is_available():
                if not self.settings.rag_rerank_auto_start:
                    raise RuntimeError(f"无法连接 TEI Reranker：{self.base_url}")
                lock_path = Path("data/runtime/tei-reranker-start.lock")
                lock_path.parent.mkdir(parents=True, exist_ok=True)
                with FileLock(
                    str(lock_path),
                    timeout=self.settings.rag_rerank_startup_timeout_seconds,
                ):
                    if not self._is_available():
                        self._start_process()
                        self._wait_until_ready()
            model, model_sha = self._model_info()
            self._warmup()
            _runtime_info = TEIRerankerRuntimeInfo(
                available=True,
                managed=self._owned_process,
                model=model,
                model_sha=model_sha,
            )
            return _runtime_info
        except Exception:
            if self._owned_process:
                self.close()
            raise

    def close(self) -> None:
        global _runtime_info

        process = self.process
        if self._owned_process and process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=self.settings.rag_rerank_shutdown_timeout_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            except ProcessLookupError:
                pass
        self._owned_process = False
        self.process = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        _runtime_info = None
