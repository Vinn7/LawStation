import math
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
class OllamaRuntimeInfo:
    available: bool
    managed: bool
    model: str
    digest: str
    reranker_available: bool = False
    reranker_model: str = ""
    reranker_digest: str = ""
    reranker_error: str = ""


_runtime_info: OllamaRuntimeInfo | None = None


def get_ollama_runtime_info() -> OllamaRuntimeInfo | None:
    return _runtime_info


def ollama_runtime_status() -> dict:
    return asdict(_runtime_info) if _runtime_info else {
        "available": False,
        "managed": False,
        "model": "",
        "digest": "",
        "reranker_available": False,
        "reranker_model": "",
        "reranker_digest": "",
        "reranker_error": "",
    }


class OllamaProcessManager:
    def __init__(self, settings):
        self.settings = settings
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.process: subprocess.Popen | None = None
        self._log_file = None
        self._owned_process = False

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.close()

    def _get_json(self, path: str, timeout: float = 2.0) -> dict:
        response = httpx.get(f"{self.base_url}{path}", timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise TypeError(f"Ollama {path} 返回了无效响应")
        return payload

    def _post_json(self, path: str, payload: dict) -> dict:
        response = httpx.post(
            f"{self.base_url}{path}",
            json=payload,
            timeout=self.settings.ollama_request_timeout_seconds,
        )
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict):
            raise TypeError(f"Ollama {path} 返回了无效响应")
        return result

    def _is_available(self) -> bool:
        try:
            self._get_json("/api/version")
            return True
        except (httpx.HTTPError, RuntimeError, TypeError, ValueError):
            return False

    def _local_host(self) -> tuple[str, int]:
        parsed = urlparse(self.base_url)
        host = parsed.hostname or ""
        if parsed.scheme not in {"http", "https"} or host not in {"127.0.0.1", "localhost", "::1"}:
            raise RuntimeError("OLLAMA_AUTO_START 仅支持本机回环地址")
        return host, parsed.port or (443 if parsed.scheme == "https" else 80)

    def _child_environment(self, host: str, port: int) -> dict[str, str]:
        allowed = {
            "HOME",
            "LANG",
            "LC_ALL",
            "PATH",
            "SSL_CERT_DIR",
            "SSL_CERT_FILE",
            "TMPDIR",
        }
        environment = {key: value for key, value in os.environ.items() if key in allowed}
        listen_host = f"[{host}]" if ":" in host else host
        environment.update({
            "OLLAMA_HOST": f"{listen_host}:{port}",
            "OLLAMA_KEEP_ALIVE": self.settings.ollama_keep_alive,
            "OLLAMA_MAX_LOADED_MODELS": str(
                getattr(self.settings, "ollama_max_loaded_models", 2)
            ),
            "OLLAMA_NO_CLOUD": "1",
        })
        return environment

    def _start_process(self) -> None:
        host, port = self._local_host()
        command = shutil.which(self.settings.ollama_command)
        if command is None:
            raise RuntimeError(
                f"未找到 Ollama 命令 `{self.settings.ollama_command}`，请先安装 Ollama"
            )
        log_path = Path(self.settings.ollama_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = log_path.open("a", encoding="utf-8")
        self.process = subprocess.Popen(
            [command, "serve"],
            env=self._child_environment(host, port),
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._owned_process = True

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.settings.ollama_startup_timeout_seconds
        while time.monotonic() < deadline:
            if self._is_available():
                if self.process is not None and self.process.poll() is not None:
                    self._owned_process = False
                return
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"Ollama 启动进程提前退出，详情见 {self.settings.ollama_log_path}"
                )
            time.sleep(0.5)
        raise RuntimeError(
            f"等待 Ollama 启动超过 {self.settings.ollama_startup_timeout_seconds:g} 秒"
        )

    def _model_info(self, model_name: str | None = None) -> tuple[str, str]:
        expected_model = model_name or self.settings.embedding_model
        models = self._get_json("/api/tags").get("models")
        if not isinstance(models, list):
            raise TypeError("Ollama /api/tags 未返回模型列表")
        for item in models:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("model") or "")
            if name == expected_model:
                return name, str(item.get("digest") or "")
        raise RuntimeError(
            f"未找到 {expected_model}，请先执行："
            f"ollama pull {expected_model}"
        )

    def _warmup(self) -> None:
        result = self._post_json("/api/embed", {
            "model": self.settings.embedding_model,
            "input": "法律检索",
            "dimensions": self.settings.embedding_dimension,
            "truncate": False,
            "keep_alive": self.settings.ollama_keep_alive,
        })
        embeddings = result.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != 1:
            raise RuntimeError("Ollama 模型预热未返回单个向量")
        vector = embeddings[0]
        if not isinstance(vector, list) or len(vector) != self.settings.embedding_dimension:
            raise RuntimeError("Ollama 模型预热返回了错误的向量维度")
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in vector):
            raise RuntimeError("Ollama 模型预热返回了无效向量")

    def _warmup_reranker(self) -> None:
        from mcp_servers.law_rag.reranker import (
            build_rerank_prompt,
            extract_yes_no_score,
        )

        def score(document: str) -> float:
            result = self._post_json("/api/generate", {
                "model": self.settings.rag_rerank_model,
                "prompt": build_rerank_prompt("用人单位未签书面劳动合同", document),
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
            })
            return extract_yes_no_score(result)

        positive = score("劳动合同法 第八十二条 未订立书面劳动合同应支付二倍工资")
        negative = score("海商法 船舶碰撞损害赔偿")
        if not positive > negative:
            raise RuntimeError("Ollama Reranker 预热相关性校验未通过")

    def ensure_ready(self) -> OllamaRuntimeInfo:
        global _runtime_info

        if self.settings.embedding_provider != "ollama":
            raise RuntimeError("OllamaProcessManager 只能用于 EMBEDDING_PROVIDER=ollama")
        if not self._is_available():
            if not self.settings.ollama_auto_start:
                raise RuntimeError(f"无法连接 Ollama：{self.base_url}")
            lock_path = Path("data/runtime/ollama-start.lock")
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with FileLock(str(lock_path), timeout=self.settings.ollama_startup_timeout_seconds):
                if not self._is_available():
                    self._start_process()
                    self._wait_until_ready()
        model, digest = self._model_info()
        if not digest:
            raise RuntimeError(f"Ollama 未返回模型 {model} 的 digest")
        self._warmup()
        reranker_available = False
        reranker_model = ""
        reranker_digest = ""
        reranker_error = ""
        if (
            getattr(self.settings, "rag_rerank_enabled", False)
            and getattr(self.settings, "rag_rerank_provider", "ollama") == "ollama"
        ):
            try:
                reranker_model, reranker_digest = self._model_info(
                    self.settings.rag_rerank_model
                )
                if not reranker_digest:
                    raise RuntimeError(
                        f"Ollama 未返回模型 {reranker_model} 的 digest"
                    )
                self._warmup_reranker()
                reranker_available = True
            except Exception as exc:
                reranker_error = str(exc)
                if getattr(self.settings, "rag_rerank_required", False):
                    raise RuntimeError(f"Ollama Reranker 启动检查失败：{exc}") from exc
        _runtime_info = OllamaRuntimeInfo(
            available=True,
            managed=self._owned_process,
            model=model,
            digest=digest,
            reranker_available=reranker_available,
            reranker_model=reranker_model or getattr(
                self.settings, "rag_rerank_model", ""
            ),
            reranker_digest=reranker_digest,
            reranker_error=reranker_error,
        )
        return _runtime_info

    def close(self) -> None:
        global _runtime_info

        process = self.process
        if self._owned_process and process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=self.settings.ollama_shutdown_timeout_seconds)
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
