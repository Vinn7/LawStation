import subprocess
from types import SimpleNamespace

import pytest

import backend.app.core.ollama as ollama_module


def settings(tmp_path):
    return SimpleNamespace(
        embedding_provider="ollama",
        embedding_model="qwen3-embedding:0.6b",
        embedding_dimension=4,
        ollama_auto_start=True,
        ollama_command="ollama",
        ollama_base_url="http://127.0.0.1:11434",
        ollama_startup_timeout_seconds=1,
        ollama_shutdown_timeout_seconds=1,
        ollama_request_timeout_seconds=2,
        ollama_keep_alive="30m",
        ollama_max_loaded_models=2,
        ollama_log_path=str(tmp_path / "logs" / "ollama.log"),
        rag_rerank_enabled=False,
        rag_rerank_provider="ollama",
    )


def test_existing_ollama_is_reused_and_not_owned(tmp_path, monkeypatch):
    manager = ollama_module.OllamaProcessManager(settings(tmp_path))
    monkeypatch.setattr(manager, "_is_available", lambda: True)
    monkeypatch.setattr(manager, "_model_info", lambda: ("qwen3-embedding:0.6b", "digest-1"))
    monkeypatch.setattr(manager, "_warmup", lambda: None)
    monkeypatch.setattr(manager, "_start_process", lambda: pytest.fail("不应启动 Ollama"))
    runtime = manager.ensure_ready()
    assert runtime.available is True
    assert runtime.managed is False
    manager.close()


def test_missing_model_returns_pull_instruction(tmp_path, monkeypatch):
    manager = ollama_module.OllamaProcessManager(settings(tmp_path))
    monkeypatch.setattr(manager, "_get_json", lambda _path: {"models": []})
    with pytest.raises(RuntimeError, match="ollama pull qwen3-embedding:0.6b"):
        manager._model_info()


def test_auto_start_disabled_fails_without_starting_process(tmp_path, monkeypatch):
    config = settings(tmp_path)
    config.ollama_auto_start = False
    manager = ollama_module.OllamaProcessManager(config)
    monkeypatch.setattr(manager, "_is_available", lambda: False)
    monkeypatch.setattr(manager, "_start_process", lambda: pytest.fail("不应启动 Ollama"))
    with pytest.raises(RuntimeError, match="无法连接 Ollama"):
        manager.ensure_ready()


def test_missing_command_fails_with_install_instruction(tmp_path, monkeypatch):
    manager = ollama_module.OllamaProcessManager(settings(tmp_path))
    monkeypatch.setattr(ollama_module.shutil, "which", lambda _command: None)
    with pytest.raises(RuntimeError, match="请先安装 Ollama"):
        manager._start_process()


def test_warmup_validates_payload_and_dimension(tmp_path, monkeypatch):
    manager = ollama_module.OllamaProcessManager(settings(tmp_path))
    captured = {}

    def post(path, payload):
        captured.update({"path": path, "payload": payload})
        return {"embeddings": [[1, 2, 3, 4]]}

    monkeypatch.setattr(manager, "_post_json", post)
    manager._warmup()
    assert captured["path"] == "/api/embed"
    assert captured["payload"]["dimensions"] == 4
    assert captured["payload"]["truncate"] is False
    assert captured["payload"]["keep_alive"] == "30m"


def test_start_uses_serve_and_filters_secrets(tmp_path, monkeypatch):
    manager = ollama_module.OllamaProcessManager(settings(tmp_path))
    captured = {}

    class Process:
        pid = 123

        def poll(self):
            return None

    def popen(args, **kwargs):
        captured.update({"args": args, **kwargs})
        return Process()

    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("LANGSMITH_API_KEY", "secret")
    monkeypatch.setattr(ollama_module.shutil, "which", lambda _command: "/usr/local/bin/ollama")
    monkeypatch.setattr(ollama_module.subprocess, "Popen", popen)
    manager._start_process()
    assert captured["args"] == ["/usr/local/bin/ollama", "serve"]
    assert "DEEPSEEK_API_KEY" not in captured["env"]
    assert "LANGSMITH_API_KEY" not in captured["env"]
    assert captured["env"]["OLLAMA_NO_CLOUD"] == "1"
    assert captured["env"]["OLLAMA_MAX_LOADED_MODELS"] == "2"
    manager._owned_process = False
    manager.close()


def test_owned_process_is_terminated_on_close(tmp_path, monkeypatch):
    manager = ollama_module.OllamaProcessManager(settings(tmp_path))
    signals = []

    class Process:
        pid = 456

        def poll(self):
            return None

        def wait(self, timeout=None):
            if timeout is not None:
                return 0
            raise subprocess.TimeoutExpired("ollama", timeout)

    manager.process = Process()
    manager._owned_process = True
    monkeypatch.setattr(ollama_module.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    manager.close()
    assert signals == [(456, ollama_module.signal.SIGTERM)]


def test_owned_process_is_killed_after_shutdown_timeout(tmp_path, monkeypatch):
    manager = ollama_module.OllamaProcessManager(settings(tmp_path))
    signals = []

    class Process:
        pid = 789

        def poll(self):
            return None

        def wait(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired("ollama", timeout)
            return 0

    manager.process = Process()
    manager._owned_process = True
    monkeypatch.setattr(ollama_module.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    manager.close()
    assert signals == [
        (789, ollama_module.signal.SIGTERM),
        (789, ollama_module.signal.SIGKILL),
    ]


def test_optional_reranker_is_warmed_in_same_ollama_service(tmp_path, monkeypatch):
    config = settings(tmp_path)
    config.rag_rerank_enabled = True
    config.rag_rerank_required = False
    config.rag_rerank_model = "dengcao/Qwen3-Reranker-0.6B:latest"
    manager = ollama_module.OllamaProcessManager(config)
    monkeypatch.setattr(manager, "_is_available", lambda: True)

    def model_info(model_name=None):
        if model_name:
            return model_name, "reranker-digest"
        return config.embedding_model, "embedding-digest"

    monkeypatch.setattr(manager, "_model_info", model_info)
    monkeypatch.setattr(manager, "_warmup", lambda: None)
    warmed = []
    monkeypatch.setattr(manager, "_warmup_reranker", lambda: warmed.append(True))
    runtime = manager.ensure_ready()
    assert warmed == [True]
    assert runtime.reranker_available is True
    assert runtime.reranker_digest == "reranker-digest"
    manager.close()


def test_optional_reranker_failure_does_not_block_embedding_startup(tmp_path, monkeypatch):
    config = settings(tmp_path)
    config.rag_rerank_enabled = True
    config.rag_rerank_required = False
    config.rag_rerank_model = "missing-reranker"
    manager = ollama_module.OllamaProcessManager(config)
    monkeypatch.setattr(manager, "_is_available", lambda: True)

    def model_info(model_name=None):
        if model_name:
            raise RuntimeError("missing")
        return config.embedding_model, "embedding-digest"

    monkeypatch.setattr(manager, "_model_info", model_info)
    monkeypatch.setattr(manager, "_warmup", lambda: None)
    runtime = manager.ensure_ready()
    assert runtime.available is True
    assert runtime.reranker_available is False
    assert runtime.reranker_error == "missing"
    manager.close()


def test_required_reranker_failure_blocks_startup(tmp_path, monkeypatch):
    config = settings(tmp_path)
    config.rag_rerank_enabled = True
    config.rag_rerank_required = True
    config.rag_rerank_model = "missing-reranker"
    manager = ollama_module.OllamaProcessManager(config)
    monkeypatch.setattr(manager, "_is_available", lambda: True)

    def model_info(model_name=None):
        if model_name:
            raise RuntimeError("missing")
        return config.embedding_model, "embedding-digest"

    monkeypatch.setattr(manager, "_model_info", model_info)
    monkeypatch.setattr(manager, "_warmup", lambda: None)
    with pytest.raises(RuntimeError, match="Reranker 启动检查失败"):
        manager.ensure_ready()
    manager.close()


def test_tei_reranker_is_not_checked_by_ollama_manager(tmp_path, monkeypatch):
    config = settings(tmp_path)
    config.rag_rerank_enabled = True
    config.rag_rerank_provider = "tei"
    config.rag_rerank_required = True
    manager = ollama_module.OllamaProcessManager(config)
    monkeypatch.setattr(manager, "_is_available", lambda: True)
    monkeypatch.setattr(
        manager,
        "_model_info",
        lambda model_name=None: (
            pytest.fail("不应检查 TEI 模型")
            if model_name else (config.embedding_model, "embedding-digest")
        ),
    )
    monkeypatch.setattr(manager, "_warmup", lambda: None)
    monkeypatch.setattr(
        manager, "_warmup_reranker", lambda: pytest.fail("不应预热 TEI 模型")
    )
    runtime = manager.ensure_ready()
    assert runtime.reranker_available is False
    manager.close()
