import subprocess
from types import SimpleNamespace

import pytest

import backend.app.core.tei as tei_module


def settings(tmp_path):
    return SimpleNamespace(
        rag_rerank_base_url="http://127.0.0.1:8081",
        rag_rerank_model="BAAI/bge-reranker-v2-m3",
        rag_rerank_model_revision="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        rag_rerank_auto_start=True,
        rag_rerank_command="text-embeddings-router",
        rag_rerank_model_cache=str(tmp_path / "models"),
        rag_rerank_startup_timeout_seconds=1,
        rag_rerank_shutdown_timeout_seconds=1,
        rag_rerank_request_timeout_seconds=2,
        rag_rerank_log_path=str(tmp_path / "logs" / "reranker.log"),
        rag_rerank_max_client_batch_size=16,
        rag_rerank_max_batch_requests=1,
    )


def test_existing_tei_is_reused_and_not_owned(tmp_path, monkeypatch):
    manager = tei_module.TEIRerankerProcessManager(settings(tmp_path))
    monkeypatch.setattr(manager, "_is_available", lambda: True)
    monkeypatch.setattr(
        manager,
        "_model_info",
        lambda: ("BAAI/bge-reranker-v2-m3", "model-sha"),
    )
    monkeypatch.setattr(manager, "_warmup", lambda: None)
    monkeypatch.setattr(manager, "_start_process", lambda: pytest.fail("不应启动 TEI"))
    runtime = manager.ensure_ready()
    assert runtime.available is True
    assert runtime.managed is False
    assert runtime.model_sha == "model-sha"
    manager.close()


def test_validate_info_requires_expected_reranker_and_resolvable_sha():
    payload = {
        "model_id": "BAAI/bge-reranker-v2-m3",
        "model_sha": "abc",
        "model_type": {"reranker": {"id2label": {"0": "LABEL_0"}}},
    }
    assert tei_module.validate_tei_reranker_info(
        payload, "BAAI/bge-reranker-v2-m3"
    ) == ("BAAI/bge-reranker-v2-m3", "abc")
    with pytest.raises(RuntimeError, match="期望"):
        tei_module.validate_tei_reranker_info(payload, "other")
    with pytest.raises(RuntimeError, match="不是 Reranker"):
        tei_module.validate_tei_reranker_info(
            {**payload, "model_type": {"embedding": {}}}, payload["model_id"]
        )
    assert tei_module.validate_tei_reranker_info(
        {**payload, "model_sha": None}, payload["model_id"], "cached-sha"
    ) == (payload["model_id"], "cached-sha")
    with pytest.raises(RuntimeError, match="模型 revision"):
        tei_module.validate_tei_reranker_info(
            {**payload, "model_sha": ""}, payload["model_id"]
        )


def test_cached_model_revision_uses_configured_pin_then_hub_ref(tmp_path):
    cache = tmp_path / "hub"
    ref = cache / "models--BAAI--bge-reranker-v2-m3" / "refs" / "main"
    ref.parent.mkdir(parents=True)
    ref.write_text("cached-sha\n", encoding="utf-8")
    assert tei_module.resolve_cached_model_revision(
        cache, "BAAI/bge-reranker-v2-m3"
    ) == "cached-sha"
    assert tei_module.resolve_cached_model_revision(
        cache, "BAAI/bge-reranker-v2-m3", "pinned-sha"
    ) == "pinned-sha"


def test_start_command_uses_model_cache_and_filters_secrets(tmp_path, monkeypatch):
    manager = tei_module.TEIRerankerProcessManager(settings(tmp_path))
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
    monkeypatch.setattr(
        tei_module.shutil,
        "which",
        lambda _command: "/opt/homebrew/bin/text-embeddings-router",
    )
    monkeypatch.setattr(tei_module.subprocess, "Popen", popen)
    manager._start_process()
    args = captured["args"]
    assert args[:3] == [
        "/opt/homebrew/bin/text-embeddings-router",
        "--model-id",
        "BAAI/bge-reranker-v2-m3",
    ]
    assert args[args.index("--port") + 1] == "8081"
    assert args[args.index("--hostname") + 1] == "127.0.0.1"
    assert args[args.index("--max-client-batch-size") + 1] == "16"
    assert args[args.index("--max-batch-requests") + 1] == "1"
    assert args[args.index("--revision") + 1] == settings(tmp_path).rag_rerank_model_revision
    assert "DEEPSEEK_API_KEY" not in captured["env"]
    assert "LANGSMITH_API_KEY" not in captured["env"]
    manager._owned_process = False
    manager.close()


def test_missing_command_has_install_instruction(tmp_path, monkeypatch):
    manager = tei_module.TEIRerankerProcessManager(settings(tmp_path))
    monkeypatch.setattr(tei_module.shutil, "which", lambda _command: None)
    with pytest.raises(RuntimeError, match="brew install text-embeddings-inference"):
        manager._start_process()


def test_warmup_requires_positive_score_above_negative(tmp_path, monkeypatch):
    manager = tei_module.TEIRerankerProcessManager(settings(tmp_path))
    captured = {}

    def post(path, payload):
        captured.update({"path": path, "payload": payload})
        return [{"index": 1, "score": 0.1}, {"index": 0, "score": 0.9}]

    monkeypatch.setattr(manager, "_post_json", post)
    manager._warmup()
    assert captured["path"] == "/rerank"
    assert captured["payload"]["raw_scores"] is False
    assert len(captured["payload"]["texts"]) == 2


def test_failed_owned_start_is_cleaned_up(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manager = tei_module.TEIRerankerProcessManager(settings(tmp_path))
    closed = []
    monkeypatch.setattr(manager, "_is_available", lambda: False)

    def start():
        manager._owned_process = True

    monkeypatch.setattr(manager, "_start_process", start)
    monkeypatch.setattr(
        manager, "_wait_until_ready", lambda: (_ for _ in ()).throw(RuntimeError("failed"))
    )
    monkeypatch.setattr(manager, "close", lambda: closed.append(True))
    with pytest.raises(RuntimeError, match="failed"):
        manager.ensure_ready()
    assert closed == [True]


def test_owned_process_is_killed_after_timeout(tmp_path, monkeypatch):
    manager = tei_module.TEIRerankerProcessManager(settings(tmp_path))
    signals = []

    class Process:
        pid = 456

        def poll(self):
            return None

        def wait(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired("tei", timeout)
            return 0

    manager.process = Process()
    manager._owned_process = True
    monkeypatch.setattr(
        tei_module.os, "killpg", lambda pid, sig: signals.append((pid, sig))
    )
    manager.close()
    assert signals == [
        (456, tei_module.signal.SIGTERM),
        (456, tei_module.signal.SIGKILL),
    ]
