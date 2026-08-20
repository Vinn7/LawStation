from types import SimpleNamespace

import numpy as np
import pytest

import mcp_servers.law_rag.engine as engine_module


def settings(source, index_dir):
    return SimpleNamespace(
        law_data_path=str(source), index_dir=str(index_dir),
        index_chunk_max_chars=1000, index_chunk_overlap_chars=150,
        embedding_model="test-embedding", embedding_dimension=4,
        index_auto_build=True, dashscope_api_key="test", dashscope_base_url="http://unused",
        index_build_batch_size=2,
    )


@pytest.mark.asyncio
async def test_valid_manifest_skips_embedding(tmp_path, monkeypatch):
    source = tmp_path / "law.json"
    source.write_text('{"民法典第一条":"第一条内容","刑法第二条":"第二条内容"}', encoding="utf-8")
    config = settings(source, tmp_path / "indexes")
    monkeypatch.setattr(engine_module, "get_settings", lambda: config)
    monkeypatch.setattr(engine_module, "audit", lambda *args, **kwargs: None)
    first = engine_module.LawSearchEngine()
    first._record_manifest = lambda: None
    calls = 0

    async def embed(texts):
        nonlocal calls
        calls += 1
        return np.ones((len(texts), 4), dtype="float32")

    first._embed = embed
    await first.initialize_index(wait=True)
    assert first.status()["status"] == "ready"
    assert calls == 1

    second = engine_module.LawSearchEngine()
    async def should_not_run(_):
        raise AssertionError("有效索引不应再次调用 Embedding")
    second._embed = should_not_run
    await second.initialize_index(wait=True)
    assert second.status()["status"] == "ready"

    forced = engine_module.LawSearchEngine()
    forced._record_manifest = lambda: None
    forced_calls = 0
    async def forced_embed(texts):
        nonlocal forced_calls
        forced_calls += 1
        return np.ones((len(texts), 4), dtype="float32")
    forced._embed = forced_embed
    await forced.initialize_index(wait=True, force=True)
    assert forced_calls == 1


def test_long_article_splits_with_stable_ids(tmp_path, monkeypatch):
    source = tmp_path / "law.json"
    source.write_text('{"测试法第一条":"' + "甲" * 1200 + '"}', encoding="utf-8")
    monkeypatch.setattr(engine_module, "get_settings", lambda: settings(source, tmp_path / "indexes"))
    first = engine_module.LawSearchEngine()
    second = engine_module.LawSearchEngine()
    assert len(first.docs) == 2
    assert [item["chunk_id"] for item in first.docs] == [item["chunk_id"] for item in second.docs]
