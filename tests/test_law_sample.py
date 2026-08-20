import hashlib
import json
from pathlib import Path

from backend.app.core.config import get_settings
from scripts.create_law_sample import create_sample


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_sample_is_reproducible_and_matches_source(tmp_path):
    source = tmp_path / "law.json"
    source.write_text(json.dumps({f"法第{i}条": f"内容{i}" for i in range(200)}, ensure_ascii=False), encoding="utf-8")
    source_before = digest(source)
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    create_sample(source, first, size=100, seed=42)
    create_sample(source, second, size=100, seed=42)
    sampled = json.loads(first.read_text(encoding="utf-8"))
    original = json.loads(source.read_text(encoding="utf-8"))
    assert len(sampled) == 100
    assert len(set(sampled)) == 100
    assert all(original[key] == value for key, value in sampled.items())
    assert digest(first) == digest(second)
    assert digest(source) == source_before


def test_different_seed_changes_sample(tmp_path):
    source = tmp_path / "law.json"
    source.write_text(json.dumps({str(i): str(i) for i in range(200)}), encoding="utf-8")
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    create_sample(source, first, 100, 1)
    create_sample(source, second, 100, 2)
    assert digest(first) != digest(second)


def test_default_source_is_sample():
    assert get_settings().law_data_path.endswith("/law_sample.json")
