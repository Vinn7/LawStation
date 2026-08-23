"""Create a reproducible source-derived retrieval set with real law/chunk IDs."""

import json
import random
import re
from collections import defaultdict
from pathlib import Path

from backend.app.core.config import get_settings
from mcp_servers.law_rag.engine import load_chunks

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "evals" / "datasets" / "lawstation-live-retrieval-v1.jsonl"
NO_MATCH_QUERIES = [
    "月球背面虚拟土地继承规则",
    "时间旅行后未来债务的清偿顺序",
    "纯人工智能能否登记为自然人配偶",
    "游戏公会称号的法定继承份额",
    "外星采矿许可证由哪个地球法院签发",
    "梦境中签订合同的强制执行程序",
    "机器人之间情感承诺的结婚登记手续",
    "平行宇宙公司的跨宇宙破产管辖",
    "数字宠物自行订立遗嘱的见证要求",
    "瞬间传送导致两个本人时的身份证归属",
    "虚拟人格担任国家机关负责人的任职条件",
    "火星殖民地宅基地使用权登记办法",
    "脑机接口自动生成想法的法定登记机关",
    "倒流时间后撤销昨日判决的法定期限",
    "克隆人的原始记忆是否属于不动产",
    "元宇宙天气控制权的行政许可程序",
    "外星语言口头遗嘱的强制公证规则",
    "量子分身同时犯罪时的刑事责任分配",
    "虚拟世界复活角色后的继承恢复程序",
    "跨越一百年时间旅行的诉讼时效暂停规则",
]


def _example(question: str, expected: dict | None = None) -> dict:
    return {
        "inputs": {"question": question, "top_k": 5, "filters": None},
        "outputs": {
            "expected_route": None,
            "expected_retrieval_status": "matched" if expected else "no_match",
            "expected_document_ids": [expected["document_id"]] if expected else [],
            "expected_chunk_ids": [expected["chunk_id"]] if expected else [],
            "expected_law_name": expected["law_name"] if expected else "",
            "expected_article_number": expected["article_number"] if expected else "",
        },
        "metadata": {
            "category": "matched" if expected else "no_match",
            "source_derived": bool(expected),
            "human_verified": False,
        },
    }


def build(size: int = 100, seed: int = 42) -> list[dict]:
    if size < len(NO_MATCH_QUERIES):
        raise ValueError(f"size 不能小于 {len(NO_MATCH_QUERIES)}")
    settings = get_settings()
    chunks = load_chunks(
        Path(settings.law_data_path),
        settings.index_chunk_max_chars,
        settings.index_chunk_overlap_chars,
    )
    by_document: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        by_document[chunk["document_id"]].append(chunk)
    single_chunks = [items[0] for items in by_document.values() if len(items) == 1]
    by_law: dict[str, list[dict]] = defaultdict(list)
    for chunk in single_chunks:
        if chunk["law_name"] and chunk["article_number"]:
            by_law[chunk["law_name"]].append(chunk)
    rng = random.Random(seed)
    law_names = sorted(by_law)
    rng.shuffle(law_names)
    matched_count = size - len(NO_MATCH_QUERIES)
    selected: list[dict] = []
    for law_name in law_names:
        candidates = by_law[law_name]
        selected.append(candidates[rng.randrange(len(candidates))])
        if len(selected) == matched_count:
            break
    if len(selected) < matched_count:
        remaining = [item for item in single_chunks if item not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: matched_count - len(selected)])
    if len(selected) != matched_count:
        raise RuntimeError("法规数据不足，无法生成指定数量的检索样本")
    examples = []
    for item in selected:
        excerpt = re.sub(r"\s+", "", item["content"])[:28]
        question = (
            f"请检索《{item['law_name']}》{item['article_number']}中关于“{excerpt}”的规定"
        )
        examples.append(_example(question, item))
    examples.extend(_example(question) for question in NO_MATCH_QUERIES)
    return examples


def main() -> None:
    examples = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in examples) + "\n",
        encoding="utf-8",
    )
    print(f"已生成 {OUTPUT.name}：{len(examples)} 条（源数据派生，未做人工法律标注）")


if __name__ == "__main__":
    main()
