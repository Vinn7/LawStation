import argparse
import json
import os
import random
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "knowledge" / "law" / "law.json"
DEFAULT_OUTPUT = ROOT / "data" / "knowledge" / "law" / "law_sample.json"


def create_sample(source: Path, output: Path, size: int = 100, seed: int = 42) -> dict[str, str]:
    records = json.loads(source.read_text(encoding="utf-8"))
    items = list(records.items())
    if size < 1 or size > len(items):
        raise ValueError(f"size 必须在 1 到 {len(items)} 之间")
    indices = sorted(random.Random(seed).sample(range(len(items)), size))
    sample = dict(items[index] for index in indices)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent, delete=False) as tmp:
        json.dump(sample, tmp, ensure_ascii=False, indent=2)
        tmp.write("\n")
        temporary = Path(tmp.name)
    os.replace(temporary, output)
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(description="生成可复现的法规样本")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    sample = create_sample(args.source, args.output, args.size, args.seed)
    print(f"已生成 {len(sample)} 条法规样本：{args.output}")


if __name__ == "__main__":
    main()
