import json
from pathlib import Path

from langsmith import Client

from backend.app.core.config import get_settings

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    settings = get_settings()
    if not settings.langsmith_api_key:
        raise SystemExit("请先在 .env 配置 LANGSMITH_API_KEY")
    client = Client(
        api_url=settings.langsmith_endpoint,
        api_key=settings.langsmith_api_key,
        workspace_id=settings.langsmith_workspace_id or None,
    )
    for path in sorted((ROOT / "evals" / "datasets").glob("*.jsonl")):
        dataset_name = path.stem
        if client.has_dataset(dataset_name=dataset_name):
            print(f"跳过已存在数据集：{dataset_name}")
            continue
        client.create_dataset(
            dataset_name,
            description="LawStation 人工编写或合成的脱敏基准数据集",
            metadata={"version": "v1", "contains_production_data": False},
        )
        examples = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]
        client.create_examples(dataset_name=dataset_name, examples=examples)
        print(f"已创建 {dataset_name}：{len(examples)} 条")


if __name__ == "__main__":
    main()
