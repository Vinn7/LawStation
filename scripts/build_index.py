import argparse
import asyncio

from mcp_servers.law_rag.engine import LawSearchEngine


async def build(force: bool) -> None:
    engine = LawSearchEngine()
    await engine.initialize_index(wait=True, force=force)
    status = engine.status()
    print(f"索引状态：{status['status']}，进度：{status['progress']:.0%}，{status['message']}")
    if status["status"] not in {"ready", "degraded"}:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 LawStation 法律索引")
    parser.add_argument("--force", action="store_true", help="忽略有效 manifest 并强制重建")
    args = parser.parse_args()
    asyncio.run(build(args.force))


if __name__ == "__main__":
    main()
