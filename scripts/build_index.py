import argparse
import asyncio

from backend.app.core.config import get_settings
from backend.app.core.ollama import OllamaProcessManager
from mcp_servers.law_rag.engine import LawSearchEngine


async def build(force: bool) -> None:
    engine = LawSearchEngine()
    try:
        await engine.initialize_index(wait=True, force=force)
        status = engine.status()
        print(f"索引状态：{status['status']}，进度：{status['progress']:.0%}，{status['message']}")
        if status["status"] not in {"ready", "degraded"}:
            raise SystemExit(1)
    finally:
        await engine.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 LawStation 法律索引")
    parser.add_argument("--force", action="store_true", help="忽略有效 manifest 并强制重建")
    args = parser.parse_args()
    settings = get_settings()
    ollama = OllamaProcessManager(settings) if settings.embedding_provider == "ollama" else None
    try:
        if ollama is not None:
            ollama.ensure_ready()
        asyncio.run(build(args.force))
    finally:
        if ollama is not None:
            ollama.close()


if __name__ == "__main__":
    main()
