import asyncio

from mcp.server.fastmcp import FastMCP

from backend.app.core.config import get_settings
from mcp_servers.law_rag.engine import LawSearchEngine

settings = get_settings()
mcp = FastMCP(
    "LawStation Law RAG",
    host=settings.mcp_debug_host,
    port=settings.mcp_debug_port,
    streamable_http_path="/",
    json_response=True,
)
_engine: LawSearchEngine | None = None
_engine_lock = asyncio.Lock()


async def initialize_engine() -> LawSearchEngine:
    global _engine
    if _engine is None:
        async with _engine_lock:
            if _engine is None:
                _engine = await asyncio.to_thread(LawSearchEngine)
                await _engine.initialize_index()
    return _engine


def get_index_status() -> dict:
    if _engine is None:
        return {"status": "checking", "dense_enabled": False, "message": "检索引擎尚未初始化"}
    return _engine.status()


async def close_engine() -> None:
    if _engine is not None:
        await _engine.close()


@mcp.tool()
async def search_laws(query: str, top_k: int = 8, filters: dict | None = None) -> list[dict]:
    """混合检索中国法律法规。法律问题、权利义务或法条核验时使用。"""
    return await (await initialize_engine()).search(query, top_k, filters)


@mcp.tool()
async def get_law_article(law_name: str, article_number: str) -> dict:
    """按法律名称与条号精确查找一条法条。"""
    engine = await initialize_engine()
    return await asyncio.to_thread(engine.get, law_name, article_number) or {
        "error": "未找到指定法条"
    }


mcp_app = mcp.streamable_http_app()

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
