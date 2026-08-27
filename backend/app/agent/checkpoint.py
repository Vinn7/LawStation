from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from backend.app.core.config import Settings


@asynccontextmanager
async def checkpoint_saver(settings: Settings):
    """Create one async saver for the application lifetime."""
    if not settings.langgraph_checkpoint_enabled:
        yield None
        return
    path = Path(settings.langgraph_checkpoint_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if settings.langgraph_strict_msgpack:
        os.environ["LANGGRAPH_STRICT_MSGPACK"] = "true"
    connection = await aiosqlite.connect(path)
    saver = AsyncSqliteSaver(
        connection,
        serde=JsonPlusSerializer(pickle_fallback=False),
    )
    await saver.setup()
    try:
        yield saver
    finally:
        await connection.close()
