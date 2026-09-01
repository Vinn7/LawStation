"""LangGraph Checkpoint 的独立 SQLite 生命周期管理。"""

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
    """为应用生命周期创建一个 AsyncSqliteSaver。

    Checkpoint 数据库与 SQLAlchemy 业务库分离，避免每个 Graph super-step 的写入
    与消息/记忆事务争锁。它保存一次 Run 的节点 State，不承担跨轮长期记忆。
    """
    if not settings.langgraph_checkpoint_enabled:
        yield None
        return
    path = Path(settings.langgraph_checkpoint_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if settings.langgraph_strict_msgpack:
        os.environ["LANGGRAPH_STRICT_MSGPACK"] = "true"
    # AsyncSqliteSaver 与 graph.astream 同为异步接口，避免在事件循环中执行同步 I/O。
    connection = await aiosqlite.connect(path)
    # 禁用 pickle fallback，只允许 JsonPlus/msgpack 可控类型，避免从 Checkpoint
    # 反序列化任意 Python 对象。
    saver = AsyncSqliteSaver(
        connection,
        serde=JsonPlusSerializer(pickle_fallback=False),
    )
    # setup 只创建 Checkpoint 自身表结构；不会编译 Graph 或运行任何 Agent 节点。
    await saver.setup()
    try:
        yield saver
    finally:
        await connection.close()
