import asyncio

import pytest

from backend.app.agent.concurrency import AgentConcurrencyManager, ConcurrencyIdentity
from backend.app.api.routes import with_sse_heartbeat
from backend.app.core.config import Settings
from backend.app.db.session import engine


def config(**changes):
    values = {
        "agent_global_concurrency": 6,
        "agent_per_user_concurrency": 2,
        "agent_per_conversation_concurrency": 1,
        "agent_queue_timeout_seconds": 1,
    }
    values.update(changes)
    return Settings(_env_file=None, **values)


@pytest.mark.asyncio
async def test_global_six_and_per_user_two_admission_limits():
    manager = AgentConcurrencyManager(config())
    active = [
        ConcurrencyIdentity(f"r-{index}", "tenant", f"user-{index // 2}", f"c-{index}")
        for index in range(6)
    ]
    for identity in active:
        await manager.reserve(identity)
        await manager.acquire(identity)

    seventh = ConcurrencyIdentity("r-7", "tenant", "user-new", "c-7")
    same_user_third = ConcurrencyIdentity("r-8", "tenant", "user-0", "c-8")
    assert await manager.would_queue(seventh)
    assert await manager.would_queue(same_user_third)

    await manager.release(active[2])
    assert not await manager.would_queue(seventh)
    assert await manager.would_queue(same_user_third)

    for identity in active[1:]:
        await manager.release(identity)
    for identity in active:
        await manager.release_reservation(identity)


def test_sqlite_uses_wal_foreign_keys_and_busy_timeout():
    if engine.dialect.name != "sqlite":
        pytest.skip("SQLite-specific concurrency settings")
    with engine.connect() as connection:
        journal_mode = connection.exec_driver_sql("PRAGMA journal_mode").scalar_one()
        foreign_keys = connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one()
        busy_timeout = connection.exec_driver_sql("PRAGMA busy_timeout").scalar_one()

    assert journal_mode.lower() == "wal"
    assert foreign_keys == 1
    assert busy_timeout == 5000


@pytest.mark.asyncio
async def test_sse_heartbeat_is_emitted_while_agent_is_silent():
    async def slow_source():
        await asyncio.sleep(0.03)
        yield 'event: token\ndata: "完成"\n\n'

    chunks = [chunk async for chunk in with_sse_heartbeat(slow_source(), 0.005)]

    assert ": heartbeat\n\n" in chunks
    assert chunks[-1] == 'event: token\ndata: "完成"\n\n'
