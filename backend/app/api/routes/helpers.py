import asyncio
import json


def obj(row):
    return {column.name: getattr(row, column.name) for column in row.__table__.columns}


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def persisted_sse(sequence: int, event: str, data) -> str:
    return (
        f"id: {sequence}\nevent: {event}\n"
        f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    )


async def with_sse_heartbeat(source, interval_seconds: float):
    """Keep an SSE connection alive without turning heartbeats into app events."""
    iterator = source.__aiter__()
    pending: asyncio.Task | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.create_task(anext(iterator))
            done, _ = await asyncio.wait(
                {pending}, timeout=max(0.01, interval_seconds)
            )
            if not done:
                yield ": heartbeat\n\n"
                continue
            try:
                item = pending.result()
            except StopAsyncIteration:
                break
            pending = None
            yield item
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        close = getattr(iterator, "aclose", None)
        if close is not None:
            await close()
