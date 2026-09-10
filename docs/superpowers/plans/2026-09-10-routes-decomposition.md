# routes.py Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `backend/app/api/routes.py` (756 lines) into a `backend/app/api/routes/` package of 8 domain-scoped files, with zero runtime behavior change and the same external interface (`backend.app.api.routes.router`, plus a handful of symbols tests import directly).

**Architecture:** Each business-domain group of endpoints becomes its own file with its own `APIRouter(prefix="/api")`. A package `__init__.py` facade combines them via `include_router` and re-exports the few symbols that tests import directly from the top-level module. This mirrors the already-completed `graph.py` → `graph/` and `memory_tasks.py` → `memory_tasks/` decompositions in this same initiative.

**Tech Stack:** FastAPI `APIRouter`, pytest, SQLAlchemy. Python executable: `/opt/anaconda3/envs/LawStation/bin/python`.

**Spec:** `docs/superpowers/specs/2026-09-10-routes-decomposition-design.md`

## Global Constraints

- Byte-for-byte extraction: every function/constant body must be copied verbatim from `backend/app/api/routes.py` — no behavior changes, no "while I'm here" cleanups.
- Zero curly quotes in the source file (confirmed via script) — extraction must not introduce or lose any Unicode punctuation; verify visually if a diff looks suspicious.
- **CPython flat-module-shadows-package import precedence** (reused finding from the `graph.py` and `memory_tasks.py` plans, not re-verified here): as long as `backend/app/api/routes.py` (the flat file) exists, `import backend.app.api.routes` resolves to that flat file, NOT the new `backend/app/api/routes/` directory sitting alongside it — even if that directory already contains files. This means Tasks 1-8 (which populate `backend/app/api/routes/*.py`) are inert from the interpreter's point of view until Task 9 deletes the flat file. Do not attempt to import the new submodules via dotted path (`backend.app.api.routes.helpers`, etc.) before Task 9 — it will silently resolve to attribute lookup on the flat module and fail. Verify new files with `python -m py_compile <file>` (compiles the file directly, no import machinery, unaffected by the shadowing) until Task 9 activates the package.
- Baseline: full test suite currently passes at **176 passed** (verified before this plan was written). This must hold after every task that touches test files, and especially after Task 9 (the activation task) and Task 10 (docs, no test impact expected).
- `SessionLocal`/`audit`-style monkeypatch problem: `tests/test_scenario_cleanup.py` does `monkeypatch.setattr(routes, "SessionLocal", sessions)` and then calls `routes._delete_scenario_conversation(...)`. Once `_delete_scenario_conversation` moves to `scenarios.py`, that submodule has its own independent `SessionLocal` binding — patching the package-level name does NOT affect it. Task 9 rewrites the test to patch `scenarios.SessionLocal` directly ("patch where it's used, not where it's defined"). No indirection layer is introduced in production code — this was explicitly decided against during the `memory_tasks.py` decomposition and the same reasoning applies here.
- `tests/test_concurrency.py` (tests `with_sse_heartbeat`) and `tests/test_feedback.py` (tests `message_feedback`) need NO changes — neither monkeypatches any module-level state; they just call the imported function directly. They will keep working once the package `__init__.py` re-exports both symbols (Task 9).

---

### Task 1: Create `helpers.py`

**Files:**
- Create: `backend/app/api/routes/helpers.py`

**Interfaces:**
- Consumes: nothing (leaf module, no dependency on any other new file)
- Produces: `obj(row)`, `sse(event: str, data) -> str`, `persisted_sse(sequence: int, event: str, data) -> str`, `with_sse_heartbeat(source, interval_seconds: float)` (async generator) — used by `conversations.py`, `memories.py` (`obj`), `agent_runs.py` (`persisted_sse`), `chat.py` (`sse`, `with_sse_heartbeat`), and re-exported (`with_sse_heartbeat` only) by the package facade in Task 9.

- [ ] **Step 1: Create the directory and the file**

Create `backend/app/api/routes/` as a plain directory (no `__init__.py` yet — that comes in Task 9, since creating it now would make Python treat the directory as a package and could interact unexpectedly with the still-existing flat `routes.py`; an `__init__.py`-less directory is inert to the import system). Write `backend/app/api/routes/helpers.py` with this exact content, extracted verbatim from `backend/app/api/routes.py` lines 160-161 (`obj`), 286-294 (`sse`, `persisted_sse`), and 409-435 (`with_sse_heartbeat`):

```python
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
```

- [ ] **Step 2: Verify syntax**

Run: `/opt/anaconda3/envs/LawStation/bin/python -m py_compile backend/app/api/routes/helpers.py`
Expected: no output, exit code 0.

- [ ] **Step 3: Verify the full suite is still unaffected**

Run: `/opt/anaconda3/envs/LawStation/bin/python -m pytest tests -q`
Expected: `176 passed` (the new file is not imported by anything yet, per the flat-module-shadows-package constraint, so this is a no-op smoke check).

- [ ] **Step 4: Commit**

```bash
git add backend/app/api/routes/helpers.py
git commit -m "refactor: extract SSE/serialization helpers into routes/helpers.py"
```

---

### Task 2: Create `index.py`

**Files:**
- Create: `backend/app/api/routes/index.py`

**Interfaces:**
- Consumes: nothing
- Produces: `router` (an `APIRouter(prefix="/api")` with one route, `GET /index/status`), consumed by the package facade in Task 9.

- [ ] **Step 1: Write the file**

Extracted verbatim from `backend/app/api/routes.py` lines 46 (import) and 52-54 (route):

```python
from fastapi import APIRouter

from mcp_servers.law_rag.server import get_index_status

router = APIRouter(prefix="/api")


@router.get("/index/status")
def index_status():
    return get_index_status()
```

- [ ] **Step 2: Verify syntax**

Run: `/opt/anaconda3/envs/LawStation/bin/python -m py_compile backend/app/api/routes/index.py`
Expected: no output, exit code 0.

- [ ] **Step 3: Commit**

```bash
git add backend/app/api/routes/index.py
git commit -m "refactor: extract index_status endpoint into routes/index.py"
```

---

### Task 3: Create `scenarios.py`

**Files:**
- Create: `backend/app/api/routes/scenarios.py`

**Interfaces:**
- Consumes: nothing from other new files (leaf module besides stdlib/framework/db imports)
- Produces: `router`, `SCENARIO_CONVERSATION_PREFIX`, `_scenario_catalog(request)`, `scenario_datasets`, `scenario_summaries`, `scenario_detail`, `scenario_run_outcome`, `_delete_scenario_conversation(ctx, conversation_id) -> list[str]`, `delete_scenario_conversation`. `_scenario_catalog` and `_delete_scenario_conversation` are re-exported by the package facade in Task 9 (tests import them), and `SessionLocal` inside this module is the exact object Task 9's test-file rewrite will monkeypatch as `scenarios.SessionLocal`.

- [ ] **Step 1: Write the file**

Extracted verbatim from `backend/app/api/routes.py` lines 21-30 (partial — only the models this group uses), 31 (partial — only `SessionLocal`), 49, and 57-158:

```python
import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import delete, select

from backend.app.core.context import get_user_context
from backend.app.db.models import (
    AgentRun,
    Conversation,
    MemoryJob,
    RetrievalTrace,
    ToolCallRecord,
)
from backend.app.db.session import SessionLocal

router = APIRouter(prefix="/api")
SCENARIO_CONVERSATION_PREFIX = "[场景] "


def _scenario_catalog(request: Request):
    catalog = getattr(request.app.state, "scenario_catalog", None)
    if catalog is None or not catalog.enabled:
        raise HTTPException(404, "场景观察模式未启用")
    return catalog


@router.get("/test-scenarios/datasets")
def scenario_datasets(request: Request, _ctx=Depends(get_user_context)):
    return _scenario_catalog(request).datasets()


@router.get("/test-scenarios/datasets/{dataset_id}/scenarios")
def scenario_summaries(dataset_id: str, request: Request, _ctx=Depends(get_user_context)):
    result = _scenario_catalog(request).scenario_summaries(dataset_id)
    if result is None:
        raise HTTPException(404, "测试数据集不存在")
    return result


@router.get("/test-scenarios/datasets/{dataset_id}/scenarios/{scenario_id}")
def scenario_detail(
    dataset_id: str, scenario_id: str, request: Request, _ctx=Depends(get_user_context)
):
    result = _scenario_catalog(request).scenario(dataset_id, scenario_id)
    if result is None:
        raise HTTPException(404, "测试场景不存在")
    return result


@router.get("/test-scenarios/agent-runs/{run_id}/outcome")
async def scenario_run_outcome(run_id: str, request: Request, ctx=Depends(get_user_context)):
    _scenario_catalog(request)
    result = await request.app.state.agent_runs.scenario_outcome(ctx, run_id)
    if result is None:
        raise HTTPException(404, "任务不存在或无权访问")
    return result


def _delete_scenario_conversation(ctx, conversation_id: str) -> list[str]:
    with SessionLocal() as db:
        conversation = db.scalar(select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.tenant_id == ctx.tenant_id,
            Conversation.user_id == ctx.user_id,
        ))
        if conversation is None:
            raise LookupError("会话不存在或无权访问")
        if not conversation.title.startswith(SCENARIO_CONVERSATION_PREFIX):
            raise PermissionError("只能清理场景观察模式创建的会话")
        active = db.scalar(select(AgentRun.id).where(
            AgentRun.tenant_id == ctx.tenant_id,
            AgentRun.user_id == ctx.user_id,
            AgentRun.conversation_id == conversation_id,
            AgentRun.status.in_(("queued", "running")),
        ).limit(1))
        if active:
            raise RuntimeError("场景会话仍有任务运行，请先停止或等待完成")
        thread_ids = list(db.scalars(select(AgentRun.langgraph_thread_id).where(
            AgentRun.tenant_id == ctx.tenant_id,
            AgentRun.user_id == ctx.user_id,
            AgentRun.conversation_id == conversation_id,
        )))
        owner = (
            MemoryJob.tenant_id == ctx.tenant_id,
            MemoryJob.user_id == ctx.user_id,
            MemoryJob.conversation_id == conversation_id,
        )
        db.execute(delete(MemoryJob).where(*owner))
        db.execute(delete(ToolCallRecord).where(
            ToolCallRecord.tenant_id == ctx.tenant_id,
            ToolCallRecord.user_id == ctx.user_id,
            ToolCallRecord.conversation_id == conversation_id,
        ))
        db.execute(delete(RetrievalTrace).where(
            RetrievalTrace.tenant_id == ctx.tenant_id,
            RetrievalTrace.user_id == ctx.user_id,
            RetrievalTrace.conversation_id == conversation_id,
        ))
        db.delete(conversation)
        db.commit()
        return thread_ids


@router.delete("/test-scenarios/conversations/{conversation_id}", status_code=204)
async def delete_scenario_conversation(
    conversation_id: str, request: Request, ctx=Depends(get_user_context)
):
    _scenario_catalog(request)
    try:
        thread_ids = await asyncio.to_thread(
            _delete_scenario_conversation, ctx, conversation_id
        )
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    for thread_id in thread_ids:
        await request.app.state.agent_runtime.delete_checkpoint_thread(thread_id)
```

- [ ] **Step 2: Verify syntax**

Run: `/opt/anaconda3/envs/LawStation/bin/python -m py_compile backend/app/api/routes/scenarios.py`
Expected: no output, exit code 0.

- [ ] **Step 3: Commit**

```bash
git add backend/app/api/routes/scenarios.py
git commit -m "refactor: extract scenario-observation-mode endpoints into routes/scenarios.py"
```

---

### Task 4: Create `conversations.py`

**Files:**
- Create: `backend/app/api/routes/conversations.py`

**Interfaces:**
- Consumes: `obj` from `.helpers` (Task 1)
- Produces: `router` with `GET /users`, `GET /conversations`, `POST /conversations`, `GET /conversations/{conversation_id}/messages`, consumed by the package facade in Task 9.

- [ ] **Step 1: Write the file**

Extracted verbatim from `backend/app/api/routes.py` lines 164-196 (note: the original `messages` function relies on `HTTPException`, imported at the original file's top level but not shown in this group's own import block in the design spec — it is needed here):

```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.context import RequestUserContext, get_user_context
from backend.app.db.models import Conversation, MessageFeedback, User
from backend.app.db.session import get_db
from backend.app.schemas import ConversationCreate
from backend.app.services.repositories import OwnedRepository

from .helpers import obj

router = APIRouter(prefix="/api")


@router.get("/users")
def users(db: Session = Depends(get_db)):
    return [obj(user) for user in db.scalars(select(User).order_by(User.name))]


@router.get("/conversations")
def conversations(ctx: RequestUserContext = Depends(get_user_context), db: Session = Depends(get_db)):
    return [obj(item) for item in OwnedRepository(db, ctx).conversations()]


@router.post("/conversations", status_code=201)
def create_conversation(payload: ConversationCreate, ctx=Depends(get_user_context), db=Depends(get_db)):
    item = Conversation(tenant_id=ctx.tenant_id, user_id=ctx.user_id, title=payload.title)
    db.add(item)
    db.commit()
    return obj(item)


@router.get("/conversations/{conversation_id}/messages")
def messages(conversation_id: str, ctx=Depends(get_user_context), db=Depends(get_db)):
    repo = OwnedRepository(db, ctx)
    if not repo.conversation(conversation_id):
        raise HTTPException(404, "会话不存在或无权访问")
    items = repo.messages(conversation_id)
    feedback_by_message = {
        item.message_id: item.score
        for item in db.scalars(select(MessageFeedback).where(
            MessageFeedback.tenant_id == ctx.tenant_id,
            MessageFeedback.user_id == ctx.user_id,
            MessageFeedback.message_id.in_([message.id for message in items]),
        ))
    } if items else {}
    return [{**obj(item), "feedback_score": feedback_by_message.get(item.id)} for item in items]
```

- [ ] **Step 2: Verify syntax**

Run: `/opt/anaconda3/envs/LawStation/bin/python -m py_compile backend/app/api/routes/conversations.py`
Expected: no output, exit code 0.

- [ ] **Step 3: Commit**

```bash
git add backend/app/api/routes/conversations.py
git commit -m "refactor: extract users/conversations/messages CRUD into routes/conversations.py"
```

---

### Task 5: Create `memories.py`

**Files:**
- Create: `backend/app/api/routes/memories.py`

**Interfaces:**
- Consumes: `obj` from `.helpers` (Task 1)
- Produces: `router` with `GET /memories`, `PATCH /memories/{memory_id}`, `POST /memories/{memory_id}/confirm`, `POST /memories/{memory_id}/reject`, `DELETE /memories/{memory_id}`, `DELETE /memories`, `GET /memory-jobs/{job_id}`, and `_set_memory_status` (private helper, not consumed elsewhere), consumed by the package facade in Task 9.

- [ ] **Step 1: Write the file**

Extracted verbatim from `backend/app/api/routes.py` lines 199-283:

```python
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select

from backend.app.core.context import get_user_context
from backend.app.core.logging import audit
from backend.app.db.models import MemoryJob
from backend.app.db.session import get_db
from backend.app.schemas import MemoryUpdate, MemoryVersionRequest
from backend.app.services.repositories import OwnedRepository

from .helpers import obj

router = APIRouter(prefix="/api")


@router.get("/memories")
def memories(
    conversation_id: str | None = Query(None),
    scope: str | None = Query(None, pattern="^(user|conversation)$"),
    status: str | None = Query(None, pattern="^(pending|active|superseded|rejected|expired)$"),
    memory_type: str | None = Query(None),
    ctx=Depends(get_user_context),
    db=Depends(get_db),
):
    repo = OwnedRepository(db, ctx)
    if conversation_id and not repo.conversation(conversation_id):
        raise HTTPException(404, "会话不存在或无权访问")
    return [obj(item) for item in repo.memories(
        conversation_id, scope=scope, status=status, memory_type=memory_type
    )]


@router.patch("/memories/{memory_id}")
def update_memory(memory_id: str, payload: MemoryUpdate, ctx=Depends(get_user_context), db=Depends(get_db)):
    result = OwnedRepository(db, ctx).update_memory(memory_id, payload.content, payload.version)
    if result == "missing":
        raise HTTPException(404, "记忆不存在或无权访问")
    if result == "conflict":
        raise HTTPException(409, "记忆已被其他操作更新，请刷新后重试")
    db.commit()
    return {"ok": True}


def _set_memory_status(memory_id: str, payload: MemoryVersionRequest, target: str, ctx, db):
    repo = OwnedRepository(db, ctx)
    owned_memory = repo.memory(memory_id)
    result = repo.set_memory_status(memory_id, payload.version, target)
    if result == "missing":
        raise HTTPException(404, "记忆不存在或无权访问")
    if result == "conflict":
        raise HTTPException(409, "记忆已被其他操作更新，请刷新后重试")
    db.commit()
    audit(
        "memory.candidate.confirmed" if target == "active" else "memory.candidate.rejected",
        request_id=ctx.request_id,
        tenant_id=ctx.tenant_id,
        user_id=ctx.user_id,
        conversation_id=owned_memory.conversation_id if owned_memory else None,
        status=target,
        memory_id=memory_id,
    )
    return {"ok": True}


@router.post("/memories/{memory_id}/confirm")
def confirm_memory(memory_id: str, payload: MemoryVersionRequest, ctx=Depends(get_user_context), db=Depends(get_db)):
    return _set_memory_status(memory_id, payload, "active", ctx, db)


@router.post("/memories/{memory_id}/reject")
def reject_memory(memory_id: str, payload: MemoryVersionRequest, ctx=Depends(get_user_context), db=Depends(get_db)):
    return _set_memory_status(memory_id, payload, "rejected", ctx, db)


@router.delete("/memories/{memory_id}", status_code=204)
def delete_memory(memory_id: str, ctx=Depends(get_user_context), db=Depends(get_db)):
    if not OwnedRepository(db, ctx).delete_memory(memory_id):
        raise HTTPException(404, "记忆不存在或无权访问")
    db.commit()


@router.delete("/memories", status_code=204)
def clear_memories(conversation_id: str | None = Query(None), ctx=Depends(get_user_context), db=Depends(get_db)):
    repo = OwnedRepository(db, ctx)
    if conversation_id and not repo.conversation(conversation_id):
        raise HTTPException(404, "会话不存在或无权访问")
    repo.clear_memories(conversation_id)
    db.commit()


@router.get("/memory-jobs/{job_id}")
def memory_job(job_id: str, ctx=Depends(get_user_context), db=Depends(get_db)):
    job = db.scalar(select(MemoryJob).where(
        MemoryJob.id == job_id,
        MemoryJob.tenant_id == ctx.tenant_id,
        MemoryJob.user_id == ctx.user_id,
    ))
    if not job:
        raise HTTPException(404, "记忆任务不存在或无权访问")
    return obj(job)
```

- [ ] **Step 2: Verify syntax**

Run: `/opt/anaconda3/envs/LawStation/bin/python -m py_compile backend/app/api/routes/memories.py`
Expected: no output, exit code 0.

- [ ] **Step 3: Commit**

```bash
git add backend/app/api/routes/memories.py
git commit -m "refactor: extract memory CRUD endpoints into routes/memories.py"
```

---

### Task 6: Create `agent_runs.py`

**Files:**
- Create: `backend/app/api/routes/agent_runs.py`

**Interfaces:**
- Consumes: `persisted_sse` from `.helpers` (Task 1)
- Produces: `router` with `POST /conversations/{conversation_id}/runs`, `GET /agent-runs/{run_id}`, `GET /conversations/{conversation_id}/active-run`, `POST /agent-runs/{run_id}/cancel`, `GET /agent-runs/{run_id}/events`, consumed by the package facade in Task 9. Verified line ranges inside this new file (for later doc-sync reference in Task 10): `create_agent_run` 18-49, `get_agent_run` 52-57, `active_agent_run` 60-67, `cancel_agent_run` 70-75, `agent_run_events` 78-127.

- [ ] **Step 1: Write the file**

Extracted verbatim from `backend/app/api/routes.py` lines 297-406 (note: `create_agent_run`'s `payload: ChatRequest` parameter and `agent_run_events`'s `json.loads` call both need imports that weren't listed in this group's illustrative import block in the design spec — `ChatRequest` and `json` are added here because the code genuinely needs them, following the same "brief's import list is illustrative, not exhaustive" precedent established during the `memory_tasks.py` decomposition):

```python
import asyncio
import json

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from backend.app.agent.concurrency import ConcurrencyIdentity, ConversationBusyError
from backend.app.core.config import get_settings
from backend.app.core.context import get_user_context
from backend.app.schemas import ChatRequest
from backend.app.services.agent_runs import TERMINAL_STATUSES, AgentRunConflict, run_payload

from .helpers import persisted_sse

router = APIRouter(prefix="/api")


@router.post("/conversations/{conversation_id}/runs", status_code=202)
async def create_agent_run(
    conversation_id: str,
    payload: ChatRequest,
    request: Request,
    ctx=Depends(get_user_context),
):
    """接受一个后台 AgentRun；202 只表示已排队，不表示回答已经生成。"""

    identity = ConcurrencyIdentity(
        request_id=ctx.request_id,
        tenant_id=ctx.tenant_id,
        user_id=ctx.user_id,
        conversation_id=conversation_id,
    )
    try:
        await request.app.state.agent_concurrency.reserve(identity)
        run = await asyncio.to_thread(
            request.app.state.agent_runs.create, ctx, conversation_id, payload.content
        )
    except ConversationBusyError as exc:
        raise HTTPException(409, str(exc)) from exc
    except LookupError as exc:
        await request.app.state.agent_concurrency.release_reservation(identity)
        raise HTTPException(404, str(exc)) from exc
    except AgentRunConflict as exc:
        await request.app.state.agent_concurrency.release_reservation(identity)
        raise HTTPException(409, str(exc)) from exc
    except Exception:
        await request.app.state.agent_concurrency.release_reservation(identity)
        raise
    return run_payload(run)


@router.get("/agent-runs/{run_id}")
async def get_agent_run(run_id: str, request: Request, ctx=Depends(get_user_context)):
    run = await asyncio.to_thread(request.app.state.agent_runs.owned, ctx, run_id)
    if run is None:
        raise HTTPException(404, "任务不存在或无权访问")
    return run_payload(run)


@router.get("/conversations/{conversation_id}/active-run")
async def active_agent_run(
    conversation_id: str, request: Request, ctx=Depends(get_user_context)
):
    run = await asyncio.to_thread(
        request.app.state.agent_runs.active_for_conversation, ctx, conversation_id
    )
    return run_payload(run) if run else None


@router.post("/agent-runs/{run_id}/cancel")
async def cancel_agent_run(run_id: str, request: Request, ctx=Depends(get_user_context)):
    run = await request.app.state.agent_runs.cancel(ctx, run_id)
    if run is None:
        raise HTTPException(404, "任务不存在或无权访问")
    return run_payload(run)


@router.get("/agent-runs/{run_id}/events")
async def agent_run_events(
    run_id: str,
    request: Request,
    after_sequence: int = Query(0, ge=0),
    last_event_id: str | None = Header(None, alias="Last-Event-ID"),
    ctx=Depends(get_user_context),
):
    """从 sequence 游标重放历史事件，并继续等待该 Run 的新事件。"""

    # Query 游标和标准 Last-Event-ID 二者取较大值，既支持显式重连，也兼容浏览器
    # EventSource 语义；非法游标在开始 StreamingResponse 前即被拒绝。
    try:
        cursor = max(after_sequence, int(last_event_id or 0))
    except ValueError as exc:
        raise HTTPException(400, "Last-Event-ID必须是非负整数") from exc
    manager = request.app.state.agent_runs
    owned = await asyncio.to_thread(manager.owned, ctx, run_id)
    if owned is None:
        raise HTTPException(404, "任务不存在或无权访问")

    async def source():
        nonlocal cursor
        while True:
            run, rows = await asyncio.to_thread(manager.events, ctx, run_id, cursor)
            if run is None:
                return
            for row in rows:
                cursor = row.sequence
                yield persisted_sse(
                    row.sequence, row.event_type, json.loads(row.payload_json)
                )
            # 只有终态且所有持久事件均已发送时才关闭流。浏览器主动断开只结束这个
            # generator，不会取消独立运行的 AgentRun Worker。
            if run.status in TERMINAL_STATUSES and cursor >= run.last_event_seq:
                return
            await manager.wait_for_events(get_settings().sse_heartbeat_seconds)
            if not rows:
                # SSE comment 不属于 AgentRunEvent，不占 sequence，也不会生成空消息。
                yield ": heartbeat\n\n"

    return StreamingResponse(
        source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
```

- [ ] **Step 2: Verify syntax**

Run: `/opt/anaconda3/envs/LawStation/bin/python -m py_compile backend/app/api/routes/agent_runs.py`
Expected: no output, exit code 0.

- [ ] **Step 3: Commit**

```bash
git add backend/app/api/routes/agent_runs.py
git commit -m "refactor: extract AgentRun lifecycle endpoints into routes/agent_runs.py"
```

---

### Task 7: Create `feedback.py`

**Files:**
- Create: `backend/app/api/routes/feedback.py`

**Interfaces:**
- Consumes: nothing from other new files
- Produces: `router` with `POST /messages/{message_id}/feedback`, `_sync_message_feedback` (private, background task), `message_feedback` — this last one is re-exported by the package facade in Task 9 (`tests/test_feedback.py` imports it directly and calls it with no monkeypatching, so no test changes needed). Verified line range inside this new file: `message_feedback` 32-78.

- [ ] **Step 1: Write the file**

Extracted verbatim from `backend/app/api/routes.py` lines 492-558 (note: `message_feedback`'s body uses `HTTPException` and `Request`, both imported at the original file's top level — added here because the code genuinely needs them):

```python
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from sqlalchemy import select

from backend.app.core.context import get_user_context
from backend.app.core.logging import audit
from backend.app.db.models import Message, MessageFeedback
from backend.app.db.session import SessionLocal, get_db
from backend.app.schemas import MessageFeedbackRequest

router = APIRouter(prefix="/api")


async def _sync_message_feedback(app, feedback_id: str) -> None:
    with SessionLocal() as db:
        feedback = db.get(MessageFeedback, feedback_id)
        if not feedback:
            return
        trace_id = feedback.langsmith_trace_id or ""
        score = feedback.score
        comment = feedback.comment
    synced = await app.state.langsmith_observability.create_user_feedback(
        trace_id=trace_id, score=score, comment=comment
    )
    with SessionLocal() as db:
        feedback = db.get(MessageFeedback, feedback_id)
        if feedback:
            feedback.sync_status = "synced" if synced else "unavailable"
            feedback.last_error = "" if synced else "LangSmith 当前不可用或该消息没有 trace"
            db.commit()


@router.post("/messages/{message_id}/feedback")
async def message_feedback(
    message_id: str,
    payload: MessageFeedbackRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    ctx=Depends(get_user_context),
    db=Depends(get_db),
):
    message = db.scalar(select(Message).where(
        Message.id == message_id,
        Message.tenant_id == ctx.tenant_id,
        Message.user_id == ctx.user_id,
        Message.role == "assistant",
    ))
    if not message:
        raise HTTPException(404, "消息不存在或无权访问")
    feedback = db.scalar(select(MessageFeedback).where(
        MessageFeedback.tenant_id == ctx.tenant_id,
        MessageFeedback.user_id == ctx.user_id,
        MessageFeedback.message_id == message_id,
    ))
    if feedback is None:
        feedback = MessageFeedback(
            tenant_id=ctx.tenant_id,
            user_id=ctx.user_id,
            message_id=message_id,
        )
        db.add(feedback)
    feedback.score = payload.score
    feedback.comment = payload.comment.strip()
    feedback.langsmith_trace_id = message.langsmith_trace_id
    feedback.sync_status = "pending"
    feedback.last_error = ""
    db.commit()
    db.refresh(feedback)
    audit(
        "message.feedback.saved",
        request_id=ctx.request_id,
        tenant_id=ctx.tenant_id,
        user_id=ctx.user_id,
        conversation_id=message.conversation_id,
        message_id=message.id,
        score=payload.score,
        status="saved",
    )
    background_tasks.add_task(_sync_message_feedback, request.app, feedback.id)
    return {"ok": True, "sync_status": feedback.sync_status}
```

- [ ] **Step 2: Verify syntax**

Run: `/opt/anaconda3/envs/LawStation/bin/python -m py_compile backend/app/api/routes/feedback.py`
Expected: no output, exit code 0.

- [ ] **Step 3: Commit**

```bash
git add backend/app/api/routes/feedback.py
git commit -m "refactor: extract message feedback endpoint into routes/feedback.py"
```

---

### Task 8: Create `chat.py`

**Files:**
- Create: `backend/app/api/routes/chat.py`

**Interfaces:**
- Consumes: `sse`, `with_sse_heartbeat` from `.helpers` (Task 1)
- Produces: `router` with `POST /conversations/{conversation_id}/messages/stream`, plus private helpers `_validate_conversation`, `_prepare_chat`, `_save_assistant`. `with_sse_heartbeat` re-export (Task 9) comes from `.helpers`, not from here, but `stream_message` itself is not re-exported anywhere (no test imports it directly). Verified line range inside this new file: `stream_message` 83-278.

This is the largest and most complex extraction in this plan — the SSE streaming chat endpoint, byte-for-byte identical to the original.

- [ ] **Step 1: Write the file**

Extracted verbatim from `backend/app/api/routes.py` lines 438-490 (`_validate_conversation`, `_prepare_chat`, `_save_assistant`) and 561-756 (`stream_message`):

```python
import asyncio
import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from backend.app.agent.concurrency import (
    AgentConcurrencyManager,
    AgentQueueTimeoutError,
    ConcurrencyIdentity,
    ConversationBusyError,
)
from backend.app.agent.service import AgentService
from backend.app.core.config import get_settings
from backend.app.core.context import RequestUserContext, get_user_context
from backend.app.core.logging import audit, summary
from backend.app.db.models import Message
from backend.app.db.session import SessionLocal
from backend.app.schemas import ChatRequest
from backend.app.services.memory import MemoryService
from backend.app.services.repositories import OwnedRepository

from .helpers import sse, with_sse_heartbeat

router = APIRouter(prefix="/api")


def _validate_conversation(ctx: RequestUserContext, conversation_id: str) -> None:
    with SessionLocal() as db:
        if not OwnedRepository(db, ctx).conversation(conversation_id):
            raise HTTPException(404, "会话不存在或无权访问")


def _prepare_chat(
    ctx: RequestUserContext,
    conversation_id: str,
    content: str,
    langsmith_trace_id: str | None = None,
):
    with SessionLocal() as db:
        repo = OwnedRepository(db, ctx)
        if not repo.conversation(conversation_id):
            raise HTTPException(404, "会话不存在或无权访问")
        user_message = Message(
            tenant_id=ctx.tenant_id,
            user_id=ctx.user_id,
            conversation_id=conversation_id,
            role="user",
            content=content,
            langsmith_trace_id=langsmith_trace_id,
        )
        db.add(user_message)
        db.commit()
        memory_context, history = MemoryService(db, ctx).context(
            conversation_id, content, user_message.id
        )
        return user_message.id, memory_context, history


def _save_assistant(
    ctx: RequestUserContext,
    conversation_id: str,
    content: str,
    status: str,
    langsmith_trace_id: str | None = None,
) -> Message:
    with SessionLocal() as db:
        assistant = Message(
            tenant_id=ctx.tenant_id,
            user_id=ctx.user_id,
            conversation_id=conversation_id,
            role="assistant",
            content=content,
            status=status,
            langsmith_trace_id=langsmith_trace_id,
        )
        db.add(assistant)
        db.commit()
        return assistant


@router.post("/conversations/{conversation_id}/messages/stream")
async def stream_message(
    conversation_id: str,
    payload: ChatRequest,
    request: Request,
    ctx=Depends(get_user_context),
):
    started = time.perf_counter()
    observability = request.app.state.langsmith_observability
    consultation_trace = observability.start_consultation(
        request_id=ctx.request_id,
        tenant_id=ctx.tenant_id,
        user_id=ctx.user_id,
        conversation_id=conversation_id,
        question=payload.content,
        model_name=get_settings().deepseek_model,
    )
    concurrency: AgentConcurrencyManager = request.app.state.agent_concurrency
    identity = ConcurrencyIdentity(
        request_id=ctx.request_id,
        tenant_id=ctx.tenant_id,
        user_id=ctx.user_id,
        conversation_id=conversation_id,
    )
    try:
        with consultation_trace.activate():
            async with consultation_trace.span(
                "chat.reserve_conversation",
                inputs={"conversation_id": conversation_id},
            ):
                await concurrency.reserve(identity)
    except ConversationBusyError as exc:
        await consultation_trace.finish(
            outputs={"status": "conversation_busy"},
            error="ConversationBusyError",
        )
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    try:
        with consultation_trace.activate():
            async with consultation_trace.span(
                "chat.validate_conversation",
                inputs={"conversation_id": conversation_id},
            ):
                await asyncio.to_thread(_validate_conversation, ctx, conversation_id)
    except Exception as exc:
        await concurrency.release_reservation(identity)
        await consultation_trace.finish(
            outputs={"status": "validation_failed"},
            error=type(exc).__name__,
        )
        raise
    audit("conversation.received", request_id=ctx.request_id, tenant_id=ctx.tenant_id, user_id=ctx.user_id, conversation_id=conversation_id, status="received", question_chars=len(payload.content), question_summary=summary(payload.content))

    async def events():
        answer = []
        citations = []
        agent = AgentService(
            request.app.state.agent_runtime,
            ctx,
            conversation_id,
            trace_config=consultation_trace.config,
            trace_id=consultation_trace.trace_id,
        )
        queue_started = time.perf_counter()
        queue_duration_ms = 0
        slot_acquired = False
        first_status_duration_ms: int | None = None
        first_text_token_duration_ms: int | None = None
        try:
            with consultation_trace.activate():
                yield sse("message_start", {"request_id": ctx.request_id})
                if await concurrency.would_queue(identity):
                    first_status_duration_ms = int((time.perf_counter() - started) * 1000)
                    yield sse("agent_status", {
                        "request_id": ctx.request_id,
                        "user_id": ctx.user_id,
                        "conversation_id": conversation_id,
                        "agent": "coordinator",
                        "status": "queued",
                        "message": "咨询任务正在排队",
                    })
                async with consultation_trace.span("chat.queue"):
                    await concurrency.acquire(identity)
                    slot_acquired = True
                queue_duration_ms = int((time.perf_counter() - queue_started) * 1000)
                async with consultation_trace.span(
                    "chat.load_memory_snapshot",
                    inputs={"question_chars": len(payload.content)},
                ):
                    user_message_id, memory_context, history = await asyncio.to_thread(
                        _prepare_chat,
                        ctx,
                        conversation_id,
                        payload.content,
                        consultation_trace.trace_id,
                    )
                audit("conversation.started", request_id=ctx.request_id, tenant_id=ctx.tenant_id, user_id=ctx.user_id, conversation_id=conversation_id, status="started", history_count=len(history), memory_context_chars=len(memory_context))
                async for item in agent.run(memory_context, history, payload.content):
                    elapsed_ms = int((time.perf_counter() - started) * 1000)
                    if item["event"] == "agent_status" and first_status_duration_ms is None:
                        first_status_duration_ms = elapsed_ms
                    if item["event"] == "token":
                        if first_text_token_duration_ms is None:
                            first_text_token_duration_ms = elapsed_ms
                        answer.append(item["data"])
                    elif item["event"] == "citations":
                        citations = item["data"]
                    yield sse(item["event"], item["data"])
                persisted_answer = agent.final_answer or "".join(answer)
                async with consultation_trace.span(
                    "chat.persist_answer",
                    inputs={"answer_chars": len(persisted_answer)},
                ):
                    assistant = await asyncio.to_thread(
                        _save_assistant, ctx, conversation_id, persisted_answer, "complete",
                        consultation_trace.trace_id,
                    )
                async with consultation_trace.span("chat.enqueue_memory"):
                    job = await asyncio.to_thread(
                        request.app.state.memory_tasks.enqueue,
                        ctx,
                        conversation_id,
                        user_message_id,
                    )
            total_duration_ms = int((time.perf_counter() - started) * 1000)
            audit("conversation.completed", request_id=ctx.request_id, tenant_id=ctx.tenant_id, user_id=ctx.user_id, conversation_id=conversation_id, status="success", duration_ms=total_duration_ms, total_duration_ms=total_duration_ms, queue_duration_ms=queue_duration_ms, first_status_duration_ms=first_status_duration_ms, first_text_token_duration_ms=first_text_token_duration_ms, answer_chars=len(assistant.content), answer_summary=summary(assistant.content), memory_job_id=job.id, tool_call_count=agent.tool_call_count, model_call_count=agent.model_call_count)
            yield sse("memory_status", {"status": "pending", "job_id": job.id})
            await consultation_trace.finish(
                outputs={
                    "status": "success",
                    "final_answer": assistant.content,
                    "citations": citations,
                    "model_call_count": agent.model_call_count,
                    "tool_call_count": agent.tool_call_count,
                    "queue_duration_ms": queue_duration_ms,
                    "total_duration_ms": total_duration_ms,
                },
            )
            yield sse("message_end", {"message_id": assistant.id, "trace_available": bool(agent.langsmith_trace_id)})
        except AgentQueueTimeoutError as exc:
            total_duration_ms = int((time.perf_counter() - started) * 1000)
            audit("conversation.failed", level=logging.WARNING, request_id=ctx.request_id, tenant_id=ctx.tenant_id, user_id=ctx.user_id, conversation_id=conversation_id, status="queue_timeout", duration_ms=total_duration_ms, total_duration_ms=total_duration_ms, queue_duration_ms=total_duration_ms, first_status_duration_ms=first_status_duration_ms, first_text_token_duration_ms=None)
            await consultation_trace.finish(
                outputs={"status": "queue_timeout"},
                error="AgentQueueTimeoutError",
            )
            yield sse("error", {"message": str(exc)})
        except asyncio.CancelledError:
            if answer:
                await asyncio.to_thread(
                    _save_assistant, ctx, conversation_id, "".join(answer), "interrupted",
                    agent.langsmith_trace_id,
                )
            total_duration_ms = int((time.perf_counter() - started) * 1000)
            audit("conversation.interrupted", level=logging.WARNING, request_id=ctx.request_id, tenant_id=ctx.tenant_id, user_id=ctx.user_id, conversation_id=conversation_id, status="interrupted", duration_ms=total_duration_ms, total_duration_ms=total_duration_ms, queue_duration_ms=queue_duration_ms, first_status_duration_ms=first_status_duration_ms, first_text_token_duration_ms=first_text_token_duration_ms, answer_chars=len("".join(answer)))
            await consultation_trace.finish(
                outputs={
                    "status": "interrupted",
                    "partial_answer": "".join(answer),
                    "answer_chars": len("".join(answer)),
                },
                error="ClientDisconnected",
            )
            raise
        except Exception as exc:  # noqa: BLE001 - legacy SSE boundary converts failures to terminal events
            if answer:
                await asyncio.to_thread(
                    _save_assistant, ctx, conversation_id, "".join(answer), "interrupted",
                    agent.langsmith_trace_id,
                )
            total_duration_ms = int((time.perf_counter() - started) * 1000)
            audit("conversation.failed", level=logging.ERROR, request_id=ctx.request_id, tenant_id=ctx.tenant_id, user_id=ctx.user_id, conversation_id=conversation_id, status="failed", duration_ms=total_duration_ms, total_duration_ms=total_duration_ms, queue_duration_ms=queue_duration_ms, first_status_duration_ms=first_status_duration_ms, first_text_token_duration_ms=first_text_token_duration_ms, error_type=type(exc).__name__, error=summary(str(exc)))
            await consultation_trace.finish(
                outputs={
                    "status": "failed",
                    "partial_answer": "".join(answer),
                },
                error=type(exc).__name__,
            )
            yield sse("error", {"message": str(exc)})
        finally:
            if slot_acquired:
                await concurrency.release(identity)
            await concurrency.release_reservation(identity)
            await consultation_trace.finish(outputs={"status": "closed"})

    settings = get_settings()
    return StreamingResponse(
        with_sse_heartbeat(events(), settings.sse_heartbeat_seconds),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
```

- [ ] **Step 2: Verify syntax**

Run: `/opt/anaconda3/envs/LawStation/bin/python -m py_compile backend/app/api/routes/chat.py`
Expected: no output, exit code 0.

- [ ] **Step 3: Diff-check against the original for byte-for-byte accuracy**

This file is large and complex — before committing, run a diff of the extracted body against the original source range to catch any transcription slip:

```bash
sed -n '438,490p;561,756p' backend/app/api/routes.py > /tmp/chat_original_body.txt
```

Manually confirm (by eye, or by comparing against the `def`/decorator lines) that every line in `/tmp/chat_original_body.txt` appears unchanged in `backend/app/api/routes/chat.py` — the only differences should be the new file's import block and the `router = APIRouter(prefix="/api")` line at the top, neither of which existed inline at that position in the original.

- [ ] **Step 4: Commit**

```bash
git add backend/app/api/routes/chat.py
git commit -m "refactor: extract SSE streaming chat endpoint into routes/chat.py"
```

---

### Task 9: Activate the package — facade, delete flat module, fix test patch path

**This is the highest-risk task: it is the single atomic switch from the old flat `routes.py` to the new `routes/` package.** Everything up to this point was inert (per the flat-module-shadows-package constraint). This task makes the package live and must leave the full test suite green.

**Files:**
- Create: `backend/app/api/routes/__init__.py`
- Delete: `backend/app/api/routes.py`
- Modify: `tests/test_scenario_cleanup.py` (lines 8, 21, 50, 52, 54, 77, 90)

**Interfaces:**
- Consumes: `router` from each of `index`, `scenarios`, `conversations`, `memories`, `agent_runs`, `feedback`, `chat` (Tasks 1-8); `with_sse_heartbeat` from `chat` (which re-exports it via `from .helpers import sse, with_sse_heartbeat` — actually `with_sse_heartbeat` is defined in `helpers.py` and merely imported into `chat.py`'s namespace, so the facade imports it from `.helpers` directly, not from `.chat`, to import it from its actual definition site); `message_feedback` from `feedback`; `_delete_scenario_conversation` and `_scenario_catalog` from `scenarios`.
- Produces: `backend.app.api.routes.router` (used unchanged by `backend/app/main.py:15`, no change needed there), plus the four re-exported symbols used directly by tests.

- [ ] **Step 1: Write the package facade**

Create `backend/app/api/routes/__init__.py`:

```python
from fastapi import APIRouter

from . import agent_runs, chat, conversations, feedback, index, memories, scenarios
from .feedback import message_feedback
from .helpers import with_sse_heartbeat
from .scenarios import _delete_scenario_conversation, _scenario_catalog

router = APIRouter()
for _module in (index, scenarios, conversations, memories, agent_runs, feedback, chat):
    router.include_router(_module.router)

__all__ = ["router"]
```

Note: `with_sse_heartbeat` is imported from `.helpers` (its actual definition site, Task 1) rather than from `.chat` (which only imports it for its own internal use) — importing from the definition site is more direct and avoids relying on `chat.py`'s internal import being named exactly `with_sse_heartbeat` in its own namespace (which it is, but there's no reason to depend on that transitively).

- [ ] **Step 2: Delete the flat module**

```bash
git rm backend/app/api/routes.py
```

This is safe now: every symbol that lived in the flat file has a home in the new package (Tasks 1-8), and the facade (`Step 1` above) reassembles the same public surface.

- [ ] **Step 3: Fix the test file's monkeypatch path**

`tests/test_scenario_cleanup.py` currently does `from backend.app.api import routes` and then `monkeypatch.setattr(routes, "SessionLocal", sessions)` before calling `routes._delete_scenario_conversation(...)`. Now that `_delete_scenario_conversation` lives in `scenarios.py` with its own independent `SessionLocal` binding, this patch must target `scenarios.SessionLocal` directly, or it will silently fail to intercept the real `SessionLocal` (the test would then try to hit a real, uninitialized database instead of the in-memory SQLite fixture).

Verified: the name `routes` is not used anywhere else in this test file besides these exact 6 references, so replace the import outright rather than adding a second name.

Edit `tests/test_scenario_cleanup.py`:

```python
# Line 8, change:
from backend.app.api import routes
# to:
from backend.app.api.routes import scenarios
```

```python
# Line 21, change:
    monkeypatch.setattr(routes, "SessionLocal", sessions)
# to:
    monkeypatch.setattr(scenarios, "SessionLocal", sessions)
```

```python
# Lines 50, 52, 54, 77, change every occurrence of:
routes._delete_scenario_conversation(...)
# to:
scenarios._delete_scenario_conversation(...)
```

(Four call sites total: `routes._delete_scenario_conversation(other, "scenario-a")` at line 50, `routes._delete_scenario_conversation(owner, "ordinary-a")` at line 52, `routes._delete_scenario_conversation(owner, "scenario-a")` at line 54 (inside an `assert ... == []`), and `routes._delete_scenario_conversation(owner, "scenario-a")` at line 77 (inside a `pytest.raises(RuntimeError, ...)` block).)

```python
# Line 90, change:
        routes._scenario_catalog(request)
# to:
        scenarios._scenario_catalog(request)
```

- [ ] **Step 4: Sanity-check no other stale references remain**

```bash
grep -rn "from backend.app.api import routes\b" tests backend --include="*.py"
```

Expected: no output (the only occurrence, in `test_scenario_cleanup.py`, was just changed in Step 3). Also confirm `backend/app/main.py` still reads `from backend.app.api.routes import router` unchanged (it imports the package's re-exported `router`, which works identically to before):

```bash
grep -n "from backend.app.api.routes import router" backend/app/main.py
```

Expected: one match, line 15, unchanged.

- [ ] **Step 5: Run the full test suite**

Run: `/opt/anaconda3/envs/LawStation/bin/python -m pytest tests -q`
Expected: `176 passed` — the same baseline count as before this plan, with `test_scenario_cleanup.py`'s three tests (`test_cleanup_is_owner_and_prefix_scoped`, `test_cleanup_rejects_active_run`, `test_disabled_catalog_is_hidden_as_not_found`) now correctly exercising the patched `scenarios.SessionLocal`.

If any test fails here, do not proceed — this is the activation point where any earlier extraction mistake (a dropped import, a wrong line range, a missing `HTTPException` import in one of the domain files) will surface for the first time, since Tasks 1-8 could only syntax-check, not actually import, their files.

- [ ] **Step 6: Commit**

```bash
git add backend/app/api/routes/__init__.py backend/app/api/routes.py tests/test_scenario_cleanup.py
git commit -m "refactor: assemble routes/ package via include_router, remove old routes.py, fix test monkeypatch path"
```

(`git add backend/app/api/routes.py` stages the deletion recorded by `git rm` in Step 2, if not already staged.)

---

### Task 10: Update documentation

**Files:**
- Modify: `ai-context/SPEC.md`
- Modify: `resume/architecture.md`
- Modify: `resume/agent-persistence-code-guide.md`
- Modify: `resume/memory.md`
- Modify: `resume/review-findings.md`
- Modify: `resume/sse-code-guide.md`
- Modify: `resume/concurrency.md`
- Modify: `resume/api-chat.md`
- Modify: `resume/langsmith.md`

**Interfaces:** None — documentation only, no code interfaces involved. This task has no automated test; verification is a final grep sweep (Step 10) confirming no stale `backend/app/api/routes.py::<symbol>` references remain outside historical spec/plan documents (which are immutable records of past decisions and are intentionally NOT updated — same precedent as `docs/agent-learning/*.md` being left alone during the `memory_tasks.py` decomposition).

Every reference below points to a symbol that moved to a specific new file. Apply each edit exactly as literal find-and-replace (this is prose, not code — small paraphrase differences are fine as long as the file path and symbol name become correct):

- [ ] **Step 1: `ai-context/SPEC.md` line 76**

Change:
```
| `backend/app/api/` | REST 与 SSE 接口，串联用户上下文、数据库、记忆和 Agent | `routes.py::stream_message`、`routes.py::sse` |
```
to:
```
| `backend/app/api/` | REST 与 SSE 接口，串联用户上下文、数据库、记忆和 Agent | `routes/chat.py::stream_message`、`routes/helpers.py::sse` |
```

- [ ] **Step 2: `ai-context/SPEC.md` line 484**

Change `backend/app/api/routes.py::stream_message` to `backend/app/api/routes/chat.py::stream_message` (rest of the sentence unchanged).

- [ ] **Step 3: `ai-context/SPEC.md` line 607**

Change `3. `backend/app/api/routes.py::stream_message`：完整对话主链路。` to `3. `backend/app/api/routes/chat.py::stream_message`：完整对话主链路。`

- [ ] **Step 4: `ai-context/SPEC.md` changelog — add a new entry**

Find the changelog section (the same section that carries the `4.6 / 2026-09-09` entry documenting the `memory_tasks.py` split) and append:

```
- **4.7 / 2026-09-10**：`backend/app/api/routes.py` 拆分为 `routes/` 包（`chat.py` + `agent_runs.py` + `scenarios.py` + `memories.py` + `conversations.py` + `feedback.py` + `index.py` + `helpers.py`），外部接口 `router` 和导入路径 `backend.app.api.routes` 不变，`tests/test_scenario_cleanup.py` 的 `SessionLocal` monkeypatch 已同步改为 `scenarios.SessionLocal`，全量测试 176 passed 验证通过。
```

- [ ] **Step 5: `resume/architecture.md`**

Line 11: change `- `backend/app/api/routes.py::stream_message`：咨询主入口。` to `- `backend/app/api/routes/chat.py::stream_message`：咨询主入口。`

Line 69: change `| API | 所有权校验、短事务、SSE 协调、回答与任务持久化 | `backend/app/api/routes.py` |` to `| API | 所有权校验、短事务、SSE 协调、回答与任务持久化 | `backend/app/api/routes/` |`

- [ ] **Step 6: `resume/agent-persistence-code-guide.md`**

Line 41: change `| 3 | `backend/app/api/routes.py:297-405` | 创建任务、查询、取消和事件订阅 API |` to `| 3 | `backend/app/api/routes/agent_runs.py` | 创建任务、查询、取消和事件订阅 API |` (drop the stale line-anchor — the whole file now equals this scope, so a range is redundant).

Line 84: change `前端调用 `POST /api/conversations/{conversation_id}/runs`。入口位于 `backend/app/api/routes.py:297-328`。` to `前端调用 `POST /api/conversations/{conversation_id}/runs`。入口位于 `backend/app/api/routes/agent_runs.py:18-49`（`create_agent_run`）。`

Line 327: change `事件订阅入口位于 `backend/app/api/routes.py:357-405`：` to `事件订阅入口位于 `backend/app/api/routes/agent_runs.py:78-127`（`agent_run_events`）：`

Line 360: change `取消入口为 `POST /api/agent-runs/{run_id}/cancel`，路由位于 `backend/app/api/routes.py:349-354`。` to `取消入口为 `POST /api/agent-runs/{run_id}/cancel`，路由位于 `backend/app/api/routes/agent_runs.py:70-75`（`cancel_agent_run`）。`

- [ ] **Step 7: `resume/memory.md`**

Line 44: change `每轮对话在 `backend/app/api/routes.py::_prepare_chat` 中先保存用户消息；Agent 完成后再通过 `_save_assistant` 保存助手回答。` to `每轮对话在 `backend/app/api/routes/chat.py::_prepare_chat` 中先保存用户消息；Agent 完成后再通过同文件的 `_save_assistant` 保存助手回答。`

Line 522: change `| `backend/app/api/routes.py` | `_prepare_chat` | 保存问题并创建记忆快照 |` to `| `backend/app/api/routes/chat.py` | `_prepare_chat` | 保存问题并创建记忆快照 |`

Line 523: change `| `backend/app/api/routes.py` | `stream_message` | 回答完成后创建记忆任务 |` to `| `backend/app/api/routes/chat.py` | `stream_message` | 回答完成后创建记忆任务 |`

- [ ] **Step 8: `resume/review-findings.md` line 151**

Change `| 1 | `backend/app/api/routes.py` | 看清用户上下文、SSE、短事务和主链路 |` to `| 1 | `backend/app/api/routes/` | 看清用户上下文、SSE、短事务和主链路 |`

- [ ] **Step 9: `resume/sse-code-guide.md`**

Line 38: change `2. `backend/app/api/routes.py::create_agent_run`：任务如何创建。` to `2. `backend/app/api/routes/agent_runs.py::create_agent_run`：任务如何创建。`

Line 40: change `4. `backend/app/api/routes.py::agent_run_events`：事件如何转换为 SSE。` to `4. `backend/app/api/routes/agent_runs.py::agent_run_events`：事件如何转换为 SSE。`

Line 63: change `后端入口是 [routes.py](../backend/app/api/routes.py#L297) 的 `create_agent_run()`：` to `后端入口是 [agent_runs.py](../backend/app/api/routes/agent_runs.py) 的 `create_agent_run()`：` (drop the `#Lxxx` anchor — it pointed at the old file's line number and would now be wrong; the new file name alone is the accurate pointer, consistent with not hand-maintaining line anchors across refactors).

Line 170: change `订阅入口是 [routes.py](../backend/app/api/routes.py#L357) 的 `agent_run_events()`。` to `订阅入口是 [agent_runs.py](../backend/app/api/routes/agent_runs.py) 的 `agent_run_events()`。`

Line 234: change `[routes.py](../backend/app/api/routes.py#L398) 返回：` to `[agent_runs.py](../backend/app/api/routes/agent_runs.py) 返回：`

Line 360: change `后端取消入口是 [routes.py](../backend/app/api/routes.py#L349)。Worker 在 Graph 事件边界检查 `cancel_requested`；被取消后写 `interrupted` 状态和 `message_end`。` to `后端取消入口是 [agent_runs.py](../backend/app/api/routes/agent_runs.py)。Worker 在 Graph 事件边界检查 `cancel_requested`；被取消后写 `interrupted` 状态和 `message_end`。`

Line 382: change `项目仍保留 [routes.py](../backend/app/api/routes.py#L562) 的：` to `项目仍保留 [chat.py](../backend/app/api/routes/chat.py) 的：`

- [ ] **Step 10: `resume/concurrency.md` line 36**

Change `` `backend/app/api/routes.py::stream_message` 在 `async with concurrency.slot(identity)` 成功后才调用 `_prepare_chat`：`` to `` `backend/app/api/routes/chat.py::stream_message` 在 `async with concurrency.slot(identity)` 成功后才调用 `_prepare_chat`：``

- [ ] **Step 11: `resume/api-chat.md` line 7**

Change `所有业务接口位于 `backend/app/api/routes.py::router`，统一前缀 `/api`。` to `所有业务接口位于 `backend/app/api/routes/__init__.py::router`（各子模块的 `router` 经 `include_router` 组合而成），统一前缀 `/api`。`

- [ ] **Step 12: `resume/langsmith.md`**

Line 126: change `- `backend/app/api/routes.py::stream_message`` to `- `backend/app/api/routes/chat.py::stream_message``

Line 136: change `- `backend/app/api/routes.py::message_feedback`` to `- `backend/app/api/routes/feedback.py::message_feedback``

- [ ] **Step 13: Final grep sweep**

```bash
grep -rn "backend/app/api/routes\.py" ai-context resume README.md 2>/dev/null
```

Expected: no output. (Historical spec/plan documents under `docs/superpowers/specs/` and `docs/superpowers/plans/`, and the dated learning notes under `docs/agent-learning/`, are excluded from this sweep — they are immutable records of decisions made before this decomposition and are intentionally left referencing the old path, matching the precedent set when `graph.py` and `memory_tasks.py` were decomposed.)

- [ ] **Step 14: Run the full test suite one more time (docs changes should not affect it, but confirm)**

Run: `/opt/anaconda3/envs/LawStation/bin/python -m pytest tests -q`
Expected: `176 passed`.

- [ ] **Step 15: Commit**

```bash
git add ai-context/SPEC.md resume/architecture.md resume/agent-persistence-code-guide.md resume/memory.md resume/review-findings.md resume/sse-code-guide.md resume/concurrency.md resume/api-chat.md resume/langsmith.md
git commit -m "docs: sync SPEC.md and resume/ with routes/ package split"
```
