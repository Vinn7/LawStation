from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.agent.checkpoint import checkpoint_saver
from backend.app.agent.concurrency import AgentConcurrencyManager
from backend.app.agent.provider import LLMProvider
from backend.app.agent.registry import MCPToolRegistry
from backend.app.agent.runtime import AgentRuntime
from backend.app.api.routes import router
from backend.app.core.config import get_settings
from backend.app.core.logging import audit, setup_logging
from backend.app.db.migrations import upgrade_database
from backend.app.db.models import Tenant, User
from backend.app.db.session import Base, SessionLocal, engine
from backend.app.observability import LangSmithObservability
from backend.app.services.agent_runs import AgentRunManager
from backend.app.services.memory_tasks import MemoryTaskManager
from mcp_servers.law_rag.server import (
    close_engine,
    get_index_status,
    initialize_engine,
    mcp,
    mcp_app,
)

ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIST = ROOT / "frontend" / "dist"


def initialize_database() -> None:
    (ROOT / "data" / "runtime").mkdir(parents=True, exist_ok=True)
    upgrade_database()
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        if db.query(Tenant).count() == 0:
            tenant = Tenant(id="00000000-0000-0000-0000-000000000001", name="演示租户")
            db.add(tenant)
            db.add_all([
                User(id="10000000-0000-0000-0000-000000000001", tenant_id=tenant.id, name="张三"),
                User(id="10000000-0000-0000-0000-000000000002", tenant_id=tenant.id, name="李四"),
            ])
            db.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    audit("application.starting", status="starting")
    observability = LangSmithObservability()
    observability.ensure_startup_ready()
    app.state.langsmith_observability = observability
    mcp_app.bind(observability)
    initialize_database()
    await initialize_engine()
    registry = MCPToolRegistry(observability=observability)
    provider = LLMProvider()
    app.state.mcp_tool_registry = registry
    async with checkpoint_saver(get_settings()) as checkpointer:
        app.state.langgraph_checkpointer = checkpointer
        app.state.agent_runtime = AgentRuntime(
            registry,
            provider,
            observability=observability,
            checkpointer=checkpointer,
        )
        app.state.agent_concurrency = AgentConcurrencyManager()
        app.state.memory_tasks = MemoryTaskManager(provider, observability=observability)
        app.state.agent_runs = AgentRunManager(
            app.state.agent_runtime,
            app.state.agent_concurrency,
            app.state.memory_tasks,
            observability,
        )
        await app.state.memory_tasks.start()
        await app.state.agent_runs.start()
        async with mcp.session_manager.run():
            audit("application.started", status="ready")
            try:
                yield
            finally:
                await app.state.agent_runs.close()
                await app.state.memory_tasks.close()
                await app.state.agent_runtime.close()
                mcp_app.bind(None)
                await observability.close()
                await close_engine()
                audit("application.stopped", status="stopped")


app = FastAPI(title="LawStation API", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Mcp-Session-Id"],
)
app.include_router(router)


@app.get("/health")
def health():
    index = get_index_status()
    langsmith = getattr(app.state, "langsmith_observability", None)
    return {"status": "ok", "mcp": "/mcp/", "index_status": index["status"], "dense_enabled": index.get("dense_enabled", False), "langsmith": langsmith.status() if langsmith else {"enabled": False, "export_status": "uninitialized"}}


app.mount("/mcp", mcp_app)
if not (FRONTEND_DIST / "index.html").is_file():
    raise RuntimeError("前端尚未构建，请使用 `python run.py` 自动构建后启动")
app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
