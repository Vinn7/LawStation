"""FastAPI 组装入口及 Agent/RAG/持久化组件的应用生命周期。"""

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
from backend.app.agent.skills import SkillRegistry
from backend.app.api.routes import router
from backend.app.core.config import get_settings
from backend.app.core.logging import audit, setup_logging
from backend.app.db.migrations import upgrade_database
from backend.app.db.models import Tenant, User
from backend.app.db.session import Base, SessionLocal, engine
from backend.app.evaluation.scenario_catalog import ScenarioCatalog
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
    """按依赖顺序创建应用级共享对象，并以相反方向安全关闭。

    Registry、Provider、Runtime、Checkpointer 和 Worker 均应用级共享；每个用户的
    State、Invocation Context、数据库 Session 与 MCP 执行 Session 仍按请求隔离。
    """

    setup_logging()
    audit("application.starting", status="starting")
    observability = LangSmithObservability()
    observability.ensure_startup_ready()
    app.state.langsmith_observability = observability
    mcp_app.bind(observability)
    initialize_database()
    app.state.scenario_catalog = ScenarioCatalog()
    await initialize_engine()
    registry = MCPToolRegistry(observability=observability)
    skill_registry = SkillRegistry()
    provider = LLMProvider()
    app.state.mcp_tool_registry = registry
    app.state.skill_registry = skill_registry
    # AsyncSqliteSaver 必须覆盖 AgentRuntime/AgentRunWorker 的完整生命周期，确保
    # Graph 执行和恢复期间连接始终有效，退出时再统一关闭。
    async with checkpoint_saver(get_settings()) as checkpointer:
        app.state.langgraph_checkpointer = checkpointer
        app.state.agent_runtime = AgentRuntime(
            registry,
            provider,
            observability=observability,
            checkpointer=checkpointer,
            skill_registry=skill_registry,
        )
        app.state.agent_concurrency = AgentConcurrencyManager()
        app.state.memory_tasks = MemoryTaskManager(provider, observability=observability)
        app.state.agent_runs = AgentRunManager(
            app.state.agent_runtime,
            app.state.agent_concurrency,
            app.state.memory_tasks,
            observability,
        )
        # 先启动回答后记忆 Worker，再启动 AgentRun Worker；后者完成回答时才能可靠
        # enqueue MemoryJob。两个 Worker 都不持有长生命周期 SQLAlchemy Session。
        await app.state.memory_tasks.start()
        await app.state.agent_runs.start()
        # 内嵌 MCP ASGI 与 FastAPI 同进程，但 Agent 仍通过 HTTP 协议调用；Session
        # Manager 必须运行后，首次懒发现和后续 Tool Call 才能建立短期 MCP Session。
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
    skills = getattr(app.state, "skill_registry", None)
    scenarios = getattr(app.state, "scenario_catalog", None)
    return {"status": "ok", "mcp": "/mcp/", "index_status": index["status"], "dense_enabled": index.get("dense_enabled", False), "langsmith": langsmith.status() if langsmith else {"enabled": False, "export_status": "uninitialized"}, "skills": skills.status_dict() if skills else {"enabled": False, "status": "uninitialized"}, "test_scenarios": scenarios.status() if scenarios else {"enabled": False, "status": "uninitialized"}}


app.mount("/mcp", mcp_app)
if not (FRONTEND_DIST / "index.html").is_file():
    raise RuntimeError("前端尚未构建，请使用 `python run.py` 自动构建后启动")
app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
