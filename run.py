import argparse
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"
DIST_INDEX = FRONTEND / "dist" / "index.html"


def frontend_sources() -> list[Path]:
    files = [FRONTEND / "package.json", FRONTEND / "package-lock.json", FRONTEND / "vite.config.ts", FRONTEND / "tsconfig.json", FRONTEND / "index.html"]
    files.extend((FRONTEND / "src").rglob("*"))
    return [path for path in files if path.is_file()]


def frontend_is_stale() -> bool:
    if not DIST_INDEX.is_file():
        return True
    built_at = DIST_INDEX.stat().st_mtime
    return any(path.stat().st_mtime > built_at for path in frontend_sources())


def build_frontend(force: bool, no_build: bool) -> None:
    stale = force or frontend_is_stale()
    if not stale:
        return
    if no_build:
        raise SystemExit("前端构建缺失或已过期；请去掉 --no-build 或先执行 npm run build")
    npm = shutil.which("npm")
    if npm is None:
        raise SystemExit("需要构建前端，但未找到 npm。请安装 Node.js 20+，或提供有效的 frontend/dist")
    if not (FRONTEND / "node_modules").is_dir():
        install = [npm, "ci"] if (FRONTEND / "package-lock.json").is_file() else [npm, "install"]
        subprocess.run(install, cwd=FRONTEND, check=True)
    subprocess.run([npm, "run", "build"], cwd=FRONTEND, check=True)
    if not DIST_INDEX.is_file():
        raise SystemExit("前端构建命令已结束，但 frontend/dist/index.html 不存在")


def parse_args(settings):
    parser = argparse.ArgumentParser(description="启动 LawStation 单进程应用")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--rebuild", action="store_true", help="强制重新构建前端")
    group.add_argument("--no-build", action="store_true", help="禁止自动构建前端")
    parser.add_argument("--host", default=settings.app_host)
    parser.add_argument("--port", type=int, default=settings.app_port)
    scenarios = parser.add_mutually_exclusive_group()
    scenarios.add_argument(
        "--test-scenarios",
        action="store_true",
        help="本次进程启用前端场景观察模式",
    )
    scenarios.add_argument(
        "--no-test-scenarios",
        action="store_true",
        help="本次进程强制关闭前端场景观察模式",
    )
    tracing = parser.add_mutually_exclusive_group()
    tracing.add_argument(
        "--langsmith-trace-all",
        action="store_true",
        help="本次进程全量上传咨询与记忆 Trace（受会话上限保护）",
    )
    tracing.add_argument(
        "--no-langsmith-trace",
        action="store_true",
        help="本次进程强制关闭 LangSmith Trace",
    )
    parser.add_argument(
        "--langsmith-trace-limit",
        type=int,
        default=None,
        metavar="COUNT",
        help="全量 Trace 模式的本次进程根 Trace 上限（默认 200）",
    )
    return parser.parse_args()


def apply_scenario_cli(args) -> None:
    """Apply process-only scenario observer overrides without editing .env."""
    if args.test_scenarios:
        os.environ["TEST_SCENARIOS_ENABLED"] = "true"
    elif args.no_test_scenarios:
        os.environ["TEST_SCENARIOS_ENABLED"] = "false"


def preflight_scenarios(settings) -> None:
    """Validate enabled scenario datasets before external model processes start."""
    if not settings.test_scenarios_enabled:
        print("场景观察模式：未启用，可使用 --test-scenarios 开启")
        return
    from backend.app.evaluation.scenario_catalog import ScenarioCatalog

    try:
        catalog = ScenarioCatalog(settings)
    except Exception as exc:
        raise SystemExit(f"场景观察模式启动检查失败：{exc}") from exc
    datasets = catalog.datasets()
    summary = "，".join(
        f"{item['id']}（{item['sample_count']}条）" for item in datasets
    )
    print(f"场景观察模式：已启用；数据集：{summary}；入口：顶部栏 / 左侧栏")


def apply_langsmith_cli(settings, args):
    """Apply process-only tracing overrides without modifying .env."""
    if args.langsmith_trace_limit is not None and not args.langsmith_trace_all:
        raise SystemExit("--langsmith-trace-limit 只能与 --langsmith-trace-all 一起使用")
    if args.langsmith_trace_limit is not None and args.langsmith_trace_limit <= 0:
        raise SystemExit("--langsmith-trace-limit 必须大于 0")
    if args.langsmith_trace_all:
        os.environ["LANGSMITH_RUNTIME_MODE"] = "all"
        os.environ["LANGSMITH_ENABLED"] = "true"
        os.environ["LANGSMITH_STRICT_STARTUP"] = "true"
        os.environ["LANGSMITH_TRACE_SAMPLE_RATE"] = "1.0"
        os.environ["LANGSMITH_CAPTURE_CONTENT"] = "true"
        os.environ["LANGSMITH_SESSION_TRACE_LIMIT"] = str(
            args.langsmith_trace_limit or settings.langsmith_session_trace_limit
        )
    elif args.no_langsmith_trace:
        os.environ["LANGSMITH_RUNTIME_MODE"] = "off"
        os.environ["LANGSMITH_ENABLED"] = "false"
        os.environ["LANGSMITH_STRICT_STARTUP"] = "false"


def main() -> None:
    os.chdir(ROOT)
    from backend.app.core.config import get_settings
    from backend.app.core.ollama import OllamaProcessManager
    from backend.app.core.tei import TEIRerankerProcessManager

    settings = get_settings()
    args = parse_args(settings)
    apply_scenario_cli(args)
    apply_langsmith_cli(settings, args)
    get_settings.cache_clear()
    settings = get_settings()
    preflight_scenarios(settings)
    if settings.langsmith_runtime_mode == "all":
        from backend.app.observability.langsmith import validate_langsmith_startup

        try:
            validate_langsmith_startup(settings)
        except Exception as exc:
            raise SystemExit(f"LangSmith 全量 Trace 启动检查失败：{exc}") from exc
        print(
            "LangSmith 全量 Trace 已启用："
            f"project={settings.langsmith_project}，"
            f"session_limit={settings.langsmith_session_trace_limit}，"
            "content=脱敏完整内容"
        )
    elif settings.langsmith_runtime_mode == "off":
        print("LangSmith Trace 已由启动参数关闭")
    build_frontend(args.rebuild, args.no_build)
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("缺少 Python 依赖，请先执行 `conda env update -f environment.yml --prune`") from exc
    ollama = OllamaProcessManager(settings) if settings.embedding_provider == "ollama" else None
    tei_reranker = (
        TEIRerankerProcessManager(settings)
        if settings.rag_rerank_enabled and settings.rag_rerank_provider == "tei"
        else None
    )
    try:
        if ollama is not None:
            try:
                runtime = ollama.ensure_ready()
            except Exception as exc:
                raise SystemExit(f"Ollama 启动检查失败：{exc}") from exc
            ownership = "由 LawStation 管理" if runtime.managed else "复用已有服务"
            print(f"Ollama 已就绪：{runtime.model}（{ownership}）")
            if settings.rag_rerank_enabled and settings.rag_rerank_provider == "ollama":
                if runtime.reranker_available:
                    print(f"Ollama Reranker 已就绪：{runtime.reranker_model}")
                else:
                    print(
                        "Ollama Reranker 未就绪，将降级使用 RRF："
                        f"{runtime.reranker_error}"
                    )
        if tei_reranker is not None:
            try:
                reranker_runtime = tei_reranker.ensure_ready()
            except Exception as exc:
                if settings.rag_rerank_required:
                    raise SystemExit(f"TEI Reranker 启动检查失败：{exc}") from exc
                print(f"TEI Reranker 未就绪，将降级使用 RRF：{exc}")
            else:
                ownership = (
                    "由 LawStation 管理"
                    if reranker_runtime.managed else "复用已有服务"
                )
                print(
                    "TEI Reranker 已就绪："
                    f"{reranker_runtime.model}@{reranker_runtime.model_sha[:12]}"
                    f"（{ownership}）"
                )
        print(f"LawStation 正在启动：http://127.0.0.1:{args.port}（配置来源：.env）")
        uvicorn.run("backend.app.main:app", host=args.host, port=args.port, reload=False)
    finally:
        if tei_reranker is not None:
            tei_reranker.close()
        if ollama is not None:
            ollama.close()


if __name__ == "__main__":
    main()
