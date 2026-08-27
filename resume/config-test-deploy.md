# LawStation 配置、测试、脚本与部署

## 1. 配置系统

`backend/app/core/config.py::Settings` 基于 `pydantic-settings`，固定从仓库根 `.env` 读取，未知字段忽略；`get_settings` 使用 `lru_cache` 保证进程内配置稳定。

配置分组：

| 分组 | 代表变量 |
|---|---|
| DeepSeek | `DEEPSEEK_API_KEY/BASE_URL/MODEL` |
| Embedding | `EMBEDDING_PROVIDER/MODEL/DIMENSION` |
| Ollama | `OLLAMA_AUTO_START/BASE_URL/BATCH_SIZE/KEEP_ALIVE/MAX_LOADED_MODELS` |
| TEI Reranker | `RAG_RERANK_BASE_URL/AUTO_START/COMMAND/MODEL_CACHE/MODEL_REVISION/STARTUP_TIMEOUT` |
| 数据 | `DATABASE_URL/LAW_DATA_PATH/INDEX_DIR` |
| MCP/Agent | `MCP_LAW_SERVER_URL`、工具/模型调用限制 |
| 并发 | global/per-user/per-conversation/queue timeout |
| 记忆 | context budget、压缩阈值、Worker、Memory LLM |
| RAG | BM25/Dense/RRF 阈值、retrieval mode、TEI BGE 候选数/批次/超时/降级 |
| 日志 | 目录、级别、轮转和摘要长度 |
| LangSmith | `config/all/off` 运行模式、Session 根 Trace 上限、采样、隐私、Judge 和生产 Trace 月度保护 |
| Eval | 实际用量账本、缓存和时间戳报告目录；不设月度硬上限 |
| Durable Run | `AGENT_RUN_LEASE_SECONDS/RECOVERY_MAX_ATTEMPTS/WORKER_POLL_SECONDS` |
| LangGraph | `LANGGRAPH_CHECKPOINT_ENABLED/PATH/RETENTION_DAYS/STRICT_MSGPACK` |
| Retrieval Gate | `RAG_MATCH_GATE_ENABLED/CONFIG_PATH/REQUIRED` |

密钥只应存在 `.env`；`.env.example` 提供空值模板。前端使用同源 `/api`，没有构建期 API Key。

## 2. Python 与 Node 环境

`environment.yml` 定义 Conda 环境：

```text
name: LawStation
python=3.12
nodejs=22
pip install -e .[dev]
```

`pyproject.toml` 明确 setuptools build system，并只打包 `backend*` 与 `mcp_servers*`。`tools/`、frontend 和 data 不进入 Python package 自动发现。

## 3. 正式构建与启动

本地：

```bash
conda env create -f environment.yml
conda activate LawStation
brew install text-embeddings-inference
python run.py
```

线上全链路诊断使用 `python run.py --langsmith-trace-all [--langsmith-trace-limit N]`；强制关闭使用 `--no-langsmith-trace`。CLI 仅覆盖当前进程，`all` 默认最多 200 条咨询与记忆根 Trace并绕过月度生产采样预算，子 Span 不重复扣减。全量模式要求 `.env` 配置 Key、Workspace ID 和 HMAC 密钥。

前端构建由启动器自动判断，也可单独执行：

```bash
cd frontend
npm ci
npm run build
```

本次 Review 遵守 SPEC，没有启动服务或 Ollama。

## 4. Docker

`Dockerfile`：

1. Node 22 Alpine 执行可复现前端构建。
2. Python 3.11 slim 安装后端 package。
3. 复制 Alembic、data 和前端 dist。
4. 最终镜像不包含 Node/node_modules。
5. 执行 `python run.py --no-build --host 0.0.0.0 --port 8000`。

`docker-compose.yml` 只有 `lawstation` 一个服务，数据目录 bind mount，并通过 `host.docker.internal:11434/8081` 使用宿主 Ollama 和 TEI；容器不自动启动推理进程。

## 5. 脚本职责

| 脚本 | 作用 | 外部资源 |
|---|---|---|
| `scripts/create_law_sample.py` | 固定种子生成 100 条样本 | 无 |
| `scripts/build_index.py` | 手工等待完整索引构建 | Ollama/DashScope，取决配置 |
| `scripts/create_eval_datasets.py` | 生成本地 Agent/回答数据集 | 无或按脚本输入 |
| `scripts/create_live_retrieval_dataset.py` | 从真实法规 ID 构建检索集 | 本地法规 |
| `scripts/create_resume_challenge_datasets.py` | 保留旧300条并扩充至600条候选；生成/校验；全量Hybrid双干扰资格冻结与25条checkpoint | prepare/validate无外部服务；仅新增批次用Codex；资格检查使用Ollama且禁用BGE |
| `scripts/seed_langsmith_datasets.py` | 上传数据集 | LangSmith |
| `scripts/run_langsmith_eval.py` | 单次 learn/smoke/compare/release | Profile 决定 |
| `scripts/run_staged_langsmith_eval.py` | 分阶段对比与实际用量记录 | 需要显式上传确认，不受月度累计值阻断 |
| `scripts/run_resume_rag_challenge_eval.py` | 回归测试、数据校验、资格冻结、通用/Dense/BGE 消融、6类 Agent 冒烟、硬门禁与简历结论归档 | 本地 Ollama + TEI；冒烟少量 DeepSeek；不上传 LangSmith、不调用 Judge |

2026-08-26实跑说明：六组RAG消融完成后，旧月度上限曾在外部调用前阻止6条Agent冒烟。移除评测
硬上限后已补跑六类Fixture，项目配置门禁通过。原始`run-manifest.json`仍保留失败审计，恢复事实
记录在`POST_RECOVERY_SUMMARY.json`；不得通过覆盖历史manifest伪造一次性成功运行。

LangSmith 评测依赖声明为 `langsmith[vcr]`。这是因为评测脚本默认配置
`LANGSMITH_TEST_CACHE`，云端 `aevaluate` 也会初始化 VCR 缓存上下文；缺少 `vcrpy` 时会在
实验容器创建后、Target 真正运行前失败。缓存只允许用于 Learn/Smoke；正式上传的
Compare/Release 会临时清除缓存环境变量，完成后再恢复，确保性能数据来自真实调用。

检索评测通过 `--rerank-mode off|on` 对比 RRF 基线与 TEI BGE Cross-Encoder。
该参数默认 `off`，用于维持既有实验基线；正式 Candidate 报告还会记录精排覆盖率、
降级率、候选数、可获得的 token 元数据、p50/p95、TEI `model_sha` 与
`ranking_version`，并可通过门禁拒绝“表面开启但实际降级”的结果。

量化套件默认完整执行；`--plan-only` 只输出数据准备状态、1,200次正式检索、最多600次资格检索、
6条 Agent 样本和最坏36次 Agent 模型调用，不创建报告或启动服务。排障时可用
`--skip-unit-tests`、`--skip-agent-smoke` 或 `--no-auto-qualify` 缩小范围，但这些运行必须在
manifest 中保留实际边界，不能冒充完整套件成绩。

`tools/*.py` 当前没有被 backend、mcp_servers、scripts 或 tests 导入，是参考/遗留能力，不属于正式主链路。

## 6. Python 测试体系

Pytest 配置位于 `pyproject.toml`，测试均在 `tests/`。

| 模块 | 主要测试文件 |
|---|---|
| Agent/Graph/MCP Registry | `test_agent_runtime.py` |
| 并发/SSE/SQLite PRAGMA | `test_concurrency.py` |
| RAG 索引与混合检索 | `test_index_manager.py` |
| Embedding Provider | `test_embeddings.py` |
| Ollama 生命周期 | `test_ollama_manager.py` |
| TEI 生命周期 | `test_tei_manager.py` |
| TEI BGE Reranker | `test_reranker.py`、`test_index_manager.py` |
| 记忆、替换与摘要 | `test_memory.py` |
| 所有权隔离 | `test_isolation.py` |
| Alembic | `test_migrations.py` |
| JSONL 审计 | `test_audit_logging.py` |
| LangSmith | `test_langsmith_observability.py`、`test_agent_runtime.py`（预算、开关、MCP header 传播） |
| 评测用量/报告 | `test_eval_resource_budget.py`、`test_eval_reporting.py`、`test_challenge_datasets.py` |
| 启动器 | `test_run.py` |

绝大多数外部服务通过 fake/mock 隔离，不应在常规测试中消耗模型额度。

## 7. 前端测试体系

Vitest + React Testing Library 覆盖：

- SSE 分块与 heartbeat comment；
- 组件状态和键盘行为；
- 用户/会话切换与后台 stream；
- 记忆面板读取、修正和删除。

命令：

```bash
cd frontend
npm test
npm run build
```

## 8. 评测与普通测试的区别

- 单元测试：验证确定性代码行为，默认无网络。
- `--profile learn`：基于 fixture 解释确定性 evaluator，无模型和 Trace。
- `--profile smoke`：真实 Agent 本地运行，不上传 Trace，默认无 Judge。
- `compare/release`：显式确认后才允许上传和消耗预算。

不能把小样本 LangSmith Judge 分数当作生产准确率；报告必须附样本量、数据集哈希和限制。

## 9. 数据与构建产物

不应提交：

- `.env`
- `frontend/node_modules`
- `frontend/dist`
- `data/runtime` 数据库和预算账本
- `data/logs`
- `data/indexes` 生成索引/staging
- `evals/reports/runs` 历史运行报告（保留 `.gitkeep`）

`law.json` 和 `law_sample.json` 属于知识源文件，不应被建库逻辑修改。

## 10. Review 校验策略

本轮文档 Review 采用：

- 源码和 import 静态核对；
- 配置、迁移、数据模型和测试交叉验证；
- 路径与 symbol 存在性检查；
- Markdown 本地链接检查；
- `resume/` 敏感词检查；
- 不启动服务、不调用外部模型。

## 11. 当前工程风险

- 前端依赖使用 `latest`，长期升级时可能出现不可预期大版本变化。
- Python 依赖多为范围锁定，没有独立的完全冻结 lock 文件。
- Docker Python 3.11 与 Conda Python 3.12 不完全一致。
- 没有仓库内 CI workflow，测试门禁依赖人工运行。
- `.env` 中非密钥配置与 CLI 参数存在端口联动问题。
- 生产部署没有 TLS、认证、限流或多实例协调。

## 12. 新增依赖、迁移与测试

- Python 新增 `langgraph-checkpoint-sqlite>=3,<4` 与 `aiosqlite>=0.20,<1`。
- Alembic head 为 `20260826_05`，创建 `agent_runs/agent_run_events` 与同会话 active 唯一索引；升级前备份名后缀为 `.pre-agent-runs.bak`。
- Checkpoint 数据库与业务库分离为 `data/runtime/langgraph-checkpoints.db`，使用 `JsonPlusSerializer(pickle_fallback=False)`。
- 新增 `tests/test_agent_runs.py` 与 `tests/test_retrieval_confidence.py`；本次离线后端全量回归为 144 passed，前端为 13 passed，生产构建通过。
- 冻结数据由 `scripts/create_accuracy_datasets.py` 生成；真实阈值校准会调用本地检索服务，不在普通测试或开发完成后自动执行。
