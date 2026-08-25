# LawStation 配置、测试、脚本与部署

## 1. 配置系统

`backend/app/core/config.py::Settings` 基于 `pydantic-settings`，固定从仓库根 `.env` 读取，未知字段忽略；`get_settings` 使用 `lru_cache` 保证进程内配置稳定。

配置分组：

| 分组 | 代表变量 |
|---|---|
| DeepSeek | `DEEPSEEK_API_KEY/BASE_URL/MODEL` |
| Embedding | `EMBEDDING_PROVIDER/MODEL/DIMENSION` |
| Ollama | `OLLAMA_AUTO_START/BASE_URL/BATCH_SIZE/KEEP_ALIVE` |
| 数据 | `DATABASE_URL/LAW_DATA_PATH/INDEX_DIR` |
| MCP/Agent | `MCP_LAW_SERVER_URL`、工具/模型调用限制 |
| 并发 | global/per-user/per-conversation/queue timeout |
| 记忆 | context budget、压缩阈值、Worker、Memory LLM |
| RAG | BM25/Dense/RRF 阈值和 retrieval mode |
| 日志 | 目录、级别、轮转和摘要长度 |
| LangSmith | `config/all/off` 运行模式、Session 根 Trace 上限、采样、隐私、Judge 和月度预算 |
| Eval | 月度预算、缓存和时间戳报告目录 |

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

`docker-compose.yml` 只有 `lawstation` 一个服务，数据目录 bind mount，并通过 `host.docker.internal` 使用宿主 Ollama。

## 5. 脚本职责

| 脚本 | 作用 | 外部资源 |
|---|---|---|
| `scripts/create_law_sample.py` | 固定种子生成 100 条样本 | 无 |
| `scripts/build_index.py` | 手工等待完整索引构建 | Ollama/DashScope，取决配置 |
| `scripts/create_eval_datasets.py` | 生成本地 Agent/回答数据集 | 无或按脚本输入 |
| `scripts/create_live_retrieval_dataset.py` | 从真实法规 ID 构建检索集 | 本地法规 |
| `scripts/seed_langsmith_datasets.py` | 上传数据集 | LangSmith |
| `scripts/run_langsmith_eval.py` | 单次 learn/smoke/compare/release | Profile 决定 |
| `scripts/run_staged_langsmith_eval.py` | 受预算保护的分阶段对比 | 需要显式上传确认 |

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
| 记忆、替换与摘要 | `test_memory.py` |
| 所有权隔离 | `test_isolation.py` |
| Alembic | `test_migrations.py` |
| JSONL 审计 | `test_audit_logging.py` |
| LangSmith | `test_langsmith_observability.py`、`test_agent_runtime.py`（预算、开关、MCP header 传播） |
| 评测预算/报告 | `test_eval_resource_budget.py`、`test_eval_reporting.py` |
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
