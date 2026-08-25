# LawStation

单进程、单端口的法律咨询 Agent：FastAPI 同时托管 React 页面、业务 API、SSE 对话与法律 RAG MCP Server。

项目架构、开发准则、技术选型、验收基线与已知缺口见 [`ai-context/SPEC.md`](ai-context/SPEC.md)。后续涉及架构边界、API、数据模型、配置或安全规则的变更，应同步更新该 Spec。

## 首次安装

项目统一使用 Conda 环境，Python、Node.js 和项目依赖都由 `environment.yml` 管理：

```bash
cd /Users/Admin1/Files/LawStation
conda env create -f environment.yml
conda activate LawStation
```

环境已存在时使用 `conda env update -f environment.yml --prune` 同步依赖。应用配置和密钥全部放在根目录 `.env`，不使用 Conda 环境变量保存业务配置。

在 `.env` 填写 `DEEPSEEK_API_KEY`。Dense 检索默认使用本机 Ollama 的 `qwen3-embedding:0.6b`，请先安装 Ollama 并下载模型：

```bash
ollama pull qwen3-embedding:0.6b
```

无需手工执行 `ollama run` 或长期保持 `ollama serve`。统一启动器会复用已运行的 Ollama；不可达时自动执行 `ollama serve`、校验精确模型标签与 digest，并通过 `/api/embed` 预热 1024 维模型。Ollama、模型或预热不可用时启动会明确失败。默认法规数据源是全量 `data/knowledge/law/law.json`。启动时会检查索引指纹：有效索引直接复用，缺失或过期时后台构建，构建期间自动使用全量 BM25，完成后热切换为 BM25 + FAISS。

如需测试或演示，可重新生成固定的 100 条样本：

```bash
python scripts/create_law_sample.py --size 100 --seed 42
```

临时使用样本时，可将 `.env` 的 `LAW_DATA_PATH` 改为 `./data/knowledge/law/law_sample.json`；生产默认保持全量路径。

## 统一启动

```bash
conda activate LawStation
python run.py
```

启动器会在前端缺失或过期时自动安装/构建前端，确保 Ollama Embedding 可用，然后启动唯一的 Uvicorn 进程。由本次启动器创建的 Ollama 会随程序退出；启动前已存在的外部 Ollama 不受影响。访问：

启动时会自动运行 Alembic 数据库迁移。首次升级分层记忆结构前，现有 SQLite 会备份为 `data/runtime/lawstation.db.pre-memory-v2.bak`。

- 程序：http://127.0.0.1:8000
- 健康检查：http://127.0.0.1:8000/health
- API 文档：http://127.0.0.1:8000/docs
- MCP：http://127.0.0.1:8000/mcp/
- 索引状态：http://127.0.0.1:8000/api/index/status

其他启动方式：

```bash
python run.py --rebuild
python run.py --no-build
python run.py --host 0.0.0.0 --port 8000
```

独立调试 MCP 仍可使用 `python -m mcp_servers.law_rag.server`，但正常运行不需要第二个服务。

## Docker

```bash
docker compose up --build
```

Compose 只启动一个 `lawstation` 容器并暴露 8000 端口。容器不负责启动 Ollama，而是通过 `host.docker.internal:11434` 使用宿主机 Ollama；运行 Compose 前须在宿主机启动 Ollama 并准备好模型。

## 索引与审计日志

手工等待索引构建完成或强制重建：

```bash
python scripts/build_index.py
python scripts/build_index.py --force
```

全量建库默认按最多 8 条一批调用 Ollama `/api/embed`，不消耗 DashScope token。索引指纹包含 Ollama provider、模型标签、模型 digest、查询指令版本和数据/切分配置；仅同指纹批次可以恢复。已成功批次保存在指纹专属 staging 目录，遇到中断或可重试的限流、超时和服务错误时可在本次或下次启动继续；只有完整索引通过校验后才会原子替换当前 FAISS 索引。

查询阶段会先应用 `law_name` 过滤，再执行 BM25 与 Dense 排序，并通过 `RAG_BM25_MIN_SCORE`、`RAG_DENSE_MIN_SCORE` 和 `RAG_RRF_MIN_SCORE` 排除无效候选。引用以 `chunk_id` 为证据边界，最终只展示回答实际使用的法条片段。完全没有有效候选时正常进入 `no_match`，不会被当作工具失败。

Ollama 进程输出追加到 `data/logs/ollama.log`。若模型不存在，请先执行 `ollama pull qwen3-embedding:0.6b`，启动器不会自动下载模型。

控制台审计事件同时以 JSON Lines 追加到 `data/logs/lawstation.log`。默认单文件 20 MB、保留 10 个备份；日志只保存脱敏摘要和工具结果标识，不记录密钥或完整法条正文。

## LangSmith 追踪与评估

LangSmith 默认关闭。启用前在 `.env` 至少配置：

```dotenv
LANGSMITH_ENABLED=true
LANGSMITH_API_KEY=你的密钥
LANGSMITH_WORKSPACE_ID=你的工作区ID
LANGSMITH_ID_HASH_SECRET=一个独立的高强度随机字符串
LANGSMITH_PROJECT=lawstation
LANGSMITH_ENVIRONMENT=development
```

咨询 Graph 和记忆整理分别进入 `lawstation-<环境>-agent` 和 `lawstation-<环境>-memory` 项目。租户、用户和会话 ID 只以 HMAC 哈希发送；API Key、请求头、数据库地址和模型内部推理始终过滤。设置 `LANGSMITH_CAPTURE_CONTENT=false` 可隐藏 trace 输入输出。LangSmith 异常不会阻断咨询，`/health` 只暴露非敏感运行状态与预算。

线上排查可按本次进程开启全链路追踪，不会改写 `.env`：

```bash
python run.py --langsmith-trace-all
python run.py --langsmith-trace-all --langsmith-trace-limit 500
python run.py --no-langsmith-trace
```

全量模式启动前严格校验 Key、HMAC、Workspace 和 LangSmith 鉴权；默认最多接收 200 条咨询/记忆根 Trace。咨询的排队、记忆快照、三 Agent、MCP、BM25、Ollama Query Embedding、FAISS、RRF 和持久化属于同一根 Trace；每个后台记忆 Job 使用一个可关联的独立根 Trace。达到上限或运行期导出故障后停止新增 Trace，但咨询继续运行。

仓库提供 60 条合成 E2E 基准、30 条分层 Agent 基准，以及 100 条引用真实
`document_id/chunk_id` 的源数据派生检索集：

```bash
python scripts/create_eval_datasets.py
python scripts/create_live_retrieval_dataset.py
python scripts/seed_langsmith_datasets.py
python scripts/run_langsmith_eval.py --profile learn
python scripts/run_langsmith_eval.py --profile smoke
python scripts/run_staged_langsmith_eval.py --profile compare --plan-only
```

默认 `learn` 使用固定输出讲解确定性指标，不访问 LangSmith、DeepSeek 或 Ollama；`smoke`
只运行 6 条分层样本且 `upload_results=false`。云端 Compare 默认仅上传 30 条根 Trace、调用
10 次 Judge，执行前必须先查看 `--plan-only`，再显式增加 `--confirm-upload`。所有评测受月度
Trace、Agent 模型和 Judge 三类本地预算保护。完整概念、指标公式和失败定位方法见
`evals/LEARNING_GUIDE.md`。

`retrieval` 用于隔离比较 BM25 与 Hybrid，`component` 使用固定法规工具结果，`live` 调用当前
MCP 和真实模型。完整本地 RAG 指标以数据集哈希、索引指纹和 Git Commit 保证可复现；LangSmith
只保存固定见证样本。源数据派生集尚未经过律师人工标注，不能宣称为专家标注的法律准确率。

除 `--plan-only` 和测试专用的 `--no-report` 外，每次评测都会在新加坡时区生成独立归档目录：

```text
evals/reports/runs/YYYYMMDD-HHMMSS-ffffff-<profile>/
```

目录内保存 JSON、同名 CSV 和 `run-manifest.json`；分阶段实验还会保存
`EVAL_REPORT.md`。`latest.json` 指向最近一次运行，`latest-success.json` 只指向最近一次完整成功的运行。历史目录不会自动清理，且默认被 Git 忽略。可通过 `.env` 中的
`EVAL_REPORT_ROOT` 和 `EVAL_REPORT_TIMEZONE` 调整归档位置及时区。

## 分层记忆

- 当前案件事实只在所属会话使用；用户偏好和稳定背景可以跨该用户的会话复用。
- 合法抽取的用户偏好和案件事实在后台整理完成后自动生效，无需再次确认。
- 用户最新明确事实与有效记忆冲突时，服务端验证模型建议的替换目标并原位更新；旧内容只保留在修订审计中，不再参与上下文。
- 页面左侧“管理我的记忆”可修正或删除已生效记忆；升级前已有的待确认记录仍可确认或拒绝。
- 回答完成后由 SQLite 持久后台任务整理记忆和增量摘要，不阻塞主回答；服务重启会恢复未完成任务。
- 记忆整理使用独立的非 Thinking JSON Output 调用，不绑定或调用 MCP 工具；简单问候等没有可沉淀内容的消息会正常完成且不创建记忆。
- `MemoryTaskManager.enqueue()` 是唯一记忆整理入口；旧 `MemoryService.consolidate()` 已禁用，不能再将用户长消息原文直接沉淀为事实。

咨询 SSE 每 15 秒发送一次不可见 heartbeat，避免长模型调用期间连接被代理关闭。低风险且没有可引用法条的回答通过确定性证据边界校验后可跳过 LLM Reviewer；中高风险、存在法规依据或检索异常时仍执行完整模型复核。

## 测试

```bash
pytest -v
cd frontend && npm test && npm run build
```

演示用户只提供逻辑隔离，不是正式身份认证。`tools/` 可在明确业务需求下修改或复用，但所有改动必须遵守最小变更原则并通过对应回归测试。
