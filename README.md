# LawStation

单进程、单端口的法律咨询 Agent：FastAPI 同时托管 React 页面、业务 API、SSE 对话与法律 RAG MCP Server。

运行时内置 4 个可插拔领域 Skill：案情结构化、证据审查、程序路线和文书就绪检查。Case Analyst 只建议 Skill ID，服务端 `SkillRegistry` 校验白名单、角色、工具权限和最多 2 个组合，选中后才按需加载完整指令；Skill 关闭或一般执行失败时，原三 Agent 链路仍可运行。

项目架构、开发准则、技术选型、验收基线与已知缺口见 [`ai-context/SPEC.md`](ai-context/SPEC.md)。后续涉及架构边界、API、数据模型、配置或安全规则的变更，应同步更新该 Spec。

## 首次安装

项目统一使用 Conda 环境，Python、Node.js 和项目依赖都由 `environment.yml` 管理：

```bash
cd /Users/Admin1/Files/LawStation
conda env create -f environment.yml
conda activate LawStation
```

环境已存在时使用 `conda env update -f environment.yml --prune` 同步依赖。应用配置和密钥全部放在根目录 `.env`，不使用 Conda 环境变量保存业务配置。

在 `.env` 填写 `DEEPSEEK_API_KEY`。Dense 检索默认使用本机 Ollama 的 `qwen3-embedding:0.6b`，精排默认使用本机 TEI 的 `BAAI/bge-reranker-v2-m3`。请先安装运行时并下载 Embedding 模型：

```bash
ollama pull qwen3-embedding:0.6b
brew install text-embeddings-inference
```

无需手工执行 `ollama run` 或长期保持 `ollama serve`。统一启动器会复用已运行的 Ollama；不可达时自动执行 `ollama serve`、校验精确模型标签与 digest，并通过 `/api/embed` 预热 1024 维模型。Ollama、模型或预热不可用时启动会明确失败。默认法规数据源是全量 `data/knowledge/law/law.json`。启动时会检查索引指纹：有效索引直接复用，缺失或过期时后台构建，构建期间自动使用全量 BM25，完成后热切换为 BM25 + FAISS。

如需测试或演示，可重新生成固定的 100 条样本：

```bash
python scripts/create_law_sample.py --size 100 --seed 42
```

项目还提供版本化的多轮 Agent 流程样例。18 类确定性蓝图约束用户切换、消息发送、取消、SSE 重连、记忆检查和 Skill 断言，DeepSeek 只负责生成合成案件话术，不能生成测试动作或权限。当前已冻结 36 条样例：

```text
evals/conversations/lawstation-dialogue-scenarios-v1.jsonl
```

重新准备或生成时需显式执行，不会随应用启动自动调用模型：

```bash
python scripts/generate_conversation_scenarios.py prepare
python scripts/generate_conversation_scenarios.py generate
python scripts/generate_conversation_scenarios.py validate
python scripts/generate_conversation_scenarios.py freeze
```

该数据集由 DeepSeek 生成并经过 Schema、敏感信息、Prompt Injection、动作白名单、Skill 权限和法规来源泄漏校验；它不是真实用户数据，也未经过律师人工标注。默认不加载、不执行。

如需逐步观察真实 Agent 链路，推荐使用本次进程启动参数：

```bash
python run.py --test-scenarios
```

也可以在 `.env` 显式设置（不推荐长期保持开启）：

```dotenv
TEST_SCENARIOS_ENABLED=true
TEST_SCENARIO_DATA_PATHS=["./evals/conversations/lawstation-dialogue-scenarios-v1.jsonl"]
TEST_SCENARIO_STEP_TIMEOUT_SECONDS=60
```

启动器会在 Ollama、TEI 和 Uvicorn 之前校验白名单数据集，并打印数据集与样本数。启动后可从顶部栏或左侧栏进入“场景观察”。如需强制覆盖 `.env` 关闭本次进程，可使用 `python run.py --no-test-scenarios`。每次点击只执行一个步骤，会为 Actor 创建独立的 `[场景]` 会话，并可手工清理。该模式使用真实 Agent/MCP/RAG/Skill/记忆链路，不注入离线 Fixture；依赖 Fixture 或缺少可验证证据的预期会标为“不可判定”。它是人工观察器，不是自动连续 Runner，不能把场景数量当作通过率。

完整的前置配置、四阶段命令、checkpoint 续跑、限额控制、产物说明和失败排查见 [`resume/conversation-scenario-generation-guide.md`](resume/conversation-scenario-generation-guide.md)。

临时使用样本时，可将 `.env` 的 `LAW_DATA_PATH` 改为 `./data/knowledge/law/law_sample.json`；生产默认保持全量路径。

## 统一启动

```bash
conda activate LawStation
python run.py
```

启动器会在前端缺失或过期时自动安装/构建前端，确保 Ollama Embedding 可用，再启动或复用 TEI BGE Reranker，最后启动唯一的 Uvicorn 进程。TEI 首次启动会把模型下载到 `data/models/huggingface`。由本次启动器创建的 Ollama 和 TEI 会随程序退出；启动前已存在的外部服务不受影响。访问：

部分 TEI 版本对 BGE 的 `/info` 返回 `model_sha=null`。可在 `.env` 使用 `RAG_RERANK_MODEL_REVISION=<Hugging Face commit>` 固定版本；程序会优先使用服务返回值，其次使用该配置，最后读取项目模型缓存的 `refs/main`，并将解析结果用于 `ranking_version`。

需要在前台单独观察或调试 Reranker 时，可使用仓库中与当前 `.env` 参数一致的手动启动脚本：

```bash
cd /Users/Admin1/Files/LawStation
./scripts/start_tei_reranker.sh
```

该脚本会占用当前终端，按 `Ctrl+C` 停止。另开终端执行 `python run.py` 时，启动器会复用这个外部 TEI，LawStation 退出时不会关闭它。若修改模型、revision、端口或批处理配置，需要同步更新脚本中的命令参数。

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

Compose 只启动一个 `lawstation` 容器并暴露 8000 端口。容器不负责启动 Ollama 或 TEI，而是通过 `host.docker.internal:11434` 和 `host.docker.internal:8081` 使用宿主服务；运行 Compose 前须准备好两个服务。

## 索引与审计日志

手工等待索引构建完成或强制重建：

```bash
python scripts/build_index.py
python scripts/build_index.py --force
```

全量建库默认按最多 8 条一批调用 Ollama `/api/embed`，不消耗 DashScope token。索引指纹包含 Ollama provider、模型标签、模型 digest、查询指令版本和数据/切分配置；仅同指纹批次可以恢复。已成功批次保存在指纹专属 staging 目录，遇到中断或可重试的限流、超时和服务错误时可在本次或下次启动继续；只有完整索引通过校验后才会原子替换当前 FAISS 索引。

查询阶段会先应用 `law_name` 过滤，再执行 BM25 与 Dense 召回、RRF 融合，并把前 12 个候选一次提交给 TEI `BAAI/bge-reranker-v2-m3` 做 Cross-Encoder 精排。响应必须完整覆盖所有候选且分数有效，否则整批降级到原始 RRF，不会被错误解释为 `no_match`。引用以 `chunk_id` 为证据边界，最终只展示回答实际使用的法条片段。

Ollama 和 TEI 进程输出分别追加到 `data/logs/ollama.log` 与 `data/logs/reranker.log`。Ollama 模型需要预先下载；TEI 首次启动会自动下载 BGE 模型：

```bash
ollama pull qwen3-embedding:0.6b
```

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

简历导向的 RAG 压力测试使用独立、明确有偏的挑战集，不能替代通用回归集：

```bash
# 从 law.json 固定抽样；保留旧候选并扩充至600条（共18个批次）
python scripts/create_resume_challenge_datasets.py prepare \
  --dense-size 300 --rerank-candidate-size 600 --seed 42
# 显式消耗 Codex 额度，以 xhigh 生成问题；支持断点续跑
python scripts/create_resume_challenge_datasets.py generate-codex
# 校验泄漏、重复和真实 Gold ID，冻结300条 Dense 与600条精排候选
python scripts/create_resume_challenge_datasets.py build
# 可单独在真实 Hybrid Top 12 上冻结满足“Gold + 至少两个预声明干扰项”的200条精排集
python scripts/create_resume_challenge_datasets.py qualify-reranker
# 只查看六组 RAG 实验、6条 Agent 冒烟、资源上限和准备状态；不创建报告或调用外部服务
python scripts/run_resume_rag_challenge_eval.py --plan-only
# 一键执行模块回归、数据校验、缺失时自动资格冻结、三组消融与 Agent 冒烟
python scripts/run_resume_rag_challenge_eval.py
```

Dense 挑战集固定300条语义改写、生活化案情、后果描述和噪声查询；Reranker 从600条候选中按
“Gold 进入未精排 Hybrid Top12 且至少两个预声明干扰项同时命中”选择前200条正式排序题。
原300条候选是扩容后的不可变前缀，只新增6个 Codex 批次。生成器看不到任何
Baseline/Candidate 结果；资格检查每25条原子保存 checkpoint，并输出完整诊断。报告增加 Hit@1、Top3、Gold 平均
排名、分类/难度分组、最多3条定性改善案例、硬门禁和基于实测数字生成的简历表述。最终200条
精排集缺失时，正式入口只在600条候选已冻结后使用未启用 BGE 的 Hybrid 进行资格冻结；可用
`--no-auto-qualify` 禁止。默认还会运行6类 Fixture Agent 冒烟，可用 `--skip-agent-smoke`
关闭；冒烟会少量调用 DeepSeek，但不会上传 LangSmith或运行 Judge。所有样本标记
`human_verified=false`，简历必须称为“Codex 生成、真实法规 ID 约束的源数据派生合成挑战集”。

默认 `learn` 使用固定输出讲解确定性指标，不访问 LangSmith、DeepSeek 或 Ollama；`smoke`
只运行 6 条分层样本且 `upload_results=false`。云端 Compare 默认仅上传 30 条根 Trace、调用
10 次 Judge，执行前必须先查看 `--plan-only`，再显式增加 `--confirm-upload`。所有评测受月度
Trace、Agent 模型和 Judge 的实际用量会写入本地账本，但评测不再设置月度硬上限；云端上传仍需显式确认，生产 Trace 仍保留独立保护。完整概念、指标公式和失败定位方法见
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

咨询默认使用持久化 AgentRun：创建任务后，前端通过带 sequence 的 SSE 订阅事件；切换用户、临时断网或刷新页面不会取消服务端 Graph，重新打开会话可恢复进度。LangGraph 使用独立 SQLite Checkpoint 数据库保存 super-step，业务消息和记忆仍由主 SQLite 管理。停止生成必须调用 cancel API，而不是仅关闭浏览器连接。

## 运行时与开发 Skill

运行时 Skill 位于 `skills/runtime/*/SKILL.md`，可通过以下环境变量整体关闭或调整加载策略：

```dotenv
AGENT_SKILLS_ENABLED=true
AGENT_SKILL_ROOT=./skills/runtime
AGENT_MAX_ACTIVE_SKILLS=2
AGENT_SKILL_STRICT_VALIDATION=true
```

Skill 是领域工作流与结构化输出约束，不是 MCP Tool。只有 Legal Research 能通过现有 MCP 调用 `search_laws/get_law_article`，Skill 不能绕过该协议边界。前端只接收安全的 `skill_status`，LangSmith 只记录 Skill ID 和版本，不上传完整 Skill Prompt。

仓库开发 Skill 位于 `.agents/skills/`：`lawstation-spec-change` 固化 SDD 与文档同步闭环，`lawstation-eval-review` 固化公平消融、报告归档和简历数字真实性规则。开发 Skill 不进入线上 Agent Prompt。

检索工具返回 `law-search-v2` Envelope，并由确定性置信度门控区分候选命中与正常 no-match。仓库内默认 gate 参数是 provisional；需要在 Ollama/TEI 和正式索引就绪后执行：

```bash
python scripts/calibrate_retrieval_gate.py
python scripts/calibrate_retrieval_gate.py --write  # 仅冻结验证集通过门禁时更新配置
```

## 测试

```bash
pytest -v
cd frontend && npm test && npm run build
```

演示用户只提供逻辑隔离，不是正式身份认证。`tools/` 可在明确业务需求下修改或复用，但所有改动必须遵守最小变更原则并通过对应回归测试。
