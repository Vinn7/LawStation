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

在 `.env` 填写 `DEEPSEEK_API_KEY`；启用 Dense 检索时再填写 `DASHSCOPE_API_KEY`。默认法规数据源是固定抽样的 `data/knowledge/law/law_sample.json`。启动时会检查索引指纹：有效索引直接复用，缺失或过期时后台构建，构建期间自动使用 BM25。

重新生成相同的 100 条样本：

```bash
python scripts/create_law_sample.py --size 100 --seed 42
```

恢复全量建库时，将 `.env` 的 `LAW_DATA_PATH` 改为 `./data/knowledge/law/law.json`。

## 统一启动

```bash
conda activate LawStation
python run.py
```

启动器会在前端缺失或过期时自动安装/构建前端，然后启动唯一的 Uvicorn 进程。访问：

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

Compose 只启动一个 `lawstation` 容器并暴露 8000 端口。

## 索引与审计日志

手工等待索引构建完成或强制重建：

```bash
python scripts/build_index.py
python scripts/build_index.py --force
```

控制台审计事件同时以 JSON Lines 追加到 `data/logs/lawstation.log`。默认单文件 20 MB、保留 10 个备份；日志只保存脱敏摘要和工具结果标识，不记录密钥或完整法条正文。

## 分层记忆

- 当前案件事实只在所属会话使用；用户偏好和稳定背景可以跨该用户的会话复用。
- 合法抽取的用户偏好和案件事实在后台整理完成后自动生效，无需再次确认。
- 页面左侧“管理我的记忆”可修正或删除已生效记忆；升级前已有的待确认记录仍可确认或拒绝。
- 回答完成后由 SQLite 持久后台任务整理记忆和增量摘要，不阻塞主回答；服务重启会恢复未完成任务。
- 记忆整理使用独立的非 Thinking JSON Output 调用，不绑定或调用 MCP 工具；简单问候等没有可沉淀内容的消息会正常完成且不创建记忆。

## 测试

```bash
pytest -v
cd frontend && npm test && npm run build
```

演示用户只提供逻辑隔离，不是正式身份认证。`tools/` 可在明确业务需求下修改或复用，但所有改动必须遵守最小变更原则并通过对应回归测试。
