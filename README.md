# LawStation

单进程、单端口的法律咨询 Agent：FastAPI 同时托管 React 页面、业务 API、SSE 对话与法律 RAG MCP Server。

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

## 测试

```bash
pytest -v
```

演示用户只提供逻辑隔离，不是正式身份认证。现有 `tools/` 仅作为参考，运行时不会导入或修改其中代码。
