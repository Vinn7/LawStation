# routes.py 拆分设计（子项目 3/4）

## 背景

`backend/app/api/routes.py`（756 行）是 LawStation 唯一的 API 路由模块：一个
`APIRouter(prefix="/api")` 实例挂载了从场景观察模式管理、会话/消息 CRUD、记忆
管理、AgentRun 生命周期、消息反馈到 SSE 流式对话的全部端点，外加若干私有
helper 函数。这是本次多文件分解计划（`docs/superpowers/plans/` 下已完成的
`graph.py`、`memory_tasks.py` 之后）的第三个子项目。

本次拆分延续前两次已验证的模式：保持外部接口（导入路径、路由前缀、行为）不
变，仅重组内部文件结构，提升可读性和内聚性。

## 现状结构

`routes.py` 内容按声明顺序：

1. **导入与模块级常量**（1-49 行）：`router = APIRouter(prefix="/api")`，
   `SCENARIO_CONVERSATION_PREFIX = "[场景] "`。
2. **`index_status`**（52-54 行）：`GET /api/index/status`，调用
   `mcp_servers.law_rag.server.get_index_status`。
3. **场景观察模式**（57-158 行）：`_scenario_catalog`（helper）、
   `scenario_datasets`、`scenario_summaries`、`scenario_detail`、
   `scenario_run_outcome`、`_delete_scenario_conversation`（helper，直接用
   `SessionLocal` 开会话）、`delete_scenario_conversation`。
4. **`obj`**（160-161 行）：通用 ORM 行转 dict 的 helper，被多个后续分组使用。
5. **用户/会话/消息基础 CRUD**（164-196 行）：`users`、`conversations`、
   `create_conversation`、`messages`。
6. **记忆管理**（199-283 行）：`memories`、`update_memory`、
   `_set_memory_status`（helper，用到 `audit`）、`confirm_memory`、
   `reject_memory`、`delete_memory`、`clear_memories`、`memory_job`。
7. **SSE 工具函数**（286-294 行）：`sse`、`persisted_sse`——纯函数，被
   agent-run 分组和聊天流分组共用。
8. **AgentRun 生命周期**（297-406 行）：`create_agent_run`、`get_agent_run`、
   `active_agent_run`、`cancel_agent_run`、`agent_run_events`。
9. **`with_sse_heartbeat`**（409-435 行）：通用 SSE 心跳包装器，仅被
   `stream_message` 使用，但被 `tests/test_concurrency.py` 直接单元测试。
10. **消息反馈**（438-558 行，注意行号与第8组交叉环绕）：
    `_validate_conversation`、`_prepare_chat`、`_save_assistant`（这三个属于
    聊天流，见下）、`_sync_message_feedback`、`message_feedback`。
11. **SSE 流式对话**（561-756 行）：`stream_message`——全仓最复杂的单个端点，
    协调并发预留、LangSmith 追踪、记忆快照加载、Agent 调用、SSE 事件转发、
    回答持久化、记忆任务入队、多种异常路径下的审计日志。

## 外部消费者

```
$ grep -rln "from backend.app.api.routes\|api\.routes\|api import routes"
tests/test_concurrency.py   — from backend.app.api.routes import with_sse_heartbeat
tests/test_feedback.py      — from backend.app.api.routes import message_feedback
tests/test_scenario_cleanup.py
                             — from backend.app.api import routes
                               routes._delete_scenario_conversation(...)
                               routes._scenario_catalog(...)
                               monkeypatch.setattr(routes, "SessionLocal", sessions)
backend/app/main.py          — from backend.app.api.routes import router
```

`with_sse_heartbeat` 和 `message_feedback` 的测试直接调用函数、不依赖任何
模块级 monkeypatch，因此只要拆分后仍可从 `backend.app.api.routes` 顶层导入
这两个符号，测试文件本身无需改动。

`test_scenario_cleanup.py` 是本次拆分唯一的复杂点：它对 `routes` 模块对象
执行 `monkeypatch.setattr(routes, "SessionLocal", sessions)`，再调用
`routes._delete_scenario_conversation(...)`。这与 `memory_tasks.py` 拆分时
遇到的问题完全同构——见下节。

## 关键设计决策：SessionLocal monkeypatch

**问题**：拆分后，`_delete_scenario_conversation` 的实现会搬到新的
`scenarios.py` 子模块里，该子模块会有自己的
`from backend.app.db.session import SessionLocal` 绑定。测试如果继续
`monkeypatch.setattr(routes, "SessionLocal", sessions)`（patch 顶层包对象
上重新导出的名字），并不会影响 `scenarios.py` 内部实际查找 `SessionLocal`
时使用的那个独立绑定——monkeypatch 会静默失效，测试会打到真实数据库而不
是假数据库。

**决策（与 memory_tasks.py 拆分一致）**：修改 `tests/test_scenario_cleanup.py`，
改为 `from backend.app.api.routes import scenarios`，并将所有
`routes._delete_scenario_conversation(...)` / `routes._scenario_catalog(...)`
/ `monkeypatch.setattr(routes, "SessionLocal", sessions)` 改为对
`scenarios` 子模块的引用。不在生产代码里引入任何间接层（例如把
`SessionLocal` 包一层可注入的工厂）——生产代码保持直接
`from backend.app.db.session import SessionLocal` 的简单写法，测试补丁改
指向真正执行的位置（"patch where it's used, not where it's defined"）。

`with_sse_heartbeat`、`message_feedback` 没有这个问题，因为它们的测试不
monkeypatch 任何模块级名字，直接从包的 `__init__.py` 重新导出即可正常工作。

## 目标目录结构

```
backend/app/api/routes/
├── __init__.py       # 门面：组合子路由 + 重新导出测试依赖的符号
├── helpers.py         # obj, sse, persisted_sse, with_sse_heartbeat
├── index.py            # index_status
├── scenarios.py        # 场景观察模式：_scenario_catalog, scenario_datasets,
│                         scenario_summaries, scenario_detail,
│                         scenario_run_outcome, _delete_scenario_conversation,
│                         delete_scenario_conversation, SCENARIO_CONVERSATION_PREFIX
├── conversations.py    # users, conversations, create_conversation, messages
├── memories.py         # memories, update_memory, _set_memory_status,
│                         confirm_memory, reject_memory, delete_memory,
│                         clear_memories, memory_job
├── agent_runs.py       # create_agent_run, get_agent_run, active_agent_run,
│                         cancel_agent_run, agent_run_events
├── feedback.py         # _sync_message_feedback, message_feedback
└── chat.py              # _validate_conversation, _prepare_chat, _save_assistant,
                          stream_message
```

原 `backend/app/api/routes.py` 被删除，替换为同名包（与 `memory_tasks.py`
→ `memory_tasks/` 的先例完全一致）。

### 各文件精确内容

**`helpers.py`**（源 160-161, 286-294, 409-435 行）：
- `obj(row)` — 源 160-161 行
- `sse(event, data)` — 源 286-287 行
- `persisted_sse(sequence, event, data)` — 源 290-294 行
- `with_sse_heartbeat(source, interval_seconds)` — 源 409-435 行
- 依赖：`asyncio`, `json`

**`index.py`**（源 46, 52-54 行）：
- `router = APIRouter(prefix="/api")`
- `index_status()` — 源 52-54 行
- 依赖：`mcp_servers.law_rag.server.get_index_status`

**`scenarios.py`**（源 21-30, 31 部分, 49, 57-158 行）：
- `router = APIRouter(prefix="/api")`
- `SCENARIO_CONVERSATION_PREFIX = "[场景] "` — 源 49 行
- `_scenario_catalog(request)` — 源 57-61 行
- `scenario_datasets(...)` — 源 64-66 行
- `scenario_summaries(...)` — 源 69-74 行
- `scenario_detail(...)` — 源 77-84 行
- `scenario_run_outcome(...)` — 源 87-93 行
- `_delete_scenario_conversation(ctx, conversation_id)` — 源 96-138 行
- `delete_scenario_conversation(...)` — 源 141-158 行
- 依赖：`asyncio`, `HTTPException`, `Request`, `Depends`, `sqlalchemy.delete/select`,
  `backend.app.core.context.get_user_context`,
  `backend.app.db.models.{AgentRun, Conversation, MemoryJob, RetrievalTrace, ToolCallRecord}`,
  `backend.app.db.session.SessionLocal`

**`conversations.py`**（源 164-196 行）：
- `router = APIRouter(prefix="/api")`
- `users(...)` — 源 164-166 行
- `conversations(...)` — 源 169-171 行
- `create_conversation(...)` — 源 174-179 行
- `messages(...)` — 源 182-196 行
- 依赖：`Depends`, `sqlalchemy.select`, `Session`,
  `backend.app.core.context.{RequestUserContext, get_user_context}`,
  `backend.app.db.models.{User, Conversation, MessageFeedback}`,
  `backend.app.db.session.get_db`, `backend.app.schemas.ConversationCreate`,
  `backend.app.services.repositories.OwnedRepository`,
  `.helpers.obj`

**`memories.py`**（源 199-283 行）：
- `router = APIRouter(prefix="/api")`
- `memories(...)` — 源 199-213 行
- `update_memory(...)` — 源 216-224 行
- `_set_memory_status(...)` — 源 227-245 行
- `confirm_memory(...)` — 源 248-250 行
- `reject_memory(...)` — 源 253-255 行
- `delete_memory(...)` — 源 258-262 行
- `clear_memories(...)` — 源 265-271 行
- `memory_job(...)` — 源 274-283 行
- 依赖：`Query`, `Depends`, `HTTPException`, `sqlalchemy.select`,
  `backend.app.core.context.get_user_context`,
  `backend.app.core.logging.audit`, `backend.app.db.models.MemoryJob`,
  `backend.app.schemas.{MemoryUpdate, MemoryVersionRequest}`,
  `backend.app.services.repositories.OwnedRepository`, `.helpers.obj`

**`agent_runs.py`**（源 297-406 行）：
- `router = APIRouter(prefix="/api")`
- `create_agent_run(...)` — 源 297-328 行
- `get_agent_run(...)` — 源 331-336 行
- `active_agent_run(...)` — 源 339-346 行
- `cancel_agent_run(...)` — 源 349-354 行
- `agent_run_events(...)` — 源 357-406 行
- 依赖：`asyncio`, `Header`, `Query`, `Request`, `HTTPException`, `Depends`,
  `backend.app.agent.concurrency.{ConcurrencyIdentity, ConversationBusyError}`,
  `backend.app.core.config.get_settings`, `backend.app.core.context.get_user_context`,
  `backend.app.services.agent_runs.{TERMINAL_STATUSES, AgentRunConflict, run_payload}`,
  `.helpers.{sse, persisted_sse}`

**`feedback.py`**（源 492-558 行）：
- `router = APIRouter(prefix="/api")`
- `_sync_message_feedback(app, feedback_id)` — 源 492-508 行
- `message_feedback(...)` — 源 511-558 行
- 依赖：`BackgroundTasks`, `Depends`, `sqlalchemy.select`,
  `backend.app.core.context.get_user_context`, `backend.app.core.logging.audit`,
  `backend.app.db.models.{Message, MessageFeedback}`,
  `backend.app.schemas.MessageFeedbackRequest`

**`chat.py`**（源 438-490, 561-756 行）：
- `router = APIRouter(prefix="/api")`
- `_validate_conversation(...)` — 源 438-441 行
- `_prepare_chat(...)` — 源 444-467 行
- `_save_assistant(...)` — 源 470-489 行
- `stream_message(...)` — 源 561-756 行
- 依赖：`asyncio`, `logging`, `time`, `Request`, `HTTPException`, `Depends`,
  `StreamingResponse`,
  `backend.app.agent.concurrency.{AgentConcurrencyManager, AgentQueueTimeoutError, ConcurrencyIdentity, ConversationBusyError}`,
  `backend.app.agent.service.AgentService`, `backend.app.core.config.get_settings`,
  `backend.app.core.context.{RequestUserContext, get_user_context}`,
  `backend.app.core.logging.{audit, summary}`, `backend.app.db.models.Message`,
  `backend.app.db.session.SessionLocal`, `backend.app.schemas.ChatRequest`,
  `backend.app.services.memory.MemoryService`,
  `backend.app.services.repositories.OwnedRepository`,
  `.helpers.{sse, with_sse_heartbeat}`

**`__init__.py`**（门面）：
```python
from fastapi import APIRouter

from . import agent_runs, chat, conversations, feedback, index, memories, scenarios
from .chat import with_sse_heartbeat
from .feedback import message_feedback
from .scenarios import _delete_scenario_conversation, _scenario_catalog

router = APIRouter()
for module in (index, scenarios, conversations, memories, agent_runs, feedback, chat):
    router.include_router(module.router)

__all__ = ["router"]
```

（`with_sse_heartbeat`、`message_feedback`、`_delete_scenario_conversation`、
`_scenario_catalog` 的重新导出仅服务于现有测试的导入路径，不是公开 API 的
一部分。已核实 `obj`/`persisted_sse`/`sse` 在 `routes.py` 之外没有任何消费
者（仅被同包内其它子模块使用），因此包门面不重新导出它们，避免维护无人使
用的兼容接口。）

## 需要同步修改的测试文件

`tests/test_scenario_cleanup.py`：已核实该文件里 `routes` 这个名字只用于
第 21/50/52/54/77/90 行这 6 处对 `_delete_scenario_conversation` /
`_scenario_catalog` / `SessionLocal` 的引用，没有其它用途，因此直接整体替换
导入，不并存两个名字：
- 第 8 行 `from backend.app.api import routes` →
  `from backend.app.api.routes import scenarios`
- 第 21 行 `monkeypatch.setattr(routes, "SessionLocal", sessions)` →
  `monkeypatch.setattr(scenarios, "SessionLocal", sessions)`
- 第 50/52/54/77 行 `routes._delete_scenario_conversation(...)` →
  `scenarios._delete_scenario_conversation(...)`
- 第 90 行 `routes._scenario_catalog(...)` → `scenarios._scenario_catalog(...)`

`tests/test_concurrency.py`、`tests/test_feedback.py`：无需改动（从包顶层
导入即可，两者都不 monkeypatch 模块级状态）。

## 不做的事

- 不改变任何端点的 URL、HTTP 方法、请求/响应模型、状态码、审计日志字段。
- 不改变 `stream_message` 的 SSE 事件顺序或异常处理路径。
- 不引入 `SessionLocal`/`audit` 等的依赖注入或工厂封装——保持 `memory_tasks.py`
  拆分时确认的"直接 import，测试改 patch 路径"原则。
- 不合并到 FastAPI 的 `APIRouter(prefix=...)` 之外的其它路由聚合机制（如
  `include_router` 的 `tags=`）；各子路由沿用原来隐式的默认 tag 行为。

## 验证方式

- 全量测试套件（当前基线：176 passed，含 graph.py 和 memory_tasks.py 拆分后
  的状态）在拆分后必须保持同样通过数。
- `tests/test_scenario_cleanup.py` 的哨兵验证：与 memory_tasks.py 拆分一致，
  最终 review 阶段用同样的 sentinel-fixture 技巧确认 `scenarios.SessionLocal`
  的 monkeypatch 确实生效、不会静默失败。
- `curl :8000/api/index/status` 等手工冒烟测试非必需（此次为纯结构重组，
  测试套件已覆盖行为）。
