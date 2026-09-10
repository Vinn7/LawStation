# LawStation Agent 持久化机制代码导读

> 本文按真实调用顺序阅读代码，重点解释业务任务、LangGraph Checkpoint、任务租约、幂等写入和 SSE 事件重放如何协作。当前实现是单机 SQLite 持久化方案，不应描述为分布式任务队列。

## 1. 先建立整体认识

一次 Agent 回答横跨 HTTP 请求、后台 Worker、LangGraph、模型与 MCP 调用，执行时间可能达到几十秒。如果把执行协程直接绑定在浏览器 SSE 连接上，页面刷新或网络中断就可能取消任务；如果仅在内存中保存进度，服务重启后也无法恢复。

LawStation 因此采用双层持久化：

```text
业务任务层：AgentRun + AgentRunEvent
├── 任务所有权、排队、状态和取消
├── Worker 领取与租约续期
├── 正式消息幂等写入
└── SSE 事件游标与断线重放

Graph 执行层：LangGraph AsyncSqliteSaver
├── 在 super-step 边界保存 Graph State
├── 记录最后完成的节点及下一节点
├── 恢复案情、证据、草稿和复核状态
└── 恢复模型/工具调用计数
```

两层分别回答不同问题：

| 问题 | 负责组件 |
|---|---|
| 任务属于谁、处于什么状态 | `AgentRun` |
| 哪个 Worker 正在执行 | `lease_owner/lease_expires_at` |
| Graph 已运行到哪个节点 | LangGraph Checkpoint |
| 客户端错过了哪些事件 | `AgentRunEvent.sequence` |
| 最终消息是否已经保存 | `user_message_id/assistant_message_id` |

## 2. 建议的源码阅读顺序

| 顺序 | 文件与关键 symbol | 阅读目标 |
|---:|---|---|
| 1 | `backend/app/main.py:70-107` `lifespan()` | 共享 Saver、Runtime 和 Worker 如何创建、关闭 |
| 2 | `backend/app/db/models.py:209-283` `AgentRun/AgentRunEvent` | 任务、租约、幂等和事件字段 |
| 3 | `backend/app/api/routes/agent_runs.py` | 创建任务、查询、取消和事件订阅 API |
| 4 | `backend/app/services/agent_runs.py:89-174` | Worker 启动及 queued Run 创建 |
| 5 | `backend/app/services/agent_runs.py:331-531` | 调度、领取、租约和完整执行 |
| 6 | `backend/app/agent/checkpoint.py:16-43` | `AsyncSqliteSaver` 生命周期 |
| 7 | `backend/app/agent/runtime.py:70-135` | 新 Graph 与恢复 Graph 的输入差异 |
| 8 | `backend/app/services/agent_runs.py:533-634` | 用户消息、助手消息和终态幂等 |
| 9 | `backend/app/services/agent_runs.py:678-706` | 事件序号如何原子追加 |
| 10 | `backend/app/services/agent_runs.py:290-329` | 服务重启和过期数据清理 |

## 3. 应用启动：持久化组件先于 Worker 创建

入口位于 `backend/app/main.py:70-107`。

```python
async with checkpoint_saver(get_settings()) as checkpointer:
    app.state.agent_runtime = AgentRuntime(..., checkpointer=checkpointer)
    app.state.agent_runs = AgentRunManager(app.state.agent_runtime, ...)
    await app.state.agent_runs.start()
    yield
```

这段代码的关键点是：

1. `checkpoint_saver()` 的生命周期覆盖全部 Graph 执行，避免任务运行时 SQLite 连接提前关闭；
2. 同一个 Saver 注入应用级 `AgentRuntime`，但不同 Run 使用不同 `thread_id`，不会共享案件 State；
3. `AgentRunManager.start()` 在接受请求前恢复遗留任务并启动后台 Worker；
4. 应用关闭时先停任务 Worker，再关闭 Runtime 和 Saver。

Checkpoint 独立保存在 `data/runtime/langgraph-checkpoints.db`，业务任务和消息保存在业务 SQLite。创建逻辑位于 `backend/app/agent/checkpoint.py:16-43`：

```python
connection = await aiosqlite.connect(path)
saver = AsyncSqliteSaver(
    connection,
    serde=JsonPlusSerializer(pickle_fallback=False),
)
await saver.setup()
```

`pickle_fallback=False` 避免反序列化任意 Pickle 对象。Checkpoint 数据库与业务数据库分离，也减少 Graph 高频状态写入与消息事务争用同一个 SQLite 文件。

## 4. 创建任务：HTTP 只负责入队

前端调用 `POST /api/conversations/{conversation_id}/runs`。入口位于 `backend/app/api/routes/agent_runs.py:18-49`（`create_agent_run`）。

API 的工作只有：

1. 生成请求级并发身份；
2. 预占当前会话，避免同一会话重复提交；
3. 调用 `AgentRunManager.create()` 创建 queued Run；
4. 返回 HTTP `202`。

`202` 表示“已接受并排队”，不表示回答已经完成。真正的模型调用由后台 Worker 执行，因此创建请求结束不会终止 Agent。

`AgentRunManager.create()` 位于 `backend/app/services/agent_runs.py:125-174`。它先用同一条 SQL 校验会话所有权，然后创建：

```python
run = AgentRun(
    id=run_id,
    status="queued",
    current_stage="queued",
    langgraph_thread_id=f"agent-run:{run_id}",
)
```

每个 Run 使用独立的 LangGraph `thread_id`。这里不能使用 `conversation_id`，否则同一会话的下一轮咨询可能继承上一轮的 `EvidencePacket`、草稿和中间状态，形成第二套跨轮记忆。

同一事务还写入 `message_start` 和 queued `agent_status`。如果两个请求并发穿过进程内检查，数据库的部分唯一索引仍保证同一用户、同一会话只能存在一个 `queued/running` Run。模型定义位于 `backend/app/db/models.py:219-226`。

## 5. AgentRun：业务任务的事实源

`AgentRun` 定义在 `backend/app/db/models.py:209-258`，字段可以分成四组。

### 5.1 所有权

```text
tenant_id / user_id / conversation_id / request_id
```

创建、读取、取消和事件查询都必须带上服务端解析出的所有权条件。客户端只能提交 Run ID，不能自行指定 LangGraph thread ID。

### 5.2 状态机

```text
queued → running → completed
                 ├→ interrupted
                 └→ failed
```

`current_stage` 进一步表示 analyzing、researching、drafting、reviewing 等阶段。终态集合定义于 `backend/app/services/agent_runs.py:37-38`。

### 5.3 执行与恢复

```text
attempt
cancel_requested
lease_owner
lease_expires_at
langgraph_thread_id
latest_checkpoint_id
```

`latest_checkpoint_id` 是业务表中的诊断关联字段；真正的节点恢复状态仍以 Checkpoint 数据库为准。

### 5.4 幂等和事件

```text
user_message_id
assistant_message_id
last_event_seq
model_call_count
tool_call_count
```

消息 ID 防止恢复后重复写消息，事件序号用于断线重放，调用计数用于恢复后继续遵守上限。

## 6. Worker 调度：从 queued 到 running

`AgentRunManager.start()` 位于 `backend/app/services/agent_runs.py:89-111`。它先执行恢复和清理，再用 `asyncio.create_task()` 启动 `_loop()`。

`_loop()` 位于 `backend/app/services/agent_runs.py:331-342`：

```python
run_ids = await asyncio.to_thread(self._queued_ids)
for run_id in run_ids:
    if run_id not in self._active:
        task = asyncio.create_task(self._execute(run_id))
```

这里的 `_active` 只防止当前进程为同一个 Run 重复创建协程。真正执行前，`_execute()` 还要：

```text
获得全局/用户并发配额
→ claim 数据库任务
→ 启动租约心跳
→ 准备消息和 MemorySnapshot
→ 执行 LangGraph
```

对应代码为 `backend/app/services/agent_runs.py:361-405`。排队期间不调用模型或 MCP，也不会提前读取可能过时的记忆快照。

## 7. 任务租约：标识当前执行者

`_claim()` 位于 `backend/app/services/agent_runs.py:515-531`。成功领取时写入：

```python
run.status = "running"
run.attempt += 1
run.lease_owner = self.worker_id
run.lease_expires_at = now + lease_seconds
```

随后 `_execute()` 创建 `_lease_heartbeat()`。续租逻辑位于 `backend/app/services/agent_runs.py:487-509`，默认每 `lease_seconds / 3` 更新一次到期时间，并同时检查：

```text
run_id 匹配
status = running
lease_owner = 当前 Worker
```

任务正常完成、失败或取消时都会清空租约。`finally` 还会取消心跳协程并释放并发配额，代码位于 `backend/app/services/agent_runs.py:478-485`。

需要准确理解当前边界：租约目前主要是单实例 Worker 的存活和归属记录。启动恢复会处理全部遗留 `running` Run，没有依据 `lease_expires_at` 实现多实例间的数据库原子抢占。因此简历可以写“任务租约与恢复”，不能写“分布式租约队列”。

## 8. 消息准备：先保证用户输入只保存一次

`_prepare()` 位于 `backend/app/services/agent_runs.py:533-557`：

```python
if current.user_message_id:
    user_message = db.get(Message, current.user_message_id)
else:
    user_message = Message(role="user", content=run.input_text)
    current.user_message_id = user_message.id
```

如果服务在用户消息保存后崩溃，恢复任务会复用 `user_message_id`，不会把同一个问题再次写入会话。

用户消息确定后，代码在同一个短事务中读取本轮 `MemorySnapshot`。数据库 Session 随 `_prepare()` 返回而关闭，不跨越后面的模型和 MCP 调用。

## 9. LangGraph Checkpoint：保存节点级执行进度

`AgentRuntime.stream()` 位于 `backend/app/agent/runtime.py:70-231`。新任务先构造完整初始 State，包括：

```text
messages / memory_context
case_analysis / evidence_packet / counsel_draft / review_result
retry_count / revision_count
current_fact_overrides
model_call_count / tool_call_count / tool_trajectory
final_answer / citations / errors
```

运行配置只设置：

```python
configurable["thread_id"] = thread_id
configurable.pop("checkpoint_ns", None)
```

根 Graph 不自定义 `checkpoint_ns`。该字段是 LangGraph 用于嵌套子图的内部路径，不是业务版本号。

Graph 通过 `compiled.astream()` 执行。启用 Checkpointer 后，LangGraph 在 super-step 边界保存 State。它不是每个 Token 保存一次，也不能保证当前节点内部的外部操作恰好执行一次。

### 9.1 新任务

```python
graph_input = initial_state
await graph.compiled.astream(graph_input, ...)
```

Graph 从 START 开始执行。

### 9.2 恢复任务

当 `attempt > 1` 时，`AgentService` 向 Runtime 传入 `resume=True`。Runtime 使用相同 thread ID 调用 `aget_state()`，代码位于 `backend/app/agent/runtime.py:113-126`：

```python
snapshot = await graph.compiled.aget_state(config)
context.metrics.model_call_count = snapshot.values["model_call_count"]
context.metrics.tool_call_count = snapshot.values["tool_call_count"]
graph_input = None
```

向已有 thread 传 `None` 表示从最新 Checkpoint 的下一节点继续，而不是重新提交初始 State。调用计数也从 Checkpoint 恢复，防止重启绕过模型、工具和循环上限。

## 10. 运行事件：只持久化安全状态

Graph 执行过程中，`AgentService` 产生 Agent 和工具状态。`_execute()` 位于 `backend/app/services/agent_runs.py:406-420`，处理规则是：

- `token` 先缓存在内存；
- `citations` 保存为最终引用候选；
- 其他安全事件立即写入 `AgentRunEvent`；
- 未复核草稿、Prompt、reasoning 和完整工具正文不写事件表。

每个事件通过 `_append_event_in_session()` 写入，位置为 `backend/app/services/agent_runs.py:688-702`：

```python
run.last_event_seq += 1
event = AgentRunEvent(
    run_id=run.id,
    sequence=run.last_event_seq,
    event_type=event_type,
    payload_json=json.dumps(payload),
)
db.add(event)
```

序号递增和 Event 插入发生在调用方的同一个事务中；数据库还用 `(run_id, sequence)` 唯一约束防止同一 Run 出现重复序号。

## 11. 最终回答：幂等写入而不是“恰好执行一次”

Graph 完成后，`_execute()` 取得最终回答、引用和 Checkpoint ID，然后调用 `_complete()`。入口位于 `backend/app/services/agent_runs.py:421-449`。

`_complete()` 位于 `backend/app/services/agent_runs.py:559-609`，包含三层幂等检查。

### 11.1 已完成任务直接复用结果

```python
if run.status == "completed" and run.assistant_message_id:
    return run.assistant_message_id
```

### 11.2 已有助手消息则不再创建

```python
assistant = db.get(Message, run.assistant_message_id)
if assistant is None:
    assistant = Message(role="assistant", content=answer)
```

### 11.3 已有正文事件则不再重复写

```python
has_tokens = select(AgentRunEvent.id).where(event_type == "token")
if not has_tokens:
    append token events
    append citations
```

因此即使服务在“助手消息和 Token 已提交、Run 还没改成 completed”的窗口崩溃，恢复后也会复用已有消息和事件。

最后 `_finish_completed()` 在 `backend/app/services/agent_runs.py:611-634` 中写入 `memory_status`、更新终态并追加唯一的 `message_end`。如果 Run 已经 completed，函数直接返回。

## 12. SSE 事件重放：恢复客户端视图

事件订阅入口位于 `backend/app/api/routes/agent_runs.py:78-127`（`agent_run_events`）：

```http
GET /api/agent-runs/{run_id}/events?after_sequence=N
Last-Event-ID: N
```

服务端取 Query 游标和 Header 的较大值，然后调用 `AgentRunManager.events()`。查询位于 `backend/app/services/agent_runs.py:195-213`，同时限定：

```text
run_id
tenant_id
user_id
sequence > cursor
ORDER BY sequence
```

返回格式为：

```text
id: 15
event: agent_status
data: {...}
```

浏览器收到事件后保存 sequence。重连时从最后确认位置继续读取，所以不会重新执行 Agent，也不会重复接收已经确认的事件。

如果任务仍在运行且暂时没有新事件，服务端通过 `asyncio.Condition` 等待 Worker 通知；超时后发送 `: heartbeat`。Heartbeat 不写数据库、不占 sequence，也不参与重放。任务进入终态且游标已经追上 `last_event_seq` 后，SSE 才结束。

浏览器断开只终止订阅生成器，不会取消后台 `_execute()`。这是“页面刷新后任务继续”的关键。

## 13. 显式取消：与浏览器断线严格区分

取消入口为 `POST /api/agent-runs/{run_id}/cancel`，路由位于 `backend/app/api/routes/agent_runs.py:70-75`（`cancel_agent_run`）。

`AgentRunManager.cancel()` 和 `_request_cancel()` 位于 `backend/app/services/agent_runs.py:253-288`：

- queued：直接转为 interrupted，追加终态事件，不调用模型；
- running：先持久化 `cancel_requested=true`，再取消当前进程中的 asyncio Task；
- `_execute()` 捕获 `CancelledError` 后写 interrupted 和 `message_end`；
- 应用正常关闭造成的 Task 取消不会伪装成用户取消，而是保留 running 供重启恢复。

由于最终正文只在 Graph 完成后写入，用户取消不会把 Counsel 未复核草稿保存成正式助手消息。

## 14. 服务重启：业务任务与 Graph 状态协同恢复

启动时 `AgentRunManager.start()` 先执行 `_recover_stale()`，位置为 `backend/app/services/agent_runs.py:290-309`：

```text
遗留 running Run
├── attempt < 最大恢复次数 → 清租约并重新 queued
└── attempt >= 最大恢复次数 → failed / RecoveryLimitExceeded
```

默认最大次数为 2，配置位于 `backend/app/core/config.py:51-54`。其含义是最多允许两次 claim：首次运行崩溃后可恢复一次，第二次仍失败则下一次启动终止任务。

重新 queued 的任务被 Worker 再次 claim，`attempt` 增加，随后通过相同 `langgraph_thread_id` 恢复 Checkpoint。业务任务层决定“是否允许重试”，LangGraph 决定“从哪个节点继续”。

## 15. 典型崩溃窗口

| 崩溃位置 | 恢复行为 | 是否可能重复 |
|---|---|---|
| Run 创建后、Worker 领取前 | queued Run 继续被轮询 | 不重复消息 |
| 用户消息保存后、Graph 启动前 | 复用 `user_message_id` | Graph 从初始节点开始 |
| Analyst Checkpoint 后、Research 前 | 从 Research 继续 | Analyst 不重跑 |
| MCP 已调用但 Research 尚未 Checkpoint | Research 节点可能重跑 | 只读 MCP 可能重复 |
| 助手消息已保存、终态未写 | 复用 `assistant_message_id` 和 Token | 不重复正式消息 |
| `message_end` 已写后 | completed Run 直接返回 | 不重复终态事件 |

所以当前一致性语义是：

```text
Graph 节点：至少一次
节点之间：Checkpoint 恢复
正式消息：幂等
最终回答事件：幂等
公开过程事件：sequence 重放
```

这不是端到端 exactly-once。当前 MCP 法规工具是只读操作，因此节点内重试可以接受；如果未来加入发送邮件、提交表单等副作用工具，还必须增加独立业务幂等键和外部操作记录。

## 16. 数据保留与清理

`_prune_expired_events()` 位于 `backend/app/services/agent_runs.py:311-329`。超过保留期的终态 Run 会：

1. 删除对应 `AgentRunEvent`；
2. 清空 `latest_checkpoint_id`；
3. 删除对应 LangGraph thread。

Run 主记录和正式 Message 仍保留，所以历史回答可以继续查看，但过程事件不能继续重放，Graph 也不能再恢复。

当前实现使用 `AGENT_RUN_EVENT_RETENTION_DAYS` 同时触发 Event 和 Checkpoint 清理；`LANGGRAPH_CHECKPOINT_RETENTION_DAYS` 尚未形成独立清理策略，这是已知技术债。

## 17. 阅读代码时最容易混淆的概念

### 17.1 AgentRun 不等于 LangGraph State

AgentRun 是业务任务事实源；Graph State 是一次执行的节点数据。前者可以公开查询，后者不能通过 API 暴露。

### 17.2 Checkpoint 不等于长期记忆

Checkpoint 只服务单个 `agent-run:<run_id>`。跨轮上下文仍来自 messages、会话摘要和长期记忆。

### 17.3 Event replay 不等于重新生成

重放只读取已保存事件，不调用模型、不执行 Graph，也不重新计算回答。

### 17.4 Heartbeat 不等于租约心跳

- SSE heartbeat：保持浏览器 HTTP 连接；
- Lease heartbeat：更新数据库中的 Worker 任务租约。

两者名称相似，但作用对象完全不同。

### 17.5 LangSmith Trace 不等于持久化状态

LangSmith用于可观测和评测。即使 LangSmith 不可用，AgentRun、Checkpoint、正式消息和事件重放仍由本地数据库保证。

## 18. 当前能力与边界

已经实现：

- 页面刷新或 SSE 断开后任务继续；
- 使用游标补回遗漏事件；
- 服务重启后恢复未完成 Graph；
- 用户、助手消息和最终正文事件幂等写入；
- 恢复后模型、工具和回流计数不归零；
- 取消和普通断线具有不同语义。

尚未实现：

- 多实例 Worker 的原子租约抢占；
- Redis、Celery、Kafka 等分布式任务基础设施；
- 节点内部副作用工具的 exactly-once；
- 独立使用 Checkpoint retention 配置；
- DeepSeek 原始 Token 的实时持久化，当前正文在 Finalize 后分块写入。

## 19. 面试表达

可以用以下方式概括：

> 为解决多 Agent 调用时间长、浏览器断线和服务重启导致任务丢失的问题，我将执行链路拆为业务任务层与 LangGraph 执行层：AgentRun 管理排队、租约、取消和最终消息关联，AsyncSqliteSaver 在节点边界持久化 Graph State；正式消息与正文事件通过关联 ID 和存在性检查实现幂等写入，前端事件按 Run 内 sequence 持久化并通过 Last-Event-ID 重放。该方案支持刷新、重连和单机重启恢复，同时明确保持节点至少一次、业务结果幂等的语义。

面试官继续追问时，应主动说明：当前租约适用于单实例 SQLite，尚未实现多 Worker 原子抢占；未来扩展到分布式部署时，应引入带条件更新的数据库 claim 或专业任务队列，并为副作用工具增加幂等键。
