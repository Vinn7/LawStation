# LangGraph Checkpoint 持久化机制

> 面向：知道"数据库事务"但没接触过"工作流引擎状态持久化"的后端工程师。
> 目标：理解 Checkpoint 到底在存什么、为什么不是"再建一张业务表"就能替代，以及本项目怎么在它之上又搭了一层业务持久化。

## 1. 要解决的问题

一次法律咨询回答，背后是 Case Analyst → Legal Research（内部还有若干轮工具调用）→ Legal Counsel → Reviewer 这样一条链路，实际耗时可能是几十秒。这期间如果发生：

- 服务进程被重启（部署、崩溃、手动重启）
- 用户刷新了页面 / 断网重连
- 前端主动关闭了 SSE 连接（但没有调用"取消"接口）

这个"跑到一半"的执行状态要怎么办？直接丢弃重跑，代价是重复消耗 LLM/工具调用配额，用户体验也差（进度条归零重来）。我们想要的是：**从上次执行到的地方继续，而不是从头开始**。

## 2. 行业内一般怎么做

这类"长时间运行的多步骤任务，要求可恢复"的问题，业界大致有三条路：

- **工作流引擎路线**（Temporal、Cadence、AWS Step Functions）：不保存"当前状态"这一个快照，而是保存**完整的事件历史**（每一步做了什么、返回了什么），恢复时通过**重放（replay）**这些历史事件重新算出当前应该处于什么状态。优点是可审计、理论上能回溯到任意历史时刻；代价是业务代码必须写成"确定性重放安全"的（比如不能在重放时重复真正调用一次外部 API），框架对代码风格有较强约束。
- **手工任务表路线**（很多自研系统 / Celery 之类的任务系统）：一张 `jobs` 表存 `status` 字段，业务代码自己在关键节点手动 `UPDATE progress = ...`。优点是简单直接、心智负担低；缺点是"进度"粒度完全靠开发者手动埋点，容易漏埋、容易和业务逻辑搅在一起，扩展到复杂分支流程时代码会变得难维护。
- **框架级状态快照路线**（LangGraph、以及一些 Actor 模型框架）：框架知道你的"工作流"是由哪些节点（node）组成的，每执行完一个节点就自动把当前完整状态序列化落盘，不需要业务代码手动埋点，代价是你要接受用它的 Graph/Node 抽象来组织业务逻辑，不能完全按自己的想法写控制流。

本项目走的是第三条路——这是理解本项目 Checkpoint 设计的前提：**它是 LangGraph 框架自动提供的能力，不是本项目自己发明的机制**。

## 3. 核心机制原理

Checkpoint 保存的是某个 `thread_id`（一次执行的唯一标识）在某个 **super-step**（一轮节点执行）完成后的**完整 State 快照**。恢复时不是"从头重新推理一遍"，而是"读出最近一次快照的 State，直接从后面还没跑的节点继续跑"。

关键在于：这个"自动存快照"的动作完全由框架在幕后完成，业务节点函数（比如 `case_analyst`、`legal_researcher`）只管接收 State、返回增量 dict，不需要自己调用任何"保存"函数。

## 4. 本项目具体实现（函数级）

### 4.1 Checkpoint 存储后端的搭建

[checkpoint.py:16-43](../../backend/app/agent/checkpoint.py#L16)：

```python
@asynccontextmanager
async def checkpoint_saver(settings: Settings):
    ...
    connection = await aiosqlite.connect(path)
    saver = AsyncSqliteSaver(connection, serde=JsonPlusSerializer(pickle_fallback=False))
    await saver.setup()
    try:
        yield saver
    finally:
        await connection.close()
```

逐个说明用到的框架内部对象：

- **`AsyncSqliteSaver`**：LangGraph 官方提供的 Checkpoint 存储后端实现之一（还有 Postgres 版等），负责"把 State 序列化写入/从 SQLite 读出"这件事的具体落地。它实现了 LangGraph 定义的 `BaseCheckpointSaver` 接口，Graph 编译时把它传进去，之后每个 super-step 自动调用它保存。
- 构造参数 `connection`：`aiosqlite`（异步 SQLite 客户端）的连接对象，**不是**本项目业务用的 SQLAlchemy 连接——这是一条独立的、专属 Checkpoint 的数据库连接，物理上和业务库（`data/runtime/lawstation.db`）是两个文件。
- `serde=JsonPlusSerializer(pickle_fallback=False)`：`serde`（serializer/deserializer）决定 State 里的 Python 对象怎么变成字节存进 SQLite。这里显式关闭 `pickle_fallback`——遇到序列化器不认识的类型会直接报错，而不是退化用 Python 的 `pickle` 兜底。这是有意的安全选择：`pickle` 反序列化任意数据是已知的安全风险（构造恶意 pickle 数据可以在反序列化时执行任意代码），Checkpoint 数据库理论上是持久化文件，不应该允许"读出来就能跑代码"这种攻击面。
- **`saver.setup()`**：只创建 Checkpoint 自己需要的表结构（`checkpoints`/`writes` 等），不会编译或运行任何 Graph 节点，也不碰业务表。

### 4.2 发起一次新的 / 续跑的 Agent 执行

[runtime.py:104-116](../../backend/app/agent/runtime.py#L104)：

```python
configurable["thread_id"] = thread_id or configurable.get("thread_id") or context.identity.request_id
configurable.pop("checkpoint_ns", None)
...
snapshot = await graph.compiled.aget_state(config)
```

- **`graph.compiled`**：`StateGraph.compile()` 的返回值，类型是 `CompiledStateGraph`——把节点/边的拓扑关系固化成一个可执行对象。编译这一步本身**不会**发起任何模型调用，纯粹是结构组装。
- **`aget_state(config)`**：`CompiledStateGraph` 提供的方法，真实签名是 `aget_state(config: RunnableConfig, *, subgraphs: bool = False) -> StateSnapshot`。`config` 是一个形如 `{"configurable": {"thread_id": "..."}}` 的字典，`thread_id` 就是 4.1 节里存进 SQLite 的那个 Checkpoint 记录的分区键。返回一个 `StateSnapshot`：如果这个 `thread_id` 之前跑过、有 Checkpoint，就能拿到"上次跑到哪一步"的完整 State；没有则返回一个空快照，等价于"从头开始"。
- **`configurable.pop("checkpoint_ns", None)`**：`checkpoint_ns`（namespace）是 LangGraph 内部专门留给**嵌套子图**的字段——如果一个 Graph 里又调用了另一个 Graph（子图），子图的 Checkpoint 要和外层区分命名空间。本项目三个业务 Agent 全在一个顶层 Graph 里，没有嵌套子图，如果不小心把这个字段传进去（比如误用了别的地方留下的 config），LangGraph 会把它当成"某个子图的 State"去查，查不到就抛 `Subgraph ... not found`。**这是本项目真实踩过的一个 bug**（对应 [SPEC.md](../../ai-context/SPEC.md) 变更记录 4.3 版本："修复将 LangGraph 根图 checkpoint_ns 误作业务版本标签导致的 Subgraph not found"）。

### 4.3 真正驱动 Graph 执行、并自动落 Checkpoint

[graph/ 包内组装出的同一个 Graph](../../backend/app/agent/graph/orchestrator.py)（`legal_researcher` 等节点方法现在按阶段拆分在 `graph/nodes/` 目录下，通过多继承组合进 `orchestrator.py` 的 `LegalConsultationGraph`）由 [runtime.py:129](../../backend/app/agent/runtime.py#L129) 驱动：

```python
async for part in graph.compiled.astream(
    input, config, stream_mode=["updates", "custom"], version="v2",
):
```

- **`astream`**：真正把 Graph 往下推进的方法，每完成一个 super-step 就**自动**调用 Checkpoint Saver 落一次快照——业务代码完全不感知这个动作。它是一个异步生成器，边跑边把每一步的产出 `yield` 出来。
- `stream_mode=["updates", "custom"]`：控制它往外吐什么——`"updates"` 是每个节点返回的 State 增量字典，`"custom"` 是节点内部用 `runtime.stream_writer(...)` 主动写出的自定义事件（本项目用来发 SSE 的 `tool_call_start` 之类事件，参见 [01-sse.md](01-sse.md)）。这两路数据混在同一个异步生成器里吐出来，靠 `part.get("type")` 区分。

### 4.4 只读元数据、失败时的两种不同容错策略

[runtime.py:243-250](../../backend/app/agent/runtime.py#L243)（读取一个已完成/进行中任务的补充信息，比如给前端展示）：

```python
config = {"configurable": {"thread_id": thread_id}}
try:
    snapshot = await graph.compiled.aget_state(config)
except Exception as exc:
    self._audit_checkpoint_read_failure(thread_id, "final_metadata", exc)
    return {}
```

这里读失败走的是**降级**：记一条审计日志，返回空字典，不影响已经生成并保存的回答正文。

对比 4.2 节"发起新执行前"的 `aget_state`——那里读失败是**严格终止本次 attempt**，不会静默地从初始 State 重新跑一遍。这两处用同一个框架函数，但容错策略完全相反，原因是语义不同：

- 4.4 节的场景：State 都已经跑完了，只是想读个补充元数据展示给用户，读不到不影响已经产出的结果。
- 4.2 节的场景：正要决定"这次是从头开始还是接着上次跑"，如果读不到真实状态却按"从头开始"处理，可能导致工具调用重复执行、配额重复消耗，甚至（未来如果引入有副作用的工具）产生重复副作用——**读不到就不能猜，必须让本次尝试失败**，交给上层重试机制处理。

## 5. 对比：本项目 vs 行业常规方案

- **相比"手工任务表 + 手动埋点"**：本项目的业务节点函数（`case_analyst`/`legal_researcher`/...）完全不需要写"现在存一下进度"的代码，State 变了框架自动帮你存——不容易漏埋点，业务逻辑和持久化逻辑解耦。
- **相比"事件溯源 + 重放"**：本项目直接存快照、重启后从快照继续跑，不要求节点逻辑"重放安全"（不用担心"如果这一步重放会不会把一次真实 API 调用又打一遍"）——心智负担更低，但代价是丢失了"回放到任意历史时间点、完整审计每一步中间过程"的能力，本项目目前也确实没有这个需求。
- **本项目在框架能力之外补的一层**（这是通用 Checkpoint 库本身不会替你想的）：Checkpoint 只管"Graph 执行状态"，本项目又用 [AgentRunManager](../../backend/app/services/agent_runs.py) 管了一层**业务任务**（所有权归属、排队、取消、SSE sequence 重放）。原因是 Checkpoint 库不知道这个任务归哪个用户、要不要允许被取消、前端断线重连该从哪条 SSE 消息开始补发——这些都是业务语义，通用框架天然不覆盖，必须自己在它上面补。

## 6. 本项目内部的关键设计取舍与易错点

- `checkpoint_ns` 误用导致的 `Subgraph not found`（见 4.2）——通用教训：使用框架预留字段前，先搞清楚它的语义边界，不要"看起来能用就用"。
- 两处 `aget_state` 失败容错策略不同（见 4.4）——通用教训：同一个 API 在不同调用场景下，"失败了该怎么办"要结合业务语义单独设计，不能一刀切。
- Checkpoint 数据库（`data/runtime/langgraph-checkpoints.db`）和业务 SQLite（`data/runtime/lawstation.db`）物理分离——避免 Graph 每个 super-step 的高频写入和消息/记忆的业务事务互相抢锁。
- Checkpoint **不是**用来做跨轮长期记忆的——它只保存"这一次 AgentRun 执行到哪了"，跨会话的用户偏好/案件事实走的是完全独立的分层记忆机制（见 [03-memory-context-engineering.md](03-memory-context-engineering.md)），两者不要混为一谈。

## 7. 动手验证方式

1. 跑一个真实对话，中途手动 `kill` 掉 `python run.py` 进程，重启后从前端刷新页面，观察这个进行中的任务能否续上。
2. 直接看 Checkpoint 数据库里存了什么：
   ```bash
   sqlite3 data/runtime/langgraph-checkpoints.db ".tables"
   sqlite3 data/runtime/langgraph-checkpoints.db "select thread_id, checkpoint_ns from checkpoints limit 5;"
   ```
3. 对照着读一遍 `LegalConsultationState`（[state.py:18](../../backend/app/agent/state.py#L18)）——这就是每次快照实际存的内容结构；再读一遍 `AgentInvocationContext`（[state.py:67](../../backend/app/agent/state.py#L67)）——对比一下这个**不会**被存进 Checkpoint 的东西是什么，为什么它不能被存（提示：里面有请求级、不可序列化或不该跨请求复用的对象）。
