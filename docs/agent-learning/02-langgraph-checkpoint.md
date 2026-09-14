# LangGraph Checkpoint 持久化机制

> 面向：知道"数据库事务"但没接触过"工作流引擎状态持久化"的后端工程师。
> 目标：理解 Checkpoint 到底在存什么、为什么不是"再建一张业务表"就能替代，本项目怎么在它之上又搭了一层业务持久化，以及这套 SQLite 方案离生产部署还差什么。

## 0. 前置知识

建议先读 [04-langgraph-stategraph.md](04-langgraph-stategraph.md) 了解 State 是什么、Graph 是怎么由节点和边组成的——Checkpoint 保存的正是这个 State 在某一时刻的快照，没有这个背景会不好理解"到底保存了什么"。读完本篇后，接着看 [07-agent-run-persistence.md](07-agent-run-persistence.md)，理解 Checkpoint 之上还补了哪一层业务持久化。

## 1. 要解决的问题

一次法律咨询回答，背后是 Case Analyst → Legal Research（内部还有若干轮工具调用）→ Legal Counsel → Reviewer 这样一条链路，实际耗时可能是几十秒。这期间如果发生：

- 服务进程被重启（部署、崩溃、手动重启）
- 用户刷新了页面 / 断网重连
- 前端主动关闭了 SSE 连接（但没有调用"取消"接口）

这个"跑到一半"的执行状态要怎么办？直接丢弃重跑，代价是重复消耗 LLM/工具调用配额，用户体验也差（进度条归零重来）。我们想要的是：**从上次执行到的地方继续，而不是从头开始**。

## 2. 核心机制原理

### 2.1 三条常见路线

这类"长时间运行的多步骤任务，要求可恢复"的问题，业界大致有三条路：

- **工作流引擎路线**（Temporal、Cadence、AWS Step Functions）：不保存"当前状态"这一个快照，而是保存**完整的事件历史**（每一步做了什么、返回了什么），恢复时通过**重放（replay）**这些历史事件重新算出当前应该处于什么状态。优点是可审计、理论上能回溯到任意历史时刻；代价是业务代码必须写成"确定性重放安全"的（比如不能在重放时重复真正调用一次外部 API），框架对代码风格有较强约束。如果本项目走这条路，`legal_researcher` 节点里每一次真实的 MCP 调用都要包一层"重放时跳过、直接读历史结果"的逻辑，比现在的实现复杂得多。
- **手工任务表路线**（很多自研系统 / Celery 之类的任务系统）：一张 `jobs` 表存 `status` 字段，业务代码自己在关键节点手动 `UPDATE progress = ...`。优点是简单直接、心智负担低；缺点是"进度"粒度完全靠开发者手动埋点，容易漏埋、容易和业务逻辑搅在一起，扩展到复杂分支流程时代码会变得难维护。
- **框架级状态快照路线**（LangGraph、以及一些 Actor 模型框架）：框架知道你的"工作流"是由哪些节点（node）组成的，每执行完一个节点就自动把当前完整状态序列化落盘，不需要业务代码手动埋点，代价是你要接受用它的 Graph/Node 抽象来组织业务逻辑，不能完全按自己的想法写控制流。

本项目走的是第三条路——这是理解本项目 Checkpoint 设计的前提：**它是 LangGraph 框架自动提供的能力，不是本项目自己发明的机制**。

### 2.2 核心概念：super-step 与快照

Checkpoint 保存的是某个 `thread_id`（一次执行的唯一标识）在某个 **super-step**（一轮节点执行）完成后的**完整 State 快照**。恢复时不是"从头重新推理一遍"，而是"读出最近一次快照的 State，直接从后面还没跑的节点继续跑"。

关键在于：这个"自动存快照"的动作完全由框架在幕后完成，业务节点函数（比如 `case_analyst`、`legal_researcher`）只管接收 State、返回增量 dict，不需要自己调用任何"保存"函数——这正是它和"手工任务表路线"最大的区别：进度追踪不需要手动埋点。

## 3. 本项目具体实现（函数级）

### 3.1 Checkpoint 存储后端的搭建

[checkpoint.py:16-43](../../backend/app/agent/checkpoint.py#L16)：

```python
@asynccontextmanager
async def checkpoint_saver(settings: Settings):
    if not settings.langgraph_checkpoint_enabled:
        yield None
        return
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

- **`langgraph_checkpoint_enabled`（默认 `True`）**：Checkpoint 其实是可以被整个关掉的配置开关。关掉之后这个函数直接 `yield None`，不会创建任何 `AsyncSqliteSaver`——`AgentRuntime`/`AgentRunManager` 里好几处都专门判断 `self.checkpointer is None` 来处理这种情况（比如 `checkpoint_info()` 一进来就 `if self.checkpointer is None: return {}`）。理解这个开关的存在，下面讲的"每个 super-step 自动落盘"才有一个明确的前提：它是在 Checkpoint 启用的情况下才成立。
- **`AsyncSqliteSaver`**：LangGraph 官方提供的 Checkpoint 存储后端实现之一（还有 Postgres 版等，见 §6），负责"把 State 序列化写入/从 SQLite 读出"这件事的具体落地。它实现了 LangGraph 定义的 `BaseCheckpointSaver` 接口，Graph 编译时把它传进去，之后每个 super-step 自动调用它保存。
- 构造参数 `connection`：`aiosqlite`（异步 SQLite 客户端）的连接对象，**不是**本项目业务用的 SQLAlchemy 连接——这是一条独立的、专属 Checkpoint 的数据库连接，物理上和业务库（`data/runtime/lawstation.db`）是两个文件。
- `serde=JsonPlusSerializer(pickle_fallback=False)`：`serde`（serializer/deserializer）决定 State 里的 Python 对象怎么变成字节存进 SQLite。这里显式关闭 `pickle_fallback`——遇到序列化器不认识的类型会直接报错，而不是退化用 Python 的 `pickle` 兜底。这是有意的安全选择：`pickle` 反序列化任意数据是已知的安全风险（构造恶意 pickle 数据可以在反序列化时执行任意代码），Checkpoint 数据库理论上是持久化文件，不应该允许"读出来就能跑代码"这种攻击面。
- **`saver.setup()`**：只创建 Checkpoint 自己需要的表结构（`checkpoints`/`writes` 等），不会编译或运行任何 Graph 节点，也不碰业务表。

### 3.2 发起一次新的 / 续跑的 Agent 执行

[runtime.py:107-126](../../backend/app/agent/runtime.py#L107)：

```python
configurable["thread_id"] = (
    thread_id or configurable.get("thread_id") or context.identity.request_id
)
configurable.pop("checkpoint_ns", None)
config["configurable"] = configurable
graph_input: LegalConsultationState | None = state   # 默认：全新构造的初始 State
if resume and self.checkpointer is not None:
    snapshot = await graph.compiled.aget_state(config)
    if snapshot and snapshot.values:
        final_state.update(snapshot.values)
        context.metrics.model_call_count = int(snapshot.values.get("model_call_count", 0))
        context.metrics.tool_call_count = int(snapshot.values.get("tool_call_count", 0))
        context.metrics.tool_trajectory = list(snapshot.values.get("tool_trajectory", []))
        graph_input = None
```

**关键点，容易理解错：`aget_state()` 不是每次执行都会调用的**。`graph_input` 默认就是上面刚构造好的全新 `state`——一次全新的、第一次尝试的 Run，代码根本不会去碰 Checkpoint，直接拿这个新 State 往下跑。**只有 `resume=True`（意味着这是一次 `attempt > 1` 的恢复执行，见 [07-agent-run-persistence.md](07-agent-run-persistence.md)）且 Checkpointer 存在时，才会调用 `aget_state()`**，把上次的 State 读回来、把持久化的调用计数同步回 Context，并且只有在这条分支里，`graph_input` 才会被改成 `None`——这才是"从最近 Checkpoint 的下一节点继续"的真正触发条件，不是"每次都查一下、查不到就当新的"。

- **`graph.compiled`**：`StateGraph.compile()` 的返回值，类型是 `CompiledStateGraph`——把节点/边的拓扑关系固化成一个可执行对象。编译这一步本身**不会**发起任何模型调用，纯粹是结构组装。
- **`aget_state(config)`**：`CompiledStateGraph` 提供的方法，真实签名是 `aget_state(config: RunnableConfig, *, subgraphs: bool = False) -> StateSnapshot`。`config` 是一个形如 `{"configurable": {"thread_id": "..."}}` 的字典，`thread_id` 就是 §3.1 里存进 SQLite 的那个 Checkpoint 记录的分区键。只有在恢复分支里被调用时，返回的 `StateSnapshot` 才会被使用；这个函数本身在没有 Checkpoint 的 `thread_id` 上也能调用、会返回一个空快照，但本项目的正常新建流程根本用不到这个"空快照"分支，因为新建从一开始就不会走到这条调用。
- **`configurable.pop("checkpoint_ns", None)`**：`checkpoint_ns`（namespace）是 LangGraph 内部专门留给**嵌套子图**的字段——如果一个 Graph 里又调用了另一个 Graph（子图），子图的 Checkpoint 要和外层区分命名空间。本项目三个业务 Agent 全在一个顶层 Graph 里，没有嵌套子图，如果不小心把这个字段传进去（比如误用了别的地方留下的 config），LangGraph 会把它当成"某个子图的 State"去查，查不到就抛 `Subgraph ... not found`。**这是本项目真实踩过的一个 bug**（对应 [SPEC.md](../../ai-context/SPEC.md) 变更记录 4.3 版本："修复将 LangGraph 根图 checkpoint_ns 误作业务版本标签导致的 Subgraph not found"）。

### 3.3 真正驱动 Graph 执行、并自动落 Checkpoint

[graph/ 包内组装出的同一个 Graph](../../backend/app/agent/graph/orchestrator.py)（`legal_researcher` 等节点方法现在按阶段拆分在 `graph/nodes/` 目录下，通过多继承组合进 `orchestrator.py` 的 `LegalConsultationGraph`）由 [runtime.py:129](../../backend/app/agent/runtime.py#L129) 驱动：

```python
async for part in graph.compiled.astream(
    graph_input,
    context=context,
    config=config,
    stream_mode=["updates", "custom"],
    version="v2",
):
```

- **`astream`**：真正把 Graph 往下推进的方法，每完成一个 super-step 就**自动**调用 Checkpoint Saver 落一次快照——业务代码完全不感知这个动作。它是一个异步生成器，边跑边把每一步的产出 `yield` 出来。
- **`graph_input`**：就是 §3.2 里那个变量——新建时是全新构造的 State，恢复时是 `None`（表示"从最近 Checkpoint 的下一节点继续"，不是重新提交一份初始 State）。
- **`context=context`**：把这次调用专属的 `AgentInvocationContext` 对象传进去，这正是 [04-langgraph-stategraph.md](04-langgraph-stategraph.md) §3.3 讲的 State/Context 机制里，Context 真正"进入"Graph 执行、被节点函数通过 `runtime.context` 读到的地方——不传这个参数，节点内部就拿不到本次调用的身份、调用计数这些请求级依赖。
- `stream_mode=["updates", "custom"]`：控制它往外吐什么——`"updates"` 是每个节点返回的 State 增量字典，`"custom"` 是节点内部用 `runtime.stream_writer(...)` 主动写出的自定义事件（本项目用来发 SSE 的 `tool_call_start` 之类事件，参见 [01-sse.md](01-sse.md)）。这两路数据混在同一个异步生成器里吐出来，靠 `part.get("type")` 区分。

### 3.4 只读元数据、失败时的两种不同容错策略

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

对比 §3.2"发起新执行前"的 `aget_state`——那里读失败是**严格终止本次 attempt**，不会静默地从初始 State 重新跑一遍。这两处用同一个框架函数，但容错策略完全相反，原因是语义不同：

- 这里的场景：State 都已经跑完了，只是想读个补充元数据展示给用户，读不到不影响已经产出的结果。
- §3.2 的场景：正要决定"这次是从头开始还是接着上次跑"，如果读不到真实状态却按"从头开始"处理，可能导致工具调用重复执行、配额重复消耗，甚至（未来如果引入有副作用的工具）产生重复副作用——**读不到就不能猜，必须让本次尝试失败**，交给上层重试机制处理。

## 4. 设计取舍

**为什么走"框架级状态快照"而不是"事件溯源 + 重放"？** 本项目直接存快照、重启后从快照继续跑，不要求节点逻辑"重放安全"（不用担心"如果这一步重放会不会把一次真实 API 调用又打一遍"）——心智负担更低，但代价是丢失了"回放到任意历史时间点、完整审计每一步中间过程"的能力，本项目目前也确实没有这个需求。

**为什么 Checkpoint 数据库和业务数据库要物理分离？** 避免 Graph 每个 super-step 的高频写入和消息/记忆的业务事务互相抢锁——两条写入路径的频率和事务边界完全不同，合在一个文件里容易互相拖慢。

**为什么 Checkpoint 之上还要再补一层业务持久化（`AgentRunManager`）？** Checkpoint 只管"Graph 执行状态"，本项目又用它管了一层**业务任务**（所有权归属、排队、取消、SSE sequence 重放，详见 [07-agent-run-persistence.md](07-agent-run-persistence.md)）。原因是 Checkpoint 库不知道这个任务归哪个用户、要不要允许被取消、前端断线重连该从哪条 SSE 消息开始补发——这些都是业务语义，通用框架天然不覆盖，必须自己在它上面补。

## 5. 易错点

- **`checkpoint_ns` 误用导致的 `Subgraph not found`**（见 §3.2）——通用教训：使用框架预留字段前，先搞清楚它的语义边界，不要"看起来能用就用"。本项目真实踩过这个坑：曾经把业务版本号误传进这个字段，导致 LangGraph 把它当成子图路径去查，查不到就报错。
- **误以为两处 `aget_state` 用的是同一套失败处理逻辑**（见 §3.4）——实际正相反，两处的容错策略是**刻意设计成相反的**：§3.2 恢复执行时读失败会让本次尝试直接失败，交给上层重试；§3.4 读补充元数据时失败则安静降级返回空字典。通用教训：同一个 API 在不同调用场景下，"失败了该怎么办"要结合业务语义单独设计，绝不能因为调用方式看着差不多就复制粘贴同一段 `try/except`。
- **误以为新建一次 Run 也会调用 `aget_state()`**（见 §3.2）——实际上只有 `resume=True` 的恢复执行才会调用它；新建流程直接用全新构造的 State，根本不查 Checkpoint。混淆这一点会导致误判"是不是每次请求都有一次多余的数据库读"。
- **把 Checkpoint 当成跨轮长期记忆**——它只保存"这一次 AgentRun 执行到哪了"，跨会话的用户偏好/案件事实走的是完全独立的分层记忆机制（见 [03-memory-context-engineering.md](03-memory-context-engineering.md)），两者不要混为一谈。混淆的后果是：以为某个信息"应该能通过 Checkpoint 恢复"，结果发现新会话根本读不到——因为它本来就不该从这里读。

## 6. 生产化差距与面试应对

LangGraph Checkpoint 机制本身在语义上是完整的，但当前的**存储后端选型**是明确为单机场景服务的，面试被追问时应该主动说清楚这个边界：

| 维度 | 本项目现状 | 生产环境通常怎么做 | 面试怎么答 |
|---|---|---|---|
| 存储后端 | `AsyncSqliteSaver`，单文件 SQLite，单进程独占访问 | LangGraph 官方生产推荐 `AsyncPostgresSaver`（`langgraph-checkpoint-postgres`），支持多实例并发读写；或直接使用 LangGraph Platform 托管服务 | "SQLite 适合单机验证，不支持多进程并发写入；如果要多实例部署，第一步是把 Checkpoint 后端换成 Postgres，接口不用变，只是存储实现不同" |
| 序列化安全 | `JsonPlusSerializer(pickle_fallback=False)`，已经是相对安全的选择 | 生产环境通常还会加上 Checkpoint 数据的加密存储（敏感业务状态落盘） | "已经关闭了 pickle 回退避免反序列化攻击面；如果 State 里包含更敏感的信息，下一步会考虑落盘加密" |
| 保留策略 | 依赖 `AGENT_RUN_EVENT_RETENTION_DAYS` 间接触发清理，`LANGGRAPH_CHECKPOINT_RETENTION_DAYS` 配置存在但未独立生效 | 独立的数据生命周期管理策略，按合规要求（如需要保留多久）单独配置 | "这是当前一处已知的配置技术债，两个保留期目前耦合在一起" |
| 跨实例可见性 | 单机单进程，`thread_id` 只在本机 SQLite 里有意义 | 多实例部署下，任意实例都要能通过共享的 Checkpoint 后端恢复任意 `thread_id` 的状态 | "当前架构是单实例假设；换成 Postgres 后端后，多个应用实例可以共享同一份 Checkpoint 数据，这是水平扩展的前提" |

## 7. 动手验证方式

1. 跑一个真实对话，中途手动 `kill` 掉 `python run.py` 进程，重启后从前端刷新页面，观察这个进行中的任务能否续上。
2. 直接看 Checkpoint 数据库里存了什么：
   ```bash
   sqlite3 data/runtime/langgraph-checkpoints.db ".tables"
   sqlite3 data/runtime/langgraph-checkpoints.db "select thread_id, checkpoint_ns from checkpoints limit 5;"
   ```
3. 对照着读一遍 `LegalConsultationState`（[state.py:18](../../backend/app/agent/state.py#L18)）——这就是每次快照实际存的内容结构；再读一遍 `AgentInvocationContext`（[state.py:67](../../backend/app/agent/state.py#L67)）——对比一下这个**不会**被存进 Checkpoint 的东西是什么，为什么它不能被存（提示：里面有请求级、不可序列化或不该跨请求复用的对象）。

**自测题：**

- 如果把 `checkpoint_ns` 设置成一个业务版本号（比如 `"v2"`），下一次 `aget_state()` 会发生什么？为什么会报 `Subgraph not found` 而不是正常返回空状态？
- 假设要把 Checkpoint 后端从 SQLite 换成 Postgres，业务节点函数（`case_analyst`、`legal_researcher` 等）的代码需要改动吗？为什么？
