# Agent 持久化：AgentRun 业务任务层

> 面向：知道 LangGraph Checkpoint 能保存 Graph 状态，但没想过"页面刷新后任务还在跑"这件事光有 Checkpoint 是不够的后端工程师。
> 目标：搞清楚"业务任务层"和"Graph 执行层"到底谁负责什么、幂等写入具体怎么做到、以及当前这套单机方案离真正的分布式任务队列还差多远。

## 0. 前置知识

建议先读 [02-langgraph-checkpoint.md](02-langgraph-checkpoint.md)，理解 LangGraph Checkpoint 本身能做什么、不能做什么——本篇要讲的 AgentRun 业务任务层，正是为了补上 Checkpoint 天生不负责的那部分（任务所有权、排队、取消、前端断线重连）。

## 1. 要解决的问题

一次法律咨询的完整链路可能跑几十秒：排队、案情分析、法规检索、生成意见、复核，中间还要调用模型和 MCP 工具。如果把这整段执行直接绑定在浏览器的 SSE 连接上——用户刷新页面、网络抖动断开、甚至只是切换到另一个标签页——都可能意外中断一次本该继续进行的任务。反过来，如果只在内存里记录"进度到哪了"，服务进程重启（部署、崩溃、手动重启）就会让所有正在跑的任务凡失得无影无踪，用户看到的是一个永远不会回来的"正在思考中"。

这是"长时间运行的 Agent 任务"这类场景的通用问题：**执行的生命周期，不能等于浏览器连接的生命周期，也不能只依赖进程内存**。

## 2. 核心机制原理

### 2.1 为什么"任务状态"和"执行进度"要分两层保存

行业里对这类问题的通用解法是把"任务"本身持久化成一个可查询、可恢复的实体，独立于任何一次具体的执行尝试：

- **任务层**：这个任务属于谁、当前是什么状态（排队/运行/完成/失败）、允许重试几次、结果在哪——这一层通常落在关系型数据库或专门的任务队列（Celery + Redis/RabbitMQ、AWS Step Functions、Temporal 的 Workflow）。
- **执行层**：如果任务本身是一个多步骤的状态机（比如这里的三 Agent Graph），执行层还要记录"跑到哪一步了、下一步该跑什么"——这是编排框架自己的持久化职责，Temporal 叫 Workflow History，LangGraph 叫 Checkpoint。

两层分开的好处是**关注点分离**：任务层不需要理解 Graph 内部的节点拓扑，执行层也不需要理解"谁能看这个任务、要不要发邮件通知"这类业务规则。合在一起写，通常最后会变成一个既难懂又难改的巨石状态机。

### 2.2 断线重连怎么做到"不丢消息、不重复消息"

浏览器和服务端之间用 SSE 或 WebSocket 保持连接时，网络抖动导致的断线重连是常态。行业里标准做法是给每条推送事件加一个**单调递增的序号**（等价于 SSE 协议原生支持的 `id:` 字段），客户端记住最后收到的序号，重连时告诉服务端"从这个序号之后继续给我"，服务端只需要能查询"某个任务里，序号大于 N 的全部历史事件"就能补齐断线期间错过的内容——这正是 SSE 规范里 `Last-Event-ID` 请求头存在的原因，Kafka 的 consumer offset、数据库 CDC 的 binlog position 也是同一个模式的不同应用。

### 2.3 幂等写入：应对"写完了但没确认成功"这类崩溃窗口

任何跨越多个步骤的持久化操作，都存在"某一步已经执行、但记录这件事发生了的那一步还没完成"的崩溃窗口。标准解法是给关键写入结果一个可复用的标识（幂等键），恢复时先查这个标识是否已经存在，存在就直接复用而不是重新执行一遍。这正是分布式系统里"至少一次执行 + 幂等写入 = 等价于精确一次的业务效果"这条经典组合的具体应用。

## 3. 本项目具体实现（函数级）

### 3.1 两层持久化的分工

```text
业务任务层：AgentRun + AgentRunEvent（业务 SQLite）
├── 所有权、排队、状态机和同会话互斥
├── Worker 领取、并发配额、租约和取消
├── 最终消息幂等写入
└── SSE sequence 事件日志与断线重放

Graph 执行层：LangGraph AsyncSqliteSaver（独立 Checkpoint SQLite）
├── super-step 边界保存 Graph State
├── 记录已完成节点、下一步该跑什么
├── 恢复案情、证据、草稿和复核状态
└── 恢复模型/工具调用计数
```

两个数据库物理分离（`data/runtime/lawstation.db` 和 `data/runtime/langgraph-checkpoints.db`），避免 Graph 高频状态写入和消息/记忆事务争用同一个 SQLite 文件。入口在 [backend/app/main.py::lifespan](../../backend/app/main.py)：Checkpoint Saver 的生命周期覆盖整个应用运行期，`AgentRunManager` 在接受任何请求之前先完成一次恢复扫描，再启动后台 Worker。

### 3.2 创建任务：HTTP 只负责入队

[routes/agent_runs.py::create_agent_run](../../backend/app/api/routes/agent_runs.py)：请求进来后只做四件事——预占会话并发身份、校验会话所有权、创建一条 `status=queued` 的 `AgentRun`、返回 HTTP `202`。`202` 的语义是"已接受、已排队"，不是"已经跑完"；真正的模型调用完全在后台 Worker 里发生，因此这个 HTTP 请求结束不会终止 Agent 执行。

每个 Run 固定 `langgraph_thread_id = f"agent-run:{run_id}"`，**不用** `conversation_id` 做 thread ID——如果用会话 ID，同一会话的下一轮咨询会在 LangGraph 层继承上一轮的 `EvidencePacket`、草稿等中间状态，等于凭空多出一套跟业务消息、`MemoryService` 打架的"第二套跨轮记忆"。

同一会话的互斥用了两层保护：进程内的 `AgentConcurrencyManager.reserve()` 立即拒绝重复提交，加上数据库的部分唯一索引 `uq_agent_run_active_conversation` 兜底并发穿透的竞争窗口——冲突返回 HTTP `409`。

### 3.3 Worker 调度、租约与并发准入

`AgentRunManager` 后台每 `0.5` 秒轮询最早的 `queued` Run，为尚未在本进程 `_active` 集合里的 Run 创建 `asyncio.Task`。真正执行前必须先拿到全局/用户并发配额（`AGENT_GLOBAL_CONCURRENCY=6`、`AGENT_PER_USER_CONCURRENCY=2`），拿到配额后才把任务 `claim` 为 `running`，写入 `lease_owner`（当前 Worker 标识）和 `lease_expires_at`（默认 120 秒，每 40 秒续约一次）。排队期间不会调用模型或 MCP，也不会提前读取可能已经过期的记忆快照。

### 3.4 消息与最终结果的幂等写入

`_prepare()` 保存用户消息时先检查 `user_message_id` 是否已存在：

```python
if current.user_message_id:
    user_message = db.get(Message, current.user_message_id)
else:
    user_message = Message(role="user", content=run.input_text)
    current.user_message_id = user_message.id
```

`_complete()` 对最终助手消息做了三层幂等检查：任务已完成且已有助手消息直接复用；助手消息不存在才创建；正文 Token 事件已存在就不再重复写。这意味着即使进程恰好在"助手消息已保存、任务状态还没改成 completed"这个窗口崩溃，恢复后也不会创建第二条助手回答，也不会重复推送已经发出的正文事件。

### 3.5 SSE sequence 重放

[routes/agent_runs.py::agent_run_events](../../backend/app/api/routes/agent_runs.py)：服务端取 Query 参数 `after_sequence` 和 `Last-Event-ID` 请求头两者的较大值作为游标，查询 `sequence > cursor` 的历史事件补发，发完存量事件后如果任务仍在运行就通过进程内 `asyncio.Condition` 等待新事件，默认 15 秒没有新事件发一次 `: heartbeat` 保活。**heartbeat 不占用序号、不写数据库，也不参与重放**——它只是防止连接被中间代理判定为空闲断开。浏览器主动断开只会终止这一次订阅生成器，不会取消后台正在执行的任务，这正是"页面刷新后任务继续跑"的关键实现点。

### 3.6 服务重启恢复

启动时 `_recover_stale()` 处理全部遗留的 `running` Run：`attempt` 小于恢复上限（默认 2）就清空租约重新置为 `queued`，达到上限就标记 `failed`。重新排队的任务被 Worker 再次领取后，`AgentService` 会传入 `resume=True`，[runtime.py](../../backend/app/agent/runtime.py) 用同一个 `thread_id` 调用 `graph.compiled.aget_state()` 取回最近一次 Checkpoint 的 State，把 `model_call_count`/`tool_call_count` 也一并恢复进请求级 Context——**恢复后调用计数不会归零**，避免服务重启变成绕过模型/工具调用上限的漏洞。

## 4. 设计取舍

**为什么不能只用 LangGraph Checkpoint？** Checkpoint 只回答"Graph 跑到哪个节点了"，完全不负责任务归属于谁、要不要排队、能不能取消、前端从哪个序号继续订阅、最终消息是否已经保存——这些都是业务任务层的职责，硬塞进 Checkpoint 会让它变成一个不该承担的"第二数据库"。

**为什么每个 Run 一个独立 thread_id，而不是每个会话一个？** 见 §3.2。核心是防止 Graph 层意外变成跨轮记忆的重复来源。

**为什么最终正文不是模型 Token 实时落盘，而是分块写入？** [runtime.py::stream](../../backend/app/agent/runtime.py) 等 Graph 完成复核和 Finalize 之后，才把批准的 `final_answer` 切成固定长度的 Token 事件推送。这样做的代价是牺牲"逐字实时输出"的观感，换来的收益是**未经复核的草稿绝不会被推送给用户**——推理过程中的状态事件和心跳维持连接感，但正文本身要等安全校验通过。

**为什么用租约而不是简单的"谁先抢到就是谁的"？** 租约（`lease_owner` + `lease_expires_at`）是为了让"某个 Worker 意外挂了、任务卡在 running 状态"这种情况能被下一次启动的恢复扫描识别并重新排队，而不是永久卡死。当前只有单机单 Worker，所以租约暂时只承担"存活标记"的作用，还没有做成能被多个 Worker 实例竞争抢占的分布式锁（见 §6）。

## 5. 易错点

- **把"断线"和"取消"混为一谈**：浏览器断开 SSE 只终止订阅，不会设置 `cancel_requested`，Graph 继续在后台跑；只有显式调用取消接口才会真正中断任务。搞混这两者会导致"用户以为取消了，结果任务还在跑并且最终还是生成了回答"的困惑。
- **恢复时重新执行整个 Graph、而不是从最近节点继续**：`aget_state()` 拿到快照后传 `graph_input=None`，表示"从最新 Checkpoint 的下一节点继续"，不是重新提交初始 State；如果不小心传了非空的初始 State，等于每次恢复都从头跑一遍，浪费已经完成的模型/工具调用。
- **把 `latest_checkpoint_id` 当作运行中的实时进度指针**：这个字段只在最终回答持久化时才被写回，运行过程中真正权威的最新节点位置始终以 Checkpoint 数据库为准，不能拿业务表这个字段做实时监控。
- **以为 SSE heartbeat 和数据库租约续约是同一回事**：两者名字相似但作用对象完全不同——SSE heartbeat 保的是浏览器 HTTP 连接，租约续约保的是 Worker 对任务的持有权，混淆会导致排查问题时找错日志、看错指标。

## 6. 生产化差距与面试应对

这套双层持久化设计（AgentRun + Checkpoint）在单机场景下已经把该覆盖的语义都覆盖了，但离真正的生产级分布式任务系统还有明确差距，主动讲清楚比被问到才承认更专业：

| 维度 | 本项目现状 | 生产环境通常怎么做 | 面试怎么答 |
|---|---|---|---|
| Worker 抢占 | 单机单 Worker 轮询 + `_active` 内存集合防重复；租约续约不涉及跨实例竞争 | 多个 Worker 实例通过数据库条件更新（`UPDATE ... WHERE status='queued' AND lease_expires_at < now()`）或专门的分布式锁/租约服务（etcd、ZooKeeper）竞争领取任务 | "当前是单实例，claim 逻辑没有做成多 Worker 可竞争的原子操作；如果要横向扩容，第一步是把领取任务改成带条件的原子 UPDATE，而不是先查询再更新" |
| 任务队列基础设施 | SQLite 轮询（0.5 秒间隔） | Redis/RabbitMQ/Kafka 驱动的消息队列，或托管的工作流引擎（Temporal、AWS Step Functions、Celery + broker） | "SQLite 轮询在当前规模足够、部署简单；规模上升后会评估迁移到专门的任务队列，好处是原生支持多消费者竞争和更细粒度的重试/死信策略" |
| 恢复语义 | Graph 节点"至少一次"、业务结果幂等；只读 MCP 工具重跑可接受 | 需要副作用（发邮件、扣款、调用第三方 API）的节点必须有独立的幂等键和外部操作记录，通常结合 Saga 模式或 Outbox 模式 | "当前所有工具都是只读检索，节点重跑没有副作用风险；如果未来加入有副作用的工具，必须补充幂等键，不能只依赖 Checkpoint 的至少一次语义" |
| Checkpoint 存储 | `AsyncSqliteSaver`，单文件 SQLite | LangGraph 官方生产推荐 Postgres Checkpointer（`langgraph-checkpoint-postgres`），或托管的 LangGraph Platform | "SQLite 适合单机部署；如果要多实例共享 Checkpoint 状态，需要换成 Postgres 或其他支持并发写入的存储" |
| 可观测性 | 本地 JSONL 审计 + 可选 LangSmith Trace，无专门的任务监控面板 | 专门的任务队列通常自带监控面板（Celery Flower、Temporal Web UI），支持死信队列、重试次数告警 | "当前没有专门的运维面板，排查依赖日志和数据库查询；生产化会补充任务级监控和告警" |
| 事件保留 | `AGENT_RUN_EVENT_RETENTION_DAYS` 同时驱动 Event 和 Checkpoint 清理，`LANGGRAPH_CHECKPOINT_RETENTION_DAYS` 配置存在但未独立生效 | 独立的数据生命周期管理，不同类型数据可以有不同的保留策略并可审计 | "这是当前一处已知的配置技术债，两个保留期配置目前耦合在一起，还没有拆开成独立生效" |

## 7. 动手验证方式

1. 发起一次咨询后立刻刷新页面，观察前端是否通过 `/active-run` 接口找回正在运行的任务，并从正确的 sequence 继续接收事件而不是重新提交请求。
2. 发起一次咨询，在任务进入 `running` 后手动重启后端服务，观察重启后任务是先被标记恢复重新排队、再从最近的 Checkpoint 节点继续，而不是从头重新跑一遍案情分析。
3. 对一个 `queued` 状态的任务和一个 `running` 状态的任务分别调用取消接口，对比两者的行为差异（`queued` 直接终态、`running` 需要先设置取消标记再中断协程）。

**自测题：**

- 如果服务进程恰好在"助手消息已经写入数据库、但任务状态字段还没改成 `completed`"这个时间点崩溃，重启恢复后会发生什么？会不会产生第二条助手消息？
- 为什么 `thread_id` 要绑定到 `run_id` 而不是 `conversation_id`？如果绑定到 `conversation_id`，同一个会话连续问两个问题会出现什么具体问题？
