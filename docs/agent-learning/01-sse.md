# SSE（Server-Sent Events）流式通信机制

> 面向：知道 HTTP 基础但没深入接触过流式通信的后端工程师。
> 目标：看懂本项目为什么用 SSE、怎么手写实现的、和业界常见做法比有什么取舍，以及生产级流式网关通常还要补什么。

## 0. 前置知识

不需要提前读其他篇，本篇可以作为整个系列的入口。如果想理解 SSE 端点背后的 Agent 执行细节，可以在读完本篇后接 [07-agent-run-persistence.md](07-agent-run-persistence.md)（讲的是"断线重连"背后真正依赖的任务持久化机制）。

## 1. 要解决的问题

LLM 生成回答是渐进式的（一个 token 一个 token 吐出来），如果等全部生成完再一次性返回给前端，用户要盯着一个空白/loading 转好几秒到几十秒。我们希望"边生成边显示"。

这本质上是一个更通用的问题：**服务器要主动、持续地把数据推给客户端，而不是客户端一次次去问"好了没"**。常见候选方案：

- **轮询（Polling）**：客户端每隔 N 秒发一次请求问"有新数据吗"。简单，但延迟高（最坏情况要等一个轮询周期）、请求量大。
- **长轮询（Long Polling）**：客户端发请求后，服务器 hold 住连接直到有数据才返回，客户端拿到后立刻发起下一次。比轮询实时，但本质还是"一问一答"，每次数据都要重新建立请求上下文。
- **WebSocket**：全双工，客户端和服务器都能随时主动发消息，连接建立后是持久的双向通道。
- **SSE（Server-Sent Events）**：单向（服务器→客户端），基于普通 HTTP，连接建立后服务器可以持续往同一个连接里写数据，浏览器负责把这个流解析成一个个"事件"。

## 2. 核心机制原理

### 2.1 协议格式本身

SSE 的响应就是一个普通 HTTP 响应，只是 `Content-Type: text/event-stream`、连接不关闭、body 持续追加内容，按下面这种文本格式组织，每个事件以**空行**结尾：

```text
id: 42
event: token
data: {"text": "你好"}

event: heartbeat

```

- `event:` 决定这条消息的类型（客户端可以按类型注册不同回调）
- `data:` 是负载，可以出现多行，会被拼接
- `id:` 是这条事件的游标，浏览器原生 `EventSource` 断线重连时会自动带上 `Last-Event-ID` 请求头，告诉服务器"从这条之后继续发"
- 以 `:` 开头的行是注释，浏览器会忽略——常被用来发"心跳"，只是为了让连接不被中间代理判定为空闲而断开，不代表真实业务事件

### 2.2 行业里怎么在这几个方案之间选

- 如果需要**双向**实时通信（比如协同编辑、多人游戏、聊天室里客户端也要主动发消息打断服务器），业界通常上 **WebSocket**，比如 Slack、多人文档协作。
- 如果只需要**服务器→客户端单向**推送（进度条、通知、日志 tail、LLM 流式输出），SSE 是更轻的选择：不需要单独的协议升级（`Upgrade: websocket`），复用普通 HTTP/HTTPS，浏览器原生 `EventSource` API 自带断线重连。ChatGPT 网页版、大多数 LLM 应用的"打字机效果"底层都是 SSE 或者结构类似的 chunked HTTP 流。
- 更老派的方案是 **Chunked Transfer Encoding 裸流**（不遵循 SSE 的 `event:`/`data:` 格式，就是把 HTTP response body 分块吐出去，客户端自己解析）。SSE 可以理解为"在 chunked 流之上定义了一套标准的文本协议"，比裸流更规范、浏览器有原生支持。如果换成原生 `EventSource` 而不是本项目这套手写方案，会立刻失去两个能力：只能发 GET（不能带请求体）、不能自定义请求头（带不了鉴权信息）——这正是本项目选择手写实现的直接原因，见 §3.4。

本项目选 SSE 而不是 WebSocket，是因为咨询场景天然是单向的：客户端发一次问题，服务器流式吐回答；不需要在回答生成过程中客户端再插话。

## 3. 本项目具体实现（函数级）

本项目有**两个** SSE 端点，服务于不同层面，但复用同一套底层机制。

### 3.1 最基础的 SSE 帧格式化：`sse()` / `persisted_sse()`

[routes/helpers.py:9-17](../../backend/app/api/routes/helpers.py#L9)：

```python
def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

def persisted_sse(sequence: int, event: str, data) -> str:
    return (
        f"id: {sequence}\nevent: {event}\n"
        f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    )
```

这两个是纯字符串拼接函数，手工实现了上一节讲的 SSE 文本格式；`persisted_sse` 比 `sse` 多了 `id:` 字段，因为它的调用方是"可断线重放"的端点（见 §3.3）。**这里没有用任何 SSE 库**——协议本身足够简单，直接拼字符串比引入一个依赖更直接，这是"协议简单到不值得为它专门找库"的一个典型例子。

### 3.2 业务对话流：`POST /api/conversations/{id}/messages/stream`

[routes/chat.py:136-267](../../backend/app/api/routes/chat.py#L136) 里的 `events()` 是一个 **async generator**（`async def events(): ... yield ...`）：

```python
async def events():
    ...
    yield sse("message_start", {"request_id": ctx.request_id})
    ...
    async for item in agent.run(memory_context, history, payload.content):
        ...
        yield sse(item["event"], item["data"])
    ...
    yield sse("memory_status", {"status": "pending", "job_id": job.id})
```

`agent.run(...)` 本身也是一个 async generator（它内部驱动 LangGraph 的 `astream`，见 [04-langgraph-stategraph.md](04-langgraph-stategraph.md)），`events()` 只是把它吐出来的每个事件再包一层 `sse()` 格式化后继续往外 `yield`——**这是一层"业务事件流"套在"Graph 执行事件流"外面的适配层**。

[routes/helpers.py:20-46](../../backend/app/api/routes/helpers.py#L20) 的 `with_sse_heartbeat(source, interval_seconds)`：

```python
async def with_sse_heartbeat(source, interval_seconds: float):
    iterator = source.__aiter__()
    pending: asyncio.Task | None = None
    while True:
        if pending is None:
            pending = asyncio.create_task(anext(iterator))
        done, _ = await asyncio.wait({pending}, timeout=max(0.01, interval_seconds))
        if not done:
            yield ": heartbeat\n\n"
            continue
        item = pending.result()
        ...
```

- `source.__aiter__()`：拿到 `events()` 这个 async generator 的迭代器。
- `anext(iterator)`：Python 内置函数，等价于同步版的 `next()`，向一个异步迭代器要"下一个值"，返回的是一个 **awaitable**，所以要用 `asyncio.create_task` 包成一个可以被 `asyncio.wait` 等待、同时又能设超时的任务。
- `asyncio.wait({pending}, timeout=...)`：**关键点**——同时等"下一个业务事件到了"和"超时"两件事，谁先发生就先处理。如果 15 秒（`SSE_HEARTBEAT_SECONDS`，[config.py:117](../../backend/app/core/config.py#L117)）内 LLM 还没吐出下一个事件（比如模型正在"思考"），就先发一条心跳 `: heartbeat\n\n`（注意前面是冒号，浏览器/Nginx 都会当注释处理，不会被解析成业务事件），连接不会被反向代理当成"死连接"掐掉，然后继续等**同一个** `pending` 任务（没有重新创建，避免打断真正在等的那次 `anext`）。

最外层用 FastAPI 的 `StreamingResponse` 把这个 async generator 包成 HTTP 响应（[routes/chat.py:270-278](../../backend/app/api/routes/chat.py#L270)）：

```python
return StreamingResponse(
    with_sse_heartbeat(events(), settings.sse_heartbeat_seconds),
    media_type="text/event-stream",
    headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
)
```

- `StreamingResponse`：FastAPI/Starlette 提供的响应类型，接收一个（同步或异步）可迭代对象，逐个把产出的内容写进 HTTP body，不会等全部产出完才发送——这是"流式"在框架层面的落地方式。
- `X-Accel-Buffering: no`：专门给 Nginx 这类反向代理看的头，告诉它别把这个响应缓冲起来再一次性转发（不然流式效果会在代理这一层被打没）。

### 3.3 可断线重连的事件回放：`GET /api/agent-runs/{run_id}/events`

这是另一个 SSE 端点，专门解决"页面刷新/断网后怎么接回之前没看完的流"这个问题（[routes/agent_runs.py:78-127](../../backend/app/api/routes/agent_runs.py#L78)）：

```python
@router.get("/agent-runs/{run_id}/events")
async def agent_run_events(
    run_id: str,
    after_sequence: int = Query(0, ge=0),
    last_event_id: str | None = Header(None, alias="Last-Event-ID"),
    ...
):
    cursor = max(after_sequence, int(last_event_id or 0))
    ...
    async def source():
        nonlocal cursor
        while True:
            run, rows = await asyncio.to_thread(manager.events, ctx, run_id, cursor)
            for row in rows:
                cursor = row.sequence
                yield persisted_sse(row.sequence, row.event_type, json.loads(row.payload_json))
            if run.status in TERMINAL_STATUSES and cursor >= run.last_event_seq:
                return
            await manager.wait_for_events(get_settings().sse_heartbeat_seconds)
            if not rows:
                yield ": heartbeat\n\n"
```

`Header(None, alias="Last-Event-ID")` 是 FastAPI 依赖注入语法，声明这个函数参数要从 HTTP 请求头的 `Last-Event-ID` 字段取值——**这正是浏览器原生 `EventSource` 断线重连时自动携带的标准头**，本项目同时兼容它和自定义 query 参数 `after_sequence`，取两者较大值。

这个端点背后的事件不是实时生成的文本片段，而是**已经持久化到数据库**的 `AgentRunEvent` 记录（`manager.events(ctx, run_id, cursor)` 是查数据库），所以叫 `persisted_sse`——**它推的是"补发历史 + 继续等新的"，而不是纯粹的实时转发**，即使服务进程重启过，只要 `AgentRun` 还在跑，前端刷新页面重新连这个端点就能从 `cursor` 之后继续收到事件，不会丢。这个机制和 [07-agent-run-persistence.md](07-agent-run-persistence.md) 讲的 Agent 任务持久化是配套的——SSE 层的"断线重连"能做到，靠的是任务层已经把事件持久化并编号，SSE 本身的协议特性只解决"客户端知道从哪个 ID 之后继续要"，真正"能不能补上"取决于下游有没有存。

### 3.4 前端消费：手写 SSE 解析（不是原生 `EventSource`）

[frontend/src/sse.ts](../../frontend/src/sse.ts) 完整实现了一个 SSE 客户端解析器，没有用浏览器原生 `EventSource`：

```typescript
export async function consumeSse(response: Response, onEvent: (event: SseEvent) => void) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const blocks = buffer.split(/\r?\n\r?\n/);   // 按空行切出一个个完整事件块
    buffer = blocks.pop() ?? '';                  // 最后一块可能不完整，留到下次拼接
    for (const block of blocks) {
      const parsed = parseSseBlock(block);
      if (parsed) onEvent(parsed);
    }
    if (done) break;
  }
}
```

- `response.body.getReader()`：`fetch` 返回的 `Response.body` 是一个 `ReadableStream`（浏览器原生流式读取 API），`getReader()` 拿到一个可以反复 `.read()` 的读取器，每次 `.read()` 拿到一块**原始字节**（不保证是完整的一条 SSE 消息，可能被 TCP 分片切碎）。
- `TextDecoder`：把字节流解码成文本，`{ stream: !done }` 表示"这不是最后一块，可能有跨块被截断的多字节字符，先缓着不要报错"。
- `buffer.split(/\r?\n\r?\n/)`：按 SSE 协议的"空行分隔事件"规则切块；`blocks.pop()` 把最后一个（可能不完整，因为下一次 `.read()` 还没到）留在 `buffer` 里等下次拼接——**这是手写流式解析器的标准套路：维护一个缓冲区，只处理已经确认完整的部分**。

放弃原生 `EventSource` 是因为业务上有两个硬需求它满足不了：需要发送对话内容（POST body，`EventSource` 只能发 GET）、需要带 `X-User-ID` 自定义认证头（`EventSource` 不允许自定义 header）。所以只能自己用 `fetch` 实现"消费流式响应"，代价是要自己写缓冲区拼接、自己处理断线重连逻辑。

## 4. 设计取舍

**为什么选 SSE 而不是 WebSocket？** 咨询场景天然单向：客户端发一次问题，服务器流式吐回答，不需要在生成过程中客户端再插话。选 WebSocket 能获得双向能力，但这里用不上，反而多背负一层协议升级和连接管理的复杂度。

**为什么放弃原生 `EventSource`、自己用 `fetch` 手写解析？** 原生 `EventSource` 只能发 GET、不能带自定义 header，本项目的对话请求需要 POST body（问题内容）和自定义鉴权头（`X-User-ID`），二者都不满足。代价是要自己实现缓冲区拼接和断线重连（后者靠额外的 `agent-runs/{id}/events` 端点补上）。

**为什么心跳只在"业务事件"和"超时"两者竞速时才发，而不是无脑定时器？** `asyncio.wait` 那段代码保证心跳不会打断正常的事件顺序——如果心跳是一个独立定时器，就需要额外处理"心跳恰好和业务事件同时到达"的竞态，现在的写法用一次 `asyncio.wait` 就把两种情况统一处理了。

**为什么要拆两个 SSE 端点，而不是一个端点包办所有场景？** `messages/stream` 是"这一次请求的实时输出"，进程重启或页面彻底关闭就没了；`agent-runs/{id}/events` 是"读数据库里已持久化的事件"，可以在任意时间点重新连上继续看。把两者合成一个端点，会导致"实时性"和"可重放性"这两个不同的需求互相牵制——拆开之后，前端可以先建 Run（走持久化端点），再决定要不要同时订阅实时流。

## 5. 易错点

- **忘记配 `proxy_buffering off`**：一旦部署时前面加了 Nginx 又没有对应配置，`X-Accel-Buffering: no` 这个响应头会被反向代理忽略，流式效果在网络层"消失"（表现为卡住不动、最后一次性吐出），排查时容易先怀疑后端代码而不是代理配置。
- **把心跳当成真实业务事件处理**：心跳是以 `:` 开头的注释行，没有 `event:` 字段，如果客户端解析逻辑没有正确区分注释行和数据行，会在心跳到达时抛异常或产生多余的空事件。
- **误以为 SSE 断线等价于任务被取消**：浏览器断开只终止本次订阅生成器，不会中断后台正在执行的 Agent 任务——这个细节在 [07-agent-run-persistence.md](07-agent-run-persistence.md) §5 有专门讨论，混淆两者会导致"以为用户取消了但任务还在跑"的困惑。
- **两个 SSE 端点混用**：如果误把"持久化事件重放"端点当成"实时流"来用（或反过来），会发现事件到达的时机和预期不一致——理解两者分工不同（见 §3.3 结尾）是排查这类问题的前提。

## 6. 生产化差距与面试应对

单机场景下这套手写 SSE 实现已经覆盖了心跳保活、断线重连、代理兼容这几个核心问题，但离生产级流式网关还有几处差距：

| 维度 | 本项目现状 | 生产环境通常怎么做 | 面试怎么答 |
|---|---|---|---|
| 连接规模 | 单机 Uvicorn 进程直接持有所有 SSE 连接 | 大规模场景通常在负载均衡层做长连接亲和（sticky session）或引入专门的网关（如 Envoy 支持 gRPC/SSE 流式代理），避免单实例连接数成为瓶颈 | "当前是单机验证阶段，连接数没有达到需要专门网关的规模；扩展到多实例后，第一步要解决的是长连接的负载均衡亲和问题" |
| 断线检测 | 依赖 TCP 层面的自然断开和心跳超时，没有主动的连接健康检查 | 生产网关通常有更主动的连接健康探测和优雅关闭机制（如结合 `Connection: close` 逐步排空连接做灰度发布） | "当前没有做主动健康检查，部署时重启会直接断开所有连接靠前端重连；生产化应该有排空机制，避免发布时用户体验骤降" |
| 多实例事件通知 | 进程内 `asyncio.Condition` 通知新事件，只在本进程内有效 | 多实例部署需要跨进程的事件通知机制（Redis Pub/Sub、数据库 LISTEN/NOTIFY，或消息队列） | "当前的事件等待机制是进程内的，如果 Worker 和 SSE 服务分布在不同实例，需要换成跨进程的发布订阅机制才能让'新事件到达'的通知跨实例传播" |
| 协议标准化 | 手写 `event:`/`data:` 拼接，前端手写解析器 | 大型系统通常会把 SSE/流式协议封装成 SDK，或直接采用更成熟的实时通信中台（如 Ably、Pusher，或自建的统一推送服务） | "当前规模下手写实现更直接、依赖更少；如果未来同时要支持多种客户端（移动端、第三方集成），会考虑抽象成统一的流式 SDK" |

## 7. 动手验证方式

1. 打开浏览器开发者工具 Network 面板，发起一次对话，找到 `messages/stream` 这个请求，点开看 Response 标签——应该能看到内容随时间逐步增长，而不是等待后一次性出现。
2. 用 `curl -N` 手动打这个端点（`-N` 关闭 curl 自己的输出缓冲），直接在终端看原始 SSE 帧长什么样：
   ```bash
   curl -N -X POST http://127.0.0.1:8000/api/conversations/<id>/messages/stream \
     -H "Content-Type: application/json" -H "X-User-ID: <id>" \
     -d '{"content":"你好"}'
   ```
3. 找一次真实对话的 `run_id`，断网几秒再联网，观察前端是否能通过 `GET /api/agent-runs/{run_id}/events?after_sequence=N` 续上，而不是整段对话消失重来。

**自测题：**

- 如果把 `with_sse_heartbeat` 里的 `asyncio.wait({pending}, timeout=...)` 改成"每 15 秒无条件发一次心跳，不管有没有新事件"，会带来什么问题？（提示：想想心跳和业务事件的顺序会不会被打乱）
- 为什么 `agent-runs/{run_id}/events` 端点即使服务重启过也能补上事件，而 `messages/stream` 端点做不到？两者的关键差异是什么？
