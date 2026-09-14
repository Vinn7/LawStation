# Tool Calling / Function Calling 机制 + MCP 协议

> 面向：熟悉"调用外部 API"这个概念，但没接触过"让 LLM 自己决定调用哪个 API、传什么参数"的后端工程师。
> 目标：理解 Function Calling 的底层协议长什么样、MCP 解决了什么问题、本项目怎么用 `tool_choice="required"` 从接口层杜绝模型输出自由文本，以及生产级工具调用平台通常还要补什么。

## 0. 前置知识

建议先读 [04-langgraph-stategraph.md](04-langgraph-stategraph.md) §3.1，理解"外层 StateGraph"和"`legal_researcher` 节点内部的子循环"是两层不同的编排——本篇讲的正是那个内层子循环具体怎么工作。读完本篇后接 [06-rag-hybrid-retrieval.md](06-rag-hybrid-retrieval.md)，看模型调用的工具背后到底在做什么。

## 1. 要解决的问题

LLM 本身只会"生成文本"，不能直接执行代码、查数据库、调 API。但很多场景需要它**自主决定**要不要去查点什么、查什么——比如用户问一个法律问题，模型得先判断"这个问题需不需要查法条""该用什么关键词查"，而不是把这个决策权留给写死的规则。这就是 **Function Calling（工具调用）** 要解决的问题：让 LLM 能够"请求"执行一个函数，并且传对参数，但真正执行这个函数的还是你自己的代码，不是模型本身。

## 2. 核心机制原理

### 2.1 三个演进阶段

- **最早期**：靠 Prompt 里写"如果需要查资料，请输出 `SEARCH: <关键词>`"这种约定格式，你的代码用正则去匹配模型输出，解析出要不要调用、调用什么。脆弱，模型很容易不按格式输出。
- **原生 Function Calling / Tool Calling**（OpenAI 2023 年起、现在几乎所有主流模型 API 都支持）：调用模型 API 时额外传一份"工具列表"（每个工具的名字、描述、参数 JSON Schema），模型的响应里除了普通文本，还可以带一个结构化的 `tool_calls` 字段（工具名 + 参数），由**框架/你的代码**负责真正去执行，执行结果再作为一条 `ToolMessage` 喂回给模型，模型基于结果继续对话或再次调用工具——这是现在的行业标准做法，本项目走的就是这条路。
- **工具怎么"接入"给模型**：如果每接一个工具就要在应用代码里写死一份"工具描述 + 执行逻辑"，工具一多就很难维护、也没法跨项目复用。**MCP（Model Context Protocol）** 就是为了解决这个问题诞生的——它定义了一套标准协议，让"提供工具的一方"（MCP Server）和"使用工具的一方"（MCP Client / Agent）解耦：只要 Server 按协议暴露工具，任何遵循 MCP 协议的 Agent 框架都能发现并调用它，不需要为每个 Agent 框架单独写一套适配代码。可以类比 USB 之于外设——不用为每个设备单独开发驱动接口。

### 2.2 Function Calling 的通用数据流

不涉及具体框架，纯协议层面：

1. 你的代码把"工具定义列表"（name/description/参数 schema）和对话历史一起发给模型 API
2. 模型可能返回普通文本，也可能返回一个或多个 `tool_calls`（每个包含工具名、参数 JSON、一个调用 ID）
3. 你的代码识别到 `tool_calls`，真正去执行对应的函数/API
4. 把执行结果包装成 `ToolMessage`（带上第 2 步里的调用 ID，让模型知道这是哪次调用的结果），加进对话历史再发回给模型
5. 模型基于工具结果继续生成——可能是最终答案，也可能是又发起新的 `tool_calls`
6. 循环直到模型不再请求工具调用

### 2.3 MCP 的分层

MCP Server（暴露工具，通常是独立进程/服务，通过 HTTP 或 stdio 等传输方式通信）与 MCP Client（发现并调用这些工具）是协议两端，中间用标准化的消息格式通信，和"这个工具具体怎么实现""调用方用的是什么 Agent 框架"完全解耦。这一层解耦在行业里的价值是：同一个 MCP Server 可以被 Claude Desktop、任何自建 Agent、任何遵循协议的客户端复用，不需要为每一种客户端单独写一套集成代码——这和 Web 服务用 REST/OpenAPI 描述接口让不同客户端都能调用是同一个思路。

## 3. 本项目具体实现（函数级）

### 3.1 MCP Server 端：暴露工具

[mcp_servers/law_rag/server.py:65-83](../../mcp_servers/law_rag/server.py#L65)（当前源码；下面先讲这次核对文档时在这两个函数里实际发现并修复的一个真实 bug，再看修复后的样子）：

```python
@mcp.tool()
async def search_laws(query: str, top_k: int = 8, filters: dict | None = None) -> dict:
    """混合检索中国法律法规，返回候选 chunk、法名、条号和检索分数；法律问题、权利义务
    或法条核验时使用，是否采纳由 Research 再判断。

    工具只读，因此当前 LangGraph 至少一次恢复语义下允许节点重跑；未来副作用
    Tool 必须另行设计幂等键，不能沿用这一假设。
    """
    return await (await initialize_engine()).search(query, top_k, filters, envelope=True)


@mcp.tool()
async def get_law_article(law_name: str, article_number: str) -> dict:
    """按法律名称与条号精确查找一条法条：按规范化法名和条号进行只读精确查询，
    返回仍需进入证据边界校验。
    """
    engine = await initialize_engine()
    return await asyncio.to_thread(engine.get, law_name, article_number) or {"error": "未找到指定法条"}
```

- **`@mcp.tool()`**：MCP Python SDK 提供的装饰器，把一个普通异步函数注册成一个 MCP 工具。它会自动读取**函数签名**（参数名、类型标注、默认值）生成参数的 JSON Schema；工具描述默认来自 `description` 参数（这里两个工具都没传），退回读取 `fn.__doc__`——已安装的 `mcp` SDK 源码写得很直白：`func_doc = description or fn.__doc__ or ""`（`mcp/server/fastmcp/tools/base.py:66`），且**不经过 `inspect.cleandoc()` 清理缩进/空行，原样发给模型**。
- **这次核对文档时在这两个函数里找到并修复的一个真实 bug：改动前，两个函数体里其实各写了两条字符串字面量，但只有第一条会真正生效。** Python 的语言规则是：函数体里*紧跟 `def` 之后的第一条*字符串表达式才会被存进 `__doc__`；后面再出现的字符串字面量只是一条被求值后立刻丢弃的普通表达式语句（no-op，`ruff` 等 linter 通常会标它 "string statement has no effect"）。用一个最小例子可以直接验证这条规则：
  ```python
  def f():
      """AAA"""
      """BBB"""
  f.__doc__  # -> 'AAA'；'BBB' 不会被任何读取 __doc__ 的代码看到
  ```
  改动前的 `server.py` 里，`search_laws` **真正**发给模型的描述其实是那段偏工程笔记的文字（"返回候选 chunk、法名、条号和检索分数……工具只读……未来副作用 Tool 必须另行设计幂等键"）；紧跟着的"混合检索中国法律法规。法律问题、权利义务或法条核验时使用。"这句读起来更像"给模型看的场景提示"，实际是死代码，`@mcp.tool()` 从未读到过它。`get_law_article` 同理：模型看到的是"按规范化法名和条号进行只读精确查询，返回仍需进入证据边界校验。"，"按法律名称与条号精确查找一条法条。"这句从未生效。这不代表改动前真正生效的那段描述对模型完全没用——"只读、允许重跑""结果仍需经过证据边界校验"这类约束本身也能帮模型判断"能不能重复调用""结果能不能直接引用"；只是它和维护者大概率想让模型看到的"这个工具是干什么用的、什么场景该用"不是同一份文字。仓库里没有能确认这两条字符串各自成文时间的历史线索，找不到可考证的"谁在什么时候补写了新说明却忘了删旧的"这类归因，只能如实描述现象。**现在两个函数已经改成上面代码块里的样子**：把两条字符串合并成一条真正的 `__doc__`，"什么时候该用这个工具"和"只读/幂等假设"这两层信息都保留，且都会被模型看到（用 `python -c "from mcp_servers.law_rag.server import search_laws; print(repr(search_laws.__doc__))"` 可以直接验证）。
  - 这是一个值得记住的通用教训：**docstring 是否真的会被工具框架读取、读到的是哪一段，必须用 `fn.__doc__`（或对应框架实际使用的元数据读取方式）直接验证，不能只靠读代码"感觉像是给模型看的"来判断。**这条教训本身就是这次写文档核对代码时的真实产物，不是编出来的例子。
- 两个工具是**不同的检索方式**：`search_laws` 是语义/关键词排序检索（返回候选，可能不止一个），`get_law_article` 是按法名+条号的精确确定性查找（一次只找一条）——模型需要根据场景自己判断该用哪个。

### 3.2 Agent 端：工具发现、缓存与单飞刷新

[registry.py:42-96](../../backend/app/agent/registry.py#L42) 的 `MCPToolRegistry` 不只是"第一次调用时发现、后面用缓存"这么简单，而是一个带**五态状态机**（`uninitialized` / `loading` / `ready` / `stale` / `failed`，见 [registry.py:74](../../backend/app/agent/registry.py#L74) 起的字段初始化）加**单飞（single-flight）刷新**的注册表：

```python
class MCPToolRegistry:
    def __init__(self, ...):
        ...
        self._lock = asyncio.Lock()
        self._tools: tuple[BaseTool, ...] = ()
        self._status = "uninitialized"
        ...

    async def get_tools(self, audit_context=None) -> list[BaseTool]:
        context = audit_context or {}
        if self._status == "ready":
            return list(self._tools)                       # 命中缓存，完全不碰锁
        if self._tools and self._status == "stale" and not self._retry_due():
            return list(self._tools)                        # 有旧工具，冷却期内先用旧的
        if self._status == "failed" and not self._retry_due():
            return []                                        # 从未成功过，冷却期内直接返回空
        return await self.refresh("initial" if self._version == 0 else "stale", context)

    async def refresh(self, reason, audit_context=None, *, force=False):
        async with self._lock:                               # 单飞的关键：拿到锁后重新判断一次状态
            if not force:
                if self._status == "ready":
                    return list(self._tools)                  # 等锁期间别人已经刷新成功，直接复用
                if self._status in {"failed", "stale"} and not self._retry_due():
                    return list(self._tools)
            ...
            tools = await asyncio.wait_for(self.client.get_tools(), timeout=self.settings.mcp_tool_timeout_seconds)
            ...
```
（完整实现见 [registry.py:80-170](../../backend/app/agent/registry.py#L80)）

- **单飞用的是最朴素的写法，不是 `Future`/去重池**：`asyncio.Lock` + "拿到锁之后再检查一遍状态"（double-checked locking）。第一个进入 `refresh()` 的协程拿锁去真正发请求；其余并发协程会在 `async with self._lock` 处排队，等锁释放后重新走一遍 `if not force: ...` 判断——这时候状态已经被第一个协程改成了 `ready`，于是直接返回缓存，不会重复调用 `self.client.get_tools()`。这个行为被单测直接验证：[tests/test_agent_runtime.py:247-256](../../tests/test_agent_runtime.py#L247) 用 `asyncio.gather` 同时发起 20 个 `get_tools()`，断言 `client.get_tools.await_count == 1`。
- **五态状态机**：`uninitialized`（还没发现过）→ `loading`（首次发现进行中）→ `ready`（有一份可用工具）；刷新失败时进入 `stale`（有旧缓存，先凑合用）或 `failed`（从未成功过，只能返回空列表）。`stale`/`failed` 都有一个 `_retry_due()`（[registry.py:201-206](../../backend/app/agent/registry.py#L201)）冷却期（配置项 `mcp_tool_discovery_retry_seconds`，默认 30 秒），避免 MCP Server 故障期间每个请求都去重新冲击发现端点。
- **失败时优先保留旧缓存**：`refresh()` 的异常分支里只有"从未成功过"（`previous_tools` 为空）才会真正标记为 `failed`；只要曾经成功过一次，哪怕这次刷新失败也只标记为 `stale`，继续把上一份工具列表交出去（[registry.py:156-159](../../backend/app/agent/registry.py#L156)）——这是"故障时宁可用旧数据也不要完全瘫痪"的取舍，代价是 Agent 可能拿着一份已经过期的工具 Schema 去调用一个远端参数已经变了的工具。
- **`version` 字段驱动下游重建**：每次刷新成功 `self._version += 1`（[registry.py:139](../../backend/app/agent/registry.py#L139)），代码注释写明这个版本号会进入 `AgentRuntime` 的 Graph 缓存 Key，工具集合变化时能原子性地重建绑定新工具 Schema 的 `research_agent`，而不是让旧的 `create_agent` 实例继续持有过期的工具列表。
- **`invalidate()` 是外部主动失效的入口**（[registry.py:172-182](../../backend/app/agent/registry.py#L172)）：这个方法不在 `MCPToolRegistry` 内部被调用，而是被 §3.4 的 `ToolAuditMiddleware` 在捕获到"非业务错误"的未知异常时调用，把状态从 `ready` 降级为 `stale` 或 `failed`——这是 §3.2 和 §3.4 之间一处容易被忽略的跨文件联动，细节见 §3.4。

### 3.3 让模型自主决定调用：`create_agent`

[graph/orchestrator.py:93-108](../../backend/app/agent/graph/orchestrator.py#L93)：

```python
self.research_agent = create_agent(
    model=model,
    tools=tools,
    response_format=ToolStrategy(EvidencePacket, handle_errors=True),
    context_schema=AgentInvocationContext,
    middleware=[
        InvocationModelLimitMiddleware(settings),
        ModelCallLimitMiddleware(run_limit=settings.agent_max_model_calls, exit_behavior="end"),
        ToolAuditMiddleware(registry, settings),
    ],
    name="legal-research-agent",
) if tools else None
```

- **`create_agent(model, tools, ...)`**：LangChain 提供的高层封装，自动构造出"模型—工具"循环：模型返回 `tool_calls` → 框架执行工具 → 结果喂回模型 → 模型继续，直到模型不再请求工具为止（本质就是 §2.2 讲的那个通用数据流，框架帮你把这个循环写好了）。
- **`middleware=[...]`**：`create_agent` 支持在循环的关键节点（模型调用前后、工具调用前后）插入自定义拦截逻辑，本项目用了三个：
  - `InvocationModelLimitMiddleware`：每次模型调用前检查/累加"这次咨询总共调用了几次模型"（跨越外层三个 Agent，不止这个子循环）
  - `ModelCallLimitMiddleware`：LangChain 内置中间件，限制**这个子循环自己**最多调用几次模型，超限后 `exit_behavior="end"` 让循环体面结束（而不是抛异常）
  - `ToolAuditMiddleware`：见 §3.4
- **`name="legal-research-agent"`**：给这个子 Agent 起的名字，主要用于日志/追踪里区分它和外层三个业务 Agent。
- 没有工具时（MCP 发现失败，即 §3.2 的 Registry 从未成功过、`get_tools()` 返回空列表）整个 `research_agent` 保留为 `None`（就是 [orchestrator.py:108](../../backend/app/agent/graph/orchestrator.py#L108) 那个 `if tools else None`）；`legal_researcher` 节点在真正调用它之前先判空，直接返回 `EvidencePacket(retrieval_status="tool_unavailable", ...)`（[nodes/research.py:41-45](../../backend/app/agent/graph/nodes/research.py#L41)，注释写明"这里明确返回 tool_unavailable，而不是把能力故障误写成正常 no_match"），而不是硬编码一个假的 Agent 对象或让后续代码对 `None` 调方法而崩溃。

### 3.4 每次真实工具调用的统一包装

[middleware.py:127-262](../../backend/app/agent/middleware.py#L127) 的 `ToolAuditMiddleware.awrap_tool_call` 包裹每一次真实 Tool Call，核心结构（节选）：

```python
async def awrap_tool_call(self, request: ToolCallRequest, handler):
    ...
    if context.metrics.tool_call_count >= self.settings.agent_max_tool_calls:
        return ToolMessage(content="本轮工具调用次数已达到上限。", ..., status="error")
    context.metrics.tool_call_count += 1
    ...
    try:
        message = await asyncio.wait_for(handler(request), timeout=self.settings.mcp_tool_timeout_seconds)
        result = _message_text(message)
        if isinstance(message, ToolMessage) and message.status == "error":
            status = "error"
            business_error = True
    except TimeoutError:
        ...                                    # 见下方三分表：超时
    except Exception as exc:
        ...
        self.registry.invalidate(f"{type(exc).__name__}: {exc}", fields)   # 见下方三分表：未知异常
        ...
```

- **`request: ToolCallRequest`**：`create_agent` 在识别出模型想调用某个工具后，把这次调用打包成的对象（包含工具名、参数、这次 invocation 的 `runtime`）。
- **`handler`**：一个函数，调用它才会真正通过 MCP Adapter 走到 §3.1 的 `search_laws`/`get_law_article` 实现；中间件可以在调用它之前/之后插入自己的逻辑（限流、超时、审计），这是一种标准的**装饰器/责任链模式**在框架里的体现。
- 达到 `agent_max_tool_calls` 上限时（[middleware.py:145-155](../../backend/app/agent/middleware.py#L145)），**不是**粗暴地报错中断整个循环，而是构造一条正常的 `ToolMessage`（`status="error"`，内容是人类可读的"已达上限"）喂回给模型——让模型在它熟悉的"处理工具结果"流程里自然地继续（基于已有结果收尾），而不是被框架从外部强行打断。设计意图见本节末尾。

**三种真实工具调用失败情形，中间件区分对待（[middleware.py:170-218](../../backend/app/agent/middleware.py#L170)）：**

| 情形 | 触发条件 | 返回给模型的 `ToolMessage` | 审计事件 | 是否让 Registry 失效 |
|---|---|---|---|---|
| 超时 | `asyncio.wait_for` 到达 `mcp_tool_timeout_seconds` 抛 `TimeoutError` | "工具调用超时，请稍后重试。" | `tool.call.timeout`（WARNING） | 否 |
| 业务错误 | `handler(request)` 正常返回，但返回值是 `status=="error"` 的 `ToolMessage`（MCP Adapter 开了 `handle_tool_errors=True`，见 [registry.py:69](../../backend/app/agent/registry.py#L69)，已经把远端工具执行异常转换成了这种"正常返回但语义失败"的消息） | 原样透传远端返回的 `ToolMessage` | `tool.call.failed`，`error_type="MCPToolError"`（ERROR） | 否 |
| 未知异常 | `handler(request)` 本身抛出非 `TimeoutError` 的异常（连接失败、协议错误、返回体 Schema 不匹配等基础设施级问题） | "工具调用失败：{摘要}" | `tool.call.failed`，`error_type=<实际异常类名>`（ERROR） | **是**——调用 `self.registry.invalidate(...)`（[middleware.py:208](../../backend/app/agent/middleware.py#L208)） |

三者的关键区别不在"要不要把失败告诉模型"（三种情形都会生成一条 `status="error"` 的 `ToolMessage` 让模型带着这个结果继续），而在**是否怀疑问题出在"工具目录本身过期/不可达"**：超时和业务错误被认为只是"这一次调用"的问题（比如这次查询恰好没结果、或者恰好慢了），不代表整个工具集合有问题；只有捕获到未知异常（通常意味着连接层或协议层出了问题）才会调用 `registry.invalidate()`，把 §3.2 的状态机标记为 `stale`/`failed`，让**下一次** `get_tools()`（不是当前这次，当前这次已经在走失败分支了）触发重新发现，而不是继续拿一份可能已经不可达的工具列表用下去。这也解释了为什么审计事件里业务错误统一记成 `error_type="MCPToolError"`（屏蔽掉底层协议细节，因为这只是"这条查询没查到/参数不对"级别的正常业务失败），而未知异常保留真实异常类名（这类失败往往需要看类名才能排查是网络问题还是协议问题）。
- `asyncio.wait_for(..., timeout=...)`：给每次真实工具调用加超时保护，避免一次检索卡死整个 Agent 执行；这个超时和 `InvocationModelLimitMiddleware`/`ModelCallLimitMiddleware` 保护的"模型调用次数"是完全不同维度的限制，见 §5。

**审计记录怎么从 MCP 协议的原始返回结构里"抠"出可读摘要**：MCP 协议里工具结果经常是一个内容块列表（例如 `[{"type": "text", "text": "<JSON 字符串>"}]`），而不是直接就是一个 JSON 值；`result_metadata()`（[middleware.py:40-79](../../backend/app/agent/middleware.py#L40)）用一个递归的 `visit()` 函数处理这种嵌套——遇到字符串就尝试当 JSON 再解析一层，遇到 dict 就找 `document_id` 字段判断"这是不是一条法条候选"，遇到 list 就展开每个元素——直到把嵌套结构里所有带 `document_id` 的候选都拍平进 `documents` 列表（最多 20 条）。这个"剥洋葱"式解析被单元测试直接验证：[tests/test_agent_runtime.py:613-630](../../tests/test_agent_runtime.py#L613) 构造了一个两层嵌套（外层 `type: text` 包一层字符串，内层才是真正的候选 JSON）的样例，断言能正确解出 1 条 `document_id="law-1"` 的记录。这个函数只服务于审计记录（写入 `ToolCallRecord`/`RetrievalTrace`），不影响真正喂给模型的工具结果——模型看到的始终是 `handler(request)` 的原始返回，审计只是"事后再解析一遍摘要出来记日志"。

**关于"上限用返回正常 ToolMessage 而不是直接中断循环"这个设计取舍，历史上是否真的踩过坑**：仓库里能找到的唯一佐证是 [orchestrator.py:100-103](../../backend/app/agent/graph/orchestrator.py#L100) 的代码注释——"工具调用上限只由 ToolAuditMiddleware 强制执行（而不是叠加 LangChain 内置 ToolCallLimitMiddleware 在模型/工具循环外层强制打断），避免两套限流机制给出不一致信号"。这条注释说明的是一个**设计意图**（不叠加两套限流机制），没有找到对应的 commit 历史或代码注释能证明"早期真的用过内置中间件、线上观察到模型输出大段自然语言解释"这类具体历史事实——所以这里如实标注为设计取舍，而不是断言一段无法考证的踩坑经历。从已安装的 `langchain==1.3.15` 源码可以确认这个取舍在机制上是站得住脚的：`ToolCallLimitMiddleware(exit_behavior="end")` 的文档化行为是"立即结束执行，并注入一条框架合成的 AI 消息"（不经过正常的"模型处理 ToolMessage"路径），这和本项目现在的做法（喂一条正常 `ToolMessage`，让模型在熟悉的路径里收尾）确实是两种不同形状的信号，同时启用两套限流逻辑理论上有互相干扰的风险——但这仍然是"读源码推导出的合理性"，不等于"生产环境实测结果"。

### 3.5 强制结构化收尾：`ToolStrategy` 和 `tool_choice="required"`

这是本项目里对"Function Calling 机制"理解得比较深入的一处应用，值得重点看。

`response_format=ToolStrategy(EvidencePacket, handle_errors=True)` 做了什么（这是直接读 LangChain 源码验证过的行为，不是猜测）：

- LangChain 会把 `EvidencePacket` 这个 Pydantic 模型**也包装成一个"工具"**（工具名默认取 Pydantic 类名，即 `"EvidencePacket"`），追加进模型可以调用的工具列表里。
- **关键点（分两层看，都是直接读已安装包源码验证的行为，不是读文档猜的）**：
  1. **LangChain 框架层**（已安装 `langchain==1.3.15`）：只要绑定了 `ToolStrategy` 产生的"结构化输出工具"，`create_agent` 内部的模型绑定逻辑就会把 `tool_choice` 设成字符串 `"any"`——`langchain/agents/factory.py:1419`：`tool_choice = "any" if structured_output_tools else request.tool_choice`。`"any"` 是 LangChain 里**模型无关**的"强制至少调用一个工具"约定，不是 OpenAI 系 API 本身认识的字面量。
  2. **模型绑定层**（已安装 `langchain_openai==1.6.0`；本项目的 `model` 是指向 DeepSeek 的 `ChatOpenAI` 实例，见 [provider.py](../../backend/app/agent/provider.py)）：`ChatOpenAI.bind_tools()` 在真正组装请求体之前，把 `"any"` 翻译成 OpenAI 系 API 认识的字面量 `"required"`——`langchain_openai/chat_models/base.py:2464-2467`，源码注释原文是 `# 'any' is not natively supported by OpenAI API. We support 'any' since other models use this instead of 'required'.`
  最终真正打到 DeepSeek 接口上的确实是 `tool_choice="required"`，但这是"框架层模型无关值 → 具体 Provider 适配层翻译"两步之后才得到的结果，不是 `ToolStrategy` 或本项目代码直接写死的字符串。这意味着模型在这个循环里**每一轮都必须返回至少一个 `tool_calls`，不能只返回自由文本**——它要么继续调用真实检索工具（`search_laws`/`get_law_article`），要么调用这个 `EvidencePacket` 工具来"收尾汇报"；也意味着如果哪天把 `model` 换成一个不是 `ChatOpenAI` 子类、且没有实现"`any` → 该厂商专属值"翻译的 Provider 封装，同一份 `ToolStrategy(EvidencePacket)` 代码不一定还能拿到"强制工具调用"的效果——这是切换模型 Provider 时容易被忽略的一处适配层依赖。
- **`EvidencePacket` 这个"工具"具体长什么样**：LangChain 把 Pydantic Schema 包装成工具时，工具名默认取 `schema.__name__`（`langchain/agents/structured_output.py` 的 `_SchemaSpec.__init__`：`self.name = str(getattr(schema, "__name__", ...))`），也就是字面量 `"EvidencePacket"`；工具描述默认取这个 Pydantic 类自己的 docstring（同一处源码注释："若未提供，使用 schema 的 docstring"）。本项目这个类的 docstring 写得比较完整（可以对比 §3.1 里 MCP 工具 docstring 踩的坑）：
  ```python
  class EvidencePacket(BaseModel):
      """Research 向 Counsel/Reviewer 交付的唯一法规证据边界。

      candidate_status 描述召回候选，evidence_status 描述候选是否被采纳，最终
      retrieval_status 再区分 matched、正常 no_match 与工具不可用/异常。
      """
      retrieval_status: Literal["matched", "no_match", "tool_unavailable", "tool_error"] = "no_match"
      candidate_status: Literal["matched", "no_match"] = "no_match"
      evidence_status: Literal["accepted", "rejected", "unavailable", "error"] = "rejected"
      research_tasks: list[ResearchTask] = Field(default_factory=list)
      evidence_items: list[EvidenceItem] = Field(default_factory=list)
      accepted_chunk_ids: list[str] = Field(default_factory=list)
      rejected_candidates: list[CandidateRejection] = Field(default_factory=list)
      unresolved_issues: list[UnresolvedIssue] = Field(default_factory=list)
      conflicts: list[str] = Field(default_factory=list)
      research_summary: str = ""
  ```
  （完整定义见 [schemas.py:137-179](../../backend/app/agent/schemas.py#L137)）"结构化收尾"具体收尾出的就是这份对象：三个 `Literal` 状态字段（`retrieval_status`/`candidate_status`/`evidence_status`）分层描述检索结果的可信程度，`evidence_items` 是真正被采纳、可引用的法条 chunk，`unresolved_issues`/`conflicts` 让模型显式声明"哪些争议点没查到、哪些法条互相矛盾"而不是悄悄吞掉。类上的 `align_status_with_evidence`（[schemas.py:171-179](../../backend/app/agent/schemas.py#L171)，一个 `model_validator(mode="after")`）还会用代码强制这三个状态字段与 `evidence_items` 是否为空保持一致，防止模型在这些字段之间自己填出自相矛盾的组合（比如"有证据条目但 `retrieval_status` 却写 `no_match`"）。
- 模型一旦调用了 `EvidencePacket` 这个工具，参数会被自动用 Pydantic 校验，校验通过后写进 `state["structured_response"]`，本项目节点代码直接读这个字段拿到一个**已经校验过的真实对象**，不需要再自己写正则去从模型的文本回复里"抠"出 JSON（对比其它无工具节点用的 [`_invoke_json`](../../backend/app/agent/graph/orchestrator.py#L154)/[`_extract_json`](../../backend/app/agent/graph/evidence.py#L35)，那些是靠 Prompt 文字约束"只返回 JSON"，本质上是弱约束）。
- `handle_errors=True`：如果模型调用 `EvidencePacket` 工具时参数没通过 Pydantic 校验，LangChain 会自动生成一条报错 `ToolMessage` 让模型重试，而不是直接抛异常中断整个流程。

代码注释里明确写了这套机制要预防什么（[orchestrator.py:87-89](../../backend/app/agent/graph/orchestrator.py#L87)）："从接口层杜绝了模型在撞到工具调用上限后转而输出大段自然语言解释的情况"——如果只靠 Prompt 文字"拜托"模型输出 JSON（类似 §4 提到的、无工具节点在用的 `response_format={"type": "json_object"}` 那一档弱约束），撞到边界情况时模型确实有"先用自由文本解释、再附 JSON（甚至不附）"这条退路；`tool_choice="required"` 从接口层直接堵死了这条退路，模型**物理上没有"只说话不调用工具"这个选项**。这是代码注释里写明的设计意图和机制上的保证，不代表本项目确实经历过一段"早期用纯 Prompt 约束、线上观察到模型输出大段自然语言解释"的具体历史——仓库里没有能证明这段具体历史的材料，这里不做超出注释本身的历史归因。

### 3.6 内部同进程、但走协议边界

MCP Server 和主业务 API 是同一个 Uvicorn 进程（[main.py:129](../../backend/app/main.py#L129) 的 `app.mount("/mcp", mcp_app)`），但 Agent **仍然通过 HTTP 协议**（回环地址 `http://127.0.0.1:8000/mcp/`，见 `mcp_law_server_url` 配置）去调用它，而不是绕开协议直接在 Python 里调用 `search_laws()` 函数——这是刻意维持的架构边界：即使部署形态是同进程，也要求 Agent 侧完全按标准 MCP 协议交互，未来要把检索能力拆成独立服务，理论上不需要改 Agent 侧任何代码。

同进程部署还有一步容易被忽略、但确实存在的生命周期依赖：Streamable HTTP 这种带 Session 的 MCP 传输方式，要求 `mcp.session_manager.run()` 先进入运行状态，之后首次工具发现和每一次 Tool Call 才能建立短期 MCP Session——[main.py:96](../../backend/app/main.py#L96) 的 `async with mcp.session_manager.run():` 把这一步正确嵌套在了 FastAPI 的 `lifespan` 里（数据库初始化、Checkpointer、Worker 都要先就绪，再进入这个 `async with` 块）。这不是"随手 `app.mount()` 一个 ASGI 子应用就完事"，遗漏这一步会导致 MCP 请求在建立 Session 阶段就失败，而不是一个容易联想到"生命周期漏写了一行"的报错。

## 4. 设计取舍

**为什么全程走原生 Function Calling，而不是 Prompt 格式约定 + 正则解析？** 参数是模型 API 原生保证的结构化 `tool_calls`，不依赖模型"听话"按格式输出，可靠性高一个量级——早期"约定格式"路线的脆弱性在行业里已经是被验证过的教训。

**为什么只有 `legal_researcher` 节点用 `ToolStrategy` 强制结构化收尾，其余节点不用？** `ToolStrategy` 依赖工具调用机制本身（把结构化输出也包装成一个"工具"），只有绑了真实工具的 `create_agent` 节点能这么用；完全无工具的节点（`case_analyst`/`legal_counsel`/`reviewer`）用不了这个技巧，只能退而求其次用 `response_format={"type": "json_object"}` 这种"至少保证是合法 JSON、但不保证符合业务 Schema"的弱一档约束（详见 [03-memory-context-engineering.md](03-memory-context-engineering.md) §3.3 的对比）。这不是遗漏，是这个机制本身的适用边界。

**为什么工具调用上限用"返回正常 ToolMessage"而不是直接中断循环？** 见 §3.4——这是代码注释里写明的设计意图（不叠加 LangChain 内置的 `ToolCallLimitMiddleware`，只让 `ToolAuditMiddleware` 在达到 `agent_max_tool_calls` 时构造一条模型"认识的" `ToolMessage`），而不是一段可考证的线上事故复盘：仓库里没找到证明"早期真的用过内置中间件、线上观察到模型输出大段自然语言解释"这一具体历史的材料。从已安装的 LangChain 源码看，这个取舍在机制上是自洽的——内置 `ToolCallLimitMiddleware(exit_behavior="end")` 会跳过正常的"模型处理 ToolMessage"路径，直接注入一条框架合成的 AI 消息，这和现在"喂一条模型熟悉的 ToolMessage 让它自己收尾"确实是两种不同形状的信号。

**为什么 MCP Server 同进程部署却仍然走 HTTP 协议，而不是直接函数调用？** MCP 协议这一层带来的价值是**解耦**——工具的实现（法条检索引擎）和工具的使用方（Agent 编排代码）之间只通过标准协议交互，理论上可以独立部署、独立演进，代价是多了一层协议开销（哪怕是同进程回环调用，也要走一次 HTTP 请求/响应，比直接函数调用慢）。

## 5. 易错点

- **改工具的 docstring 时不当心**：docstring 会被发给模型，改工具描述文字要当成"在改 Prompt"一样谨慎对待，随手改一句话的措辞可能改变模型对"该不该调用这个工具"的判断，而不是单纯的代码注释调整。
- **以为函数体里"读起来像描述"的字符串字面量就是真正生效的 docstring**：Python 只认函数体第一条字符串语句为 `__doc__`；本项目 `server.py` 里 `search_laws`/`get_law_article` 两个工具在这次核对文档之前就各自写了两条字符串字面量（细节见 §3.1），第二条其实是从未被 `@mcp.tool()` 读取过的死代码，已经修复合并成一条。不确定当前真正生效的是哪一段文字时，直接在 Python 里 `print(repr(fn.__doc__))` 核实，不要凭代码读起来的感觉猜——这条经验不止对 MCP 工具成立，对任何"框架靠读 `__doc__` 生成给模型/给人看的描述"的场景都适用；也提醒改任何已有 docstring 的函数时，优先在原有字符串内编辑，而不是在它前面/后面追加一条新的字符串字面量。
- **以为其余节点也能直接抄 `legal_researcher` 的 `ToolStrategy` 写法**：理解 §4 里"只能用在绑了工具的节点"这个限制，能帮你判断"这段代码为什么没有直接抄 `legal_researcher` 的做法"，避免在无工具节点上尝试一个根本用不了的机制。
- **把工具调用超时和模型调用超时搞混**：`asyncio.wait_for(..., timeout=mcp_tool_timeout_seconds)` 保护的是单次真实工具调用（比如一次检索），和模型本身推理超时是两个独立的超时配置，排查"为什么这一步这么慢"时要先分清楚是哪一层卡住了。

## 6. 生产化差距与面试应对

这套工具调用机制在协议正确性和结构化约束上已经做得比较扎实，但离生产级多工具、多提供商的 Agent 平台还有几处差距：

| 维度 | 本项目现状 | 生产环境通常怎么做 | 面试怎么答 |
|---|---|---|---|
| 工具规模 | 单个 MCP Server，两个工具（`search_laws`/`get_law_article`） | 大规模 Agent 平台通常要管理几十上百个工具，需要工具目录、权限分组、按场景动态加载子集（避免工具列表过长影响模型选择准确率） | "当前工具数量少，全部暴露给模型没问题；工具规模上升后，需要考虑按场景动态筛选可用工具子集，而不是每次把全部工具塞进 Prompt" |
| 多 MCP Server 编排 | 单个 MCP Server，Agent 直连 | 生产系统常有多个 MCP Server（不同数据源/能力），需要一层 MCP Gateway 或路由层统一发现和调用多个 Server | "当前只有一个法规检索 Server；如果未来接入更多外部数据源，会需要一层网关做统一的工具发现和路由，而不是让 Agent 直连每一个 Server" |
| 工具调用可观测性 | JSONL 审计 + 可选 LangSmith Trace，覆盖单次调用的耗时和结果 | 生产级工具调用平台通常有专门的工具调用成功率、延迟分布、错误分类的持续监控面板 | "当前的可观测性够排查单次问题，还没有形成持续的工具健康度监控；这是从'能查'到'能主动发现异常趋势'的差距" |
| 权限与鉴权 | MCP Server 靠进程内 Bridge Token 证明调用来自本进程，不是完整的用户级鉴权 | 生产多租户场景通常需要工具调用带上用户/租户身份，支持按身份做细粒度权限控制 | "当前的 Token 只证明调用来自本进程，不是用户级鉴权；如果工具需要访问用户特定的数据，需要补充身份传播机制" |
| 故障隔离 | 单个工具超时/失败只影响当前这次调用，靠 `ToolAuditMiddleware` 统一处理 | 大规模场景通常会有工具级的熔断（circuit breaker）：某个工具持续失败时短期内不再尝试调用它，而不是每次都等超时 | "当前每次调用独立超时，没有工具级熔断；如果某个外部工具持续不可用，目前的行为是每次都等到超时才降级，生产化可以加一层熔断减少无谓等待" |
| 工具描述正确性校验 | 没有任何自动化机制确保 `fn.__doc__`（真正发给模型的那段文字）就是维护者以为在维护的那段文字——§3.1 就实测发现了 `server.py` 两个工具各自藏着一条从未生效的死代码 docstring，靠人工读代码完全没发现，是这次核对文档时才用 `fn.__doc__` 验证出来的 | 生产级工具平台通常会把工具描述纳入可测试、可 diff 的产物（比如工具 Schema/描述快照测试、或者把描述抽成独立配置文件走评审），而不是散落在函数体里的字符串字面量 | "当前工具数量少，靠人工审查勉强够用，但已经实际出现过'代码里两条字符串、只有一条生效'这种不直观的问题；工具规模上升后需要把工具描述纳入自动化校验，不能依赖'读代码时感觉对不对'" |

## 7. 动手验证方式

1. 单独跑 MCP 调试模式：`python -m mcp_servers.law_rag.server`，用 MCP Inspector（或任何支持 MCP 协议的客户端工具）连上去，手动发一次 `search_laws` 调用，观察请求/响应的原始结构。
2. 亲手验证 §3.1 现在的 docstring 是否已经完整：`python -c "from mcp_servers.law_rag.server import search_laws; print(repr(search_laws.__doc__))"`，确认"什么时候该用这个工具"和"只读/幂等假设"两层信息都在同一条 `__doc__` 里、都会被发给模型——不要只读这篇文档的结论，自己跑一遍确认。如果想亲眼重现修复前的 bug，可以在本地临时把某个函数体首行字符串后面再加一条字符串字面量，重新 `print(repr(fn.__doc__))`，观察后加的那条确实不会出现。
3. 对照 §3.4 的三种失败分类表格，去 [middleware.py:197-208](../../backend/app/agent/middleware.py#L197) 确认一个更进阶的问题：为什么只有"未知异常"分支会调用 `self.registry.invalidate(...)`，业务错误（MCP 远端正常返回但 `status="error"`）不会？如果把业务错误也无条件触发 `invalidate`，可能带来什么副作用（提示：结合 §3.2 的 `stale`/`failed` 状态机和 `_retry_due()` 冷却期想一想——业务错误往往只是"这次查询没结果"，不代表整个工具集合不可用）。
4. 故意让 `AGENT_MAX_TOOL_CALLS` 设得很小（比如 1），跑一个需要多次检索的复杂问题，观察模型收到"已达上限"的 `ToolMessage` 后是怎么收尾的（应该是直接调用 `EvidencePacket` 工具汇报已有结果，而不是输出大段解释文字）。

**自测题：**

- 如果去掉 `response_format=ToolStrategy(EvidencePacket, handle_errors=True)`，只保留 `tools=tools`，模型在完成检索后会怎么收尾？和现在的行为比，缺了哪个保证？
- MCP Server 的 docstring 会被发给模型这件事，如果被团队里不了解这个机制的人当成"随手写的注释"改动，可能造成什么后果？结合 §3.1 的真实案例——如果那个人往一个已经有 docstring 的函数体里*追加*一条新的字符串字面量（而不是替换原有的），会发生什么，而且大概率不会被 Code Review 发现？
- §3.5 提到 LangChain 内部实际设置的是模型无关的 `tool_choice="any"`，是 `ChatOpenAI.bind_tools()` 才把它翻译成 OpenAI 系 API 认识的 `"required"`。如果本项目把 `model` 换成一个没有做这层翻译的自定义 Provider 封装，`ToolStrategy(EvidencePacket)` 还能保证"模型每一轮都必须调用工具"吗？大概率会退化成什么行为？
