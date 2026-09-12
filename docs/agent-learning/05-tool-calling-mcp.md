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

[mcp_servers/law_rag/server.py:65-83](../../mcp_servers/law_rag/server.py#L65)：

```python
@mcp.tool()
async def search_laws(query: str, top_k: int = 8, filters: dict | None = None) -> dict:
    """混合检索中国法律法规。法律问题、权利义务或法条核验时使用。"""
    return await (await initialize_engine()).search(query, top_k, filters, envelope=True)

@mcp.tool()
async def get_law_article(law_name: str, article_number: str) -> dict:
    """按法律名称与条号精确查找一条法条。"""
    engine = await initialize_engine()
    return await asyncio.to_thread(engine.get, law_name, article_number) or {"error": "未找到指定法条"}
```

- **`@mcp.tool()`**：MCP Python SDK 提供的装饰器，把一个普通异步函数注册成一个 MCP 工具。它会自动读取**函数签名**（参数名、类型标注、默认值）生成参数的 JSON Schema，读取**函数的 docstring** 作为工具描述——**这份 docstring 不是给人看的注释，它会被发送给模型，直接影响模型判断"这个问题该不该调用这个工具"**，所以写法上要偏"这个工具是干什么用的、什么场景该用"，而不是实现细节。
- 两个工具是**不同的检索方式**：`search_laws` 是语义/关键词排序检索（返回候选，可能不止一个），`get_law_article` 是按法名+条号的精确确定性查找（一次只找一条）——模型需要根据场景自己判断该用哪个。

### 3.2 Agent 端：工具发现与缓存

[registry.py::MCPToolRegistry.get_tools()](../../backend/app/agent/registry.py)：应用启动后**首次**调用时才真正去 MCP Server 发现有哪些工具（避免每次请求都重新握手），后续复用缓存的 `BaseTool` 列表；失效/过期后用**单飞（single-flight）**模式刷新——多个并发请求同时触发刷新时，只有一个真正发请求，其余等这一次的结果，避免惊群效应。

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
- 没有工具时（MCP 发现失败）整个 `research_agent` 保留为 `None`，节点内部走"工具不可用"的降级分支，而不是硬编码一个假的 Agent 对象。

### 3.4 每次真实工具调用的统一包装

[middleware.py::ToolAuditMiddleware.awrap_tool_call](../../backend/app/agent/middleware.py)（节选逻辑）：

```python
async def awrap_tool_call(self, request: ToolCallRequest, handler):
    ...
    if context.metrics.tool_call_count >= self.settings.agent_max_tool_calls:
        return ToolMessage(content="本轮工具调用次数已达到上限。", ..., status="error")
    context.metrics.tool_call_count += 1
    ...
    message = await asyncio.wait_for(handler(request), timeout=self.settings.mcp_tool_timeout_seconds)
    ...
```

- **`request: ToolCallRequest`**：`create_agent` 在识别出模型想调用某个工具后，把这次调用打包成的对象（包含工具名、参数、这次 invocation 的 `runtime`）。
- **`handler`**：一个函数，调用它才会真正通过 MCP 走到 §3.1 的 `search_laws`/`get_law_article` 实现；中间件可以在调用它之前/之后插入自己的逻辑（限流、超时、审计），这是一种标准的**装饰器/责任链模式**在框架里的体现。
- 达到 `agent_max_tool_calls` 上限时，**不是**粗暴地报错中断整个循环，而是构造一条正常的 `ToolMessage`（`status="error"`，内容是人类可读的"已达上限"）喂回给模型——让模型在它熟悉的"处理工具结果"流程里自然地继续（基于已有结果收尾），而不是被框架从外部强行打断。这是本项目一处经过实测验证的设计取舍（早期用过 LangChain 内置的、更"强制"的限流中间件，实测会导致模型收到不一致信号、转而输出大段自然语言解释而不是正常收尾，后来改成了现在这种更"温和"的方式）。
- `asyncio.wait_for(..., timeout=...)`：给每次真实工具调用加超时保护，避免一次检索卡死整个 Agent 执行。

### 3.5 强制结构化收尾：`ToolStrategy` 和 `tool_choice="required"`

这是本项目里对"Function Calling 机制"理解得比较深入的一处应用，值得重点看。

`response_format=ToolStrategy(EvidencePacket, handle_errors=True)` 做了什么（这是直接读 LangChain 源码验证过的行为，不是猜测）：

- LangChain 会把 `EvidencePacket` 这个 Pydantic 模型**也包装成一个"工具"**（工具名默认取 Pydantic 类名，即 `"EvidencePacket"`），追加进模型可以调用的工具列表里。
- **关键点**：只要这个"结构化输出工具"存在，LangChain 在每一轮模型调用时都会显式设置 `tool_choice="required"`（对应 OpenAI 系 API 的标准参数）——这意味着模型在这个循环里**每一轮都必须返回至少一个 `tool_calls`，不能只返回自由文本**。它要么继续调用真实检索工具（`search_laws`/`get_law_article`），要么调用这个 `EvidencePacket` 工具来"收尾汇报"。
- 模型一旦调用了 `EvidencePacket` 这个工具，参数会被自动用 Pydantic 校验，校验通过后写进 `state["structured_response"]`，本项目节点代码直接读这个字段拿到一个**已经校验过的真实对象**，不需要再自己写正则去从模型的文本回复里"抠"出 JSON（对比其它无工具节点用的 [`_invoke_json`](../../backend/app/agent/graph/orchestrator.py#L154)/[`_extract_json`](../../backend/app/agent/graph/evidence.py#L35)，那些是靠 Prompt 文字约束"只返回 JSON"，本质上是弱约束）。
- `handle_errors=True`：如果模型调用 `EvidencePacket` 工具时参数没通过 Pydantic 校验，LangChain 会自动生成一条报错 `ToolMessage` 让模型重试，而不是直接抛异常中断整个流程。

这套机制解决了一个真实踩过的问题：早期实现里模型只是被 Prompt 文字"拜托"输出 JSON，撞到工具调用上限之类的边界情况时，模型有时会先用自然语言解释一大段"我已经检索了几次、接下来打算怎么办"，再附上 JSON（甚至有时候不附）——`tool_choice="required"` 从接口层面直接杜绝了这种可能性，模型**物理上没有"只说话不调用工具"这个选项**。

### 3.6 内部同进程、但走协议边界

MCP Server 和主业务 API 是同一个 Uvicorn 进程（[main.py](../../backend/app/main.py) 里 `app.mount("/mcp", mcp_app)`），但 Agent **仍然通过 HTTP 协议**（回环地址 `http://127.0.0.1:8000/mcp/`，见 `MCP_LAW_SERVER_URL` 配置）去调用它，而不是绕开协议直接在 Python 里调用 `search_laws()` 函数——这是刻意维持的架构边界：即使部署形态是同进程，也要求 Agent 侧完全按标准 MCP 协议交互，未来要把检索能力拆成独立服务，理论上不需要改 Agent 侧任何代码。

## 4. 设计取舍

**为什么全程走原生 Function Calling，而不是 Prompt 格式约定 + 正则解析？** 参数是模型 API 原生保证的结构化 `tool_calls`，不依赖模型"听话"按格式输出，可靠性高一个量级——早期"约定格式"路线的脆弱性在行业里已经是被验证过的教训。

**为什么只有 `legal_researcher` 节点用 `ToolStrategy` 强制结构化收尾，其余节点不用？** `ToolStrategy` 依赖工具调用机制本身（把结构化输出也包装成一个"工具"），只有绑了真实工具的 `create_agent` 节点能这么用；完全无工具的节点（`case_analyst`/`legal_counsel`/`reviewer`）用不了这个技巧，只能退而求其次用 `response_format={"type": "json_object"}` 这种"至少保证是合法 JSON、但不保证符合业务 Schema"的弱一档约束（详见 [03-memory-context-engineering.md](03-memory-context-engineering.md) §3.3 的对比）。这不是遗漏，是这个机制本身的适用边界。

**为什么工具调用上限用"返回正常 ToolMessage"而不是直接中断循环？** 见 §3.4——这是从真实的线上行为异常里反推出来的设计调整：给模型一个"它认识的"结构化信号，比在框架外部粗暴打断更可靠，模型能在熟悉的"处理工具结果"路径里自然收尾。

**为什么 MCP Server 同进程部署却仍然走 HTTP 协议，而不是直接函数调用？** MCP 协议这一层带来的价值是**解耦**——工具的实现（法条检索引擎）和工具的使用方（Agent 编排代码）之间只通过标准协议交互，理论上可以独立部署、独立演进，代价是多了一层协议开销（哪怕是同进程回环调用，也要走一次 HTTP 请求/响应，比直接函数调用慢）。

## 5. 易错点

- **改工具的 docstring 时不当心**：docstring 会被发给模型，改工具描述文字要当成"在改 Prompt"一样谨慎对待，随手改一句话的措辞可能改变模型对"该不该调用这个工具"的判断，而不是单纯的代码注释调整。
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

## 7. 动手验证方式

1. 单独跑 MCP 调试模式：`python -m mcp_servers.law_rag.server`，用 MCP Inspector（或任何支持 MCP 协议的客户端工具）连上去，手动发一次 `search_laws` 调用，观察请求/响应的原始结构。
2. 读一遍 [middleware.py](../../backend/app/agent/middleware.py) 完整的 `ToolAuditMiddleware.awrap_tool_call`，对照 §3.4，找出它是怎么区分"超时"、"业务返回的 error"和"未知异常"三种失败情况，分别记了什么审计事件。
3. 故意让 `AGENT_MAX_TOOL_CALLS` 设得很小（比如 1），跑一个需要多次检索的复杂问题，观察模型收到"已达上限"的 `ToolMessage` 后是怎么收尾的（应该是直接调用 `EvidencePacket` 工具汇报已有结果，而不是输出大段解释文字）。

**自测题：**

- 如果去掉 `response_format=ToolStrategy(EvidencePacket, handle_errors=True)`，只保留 `tools=tools`，模型在完成检索后会怎么收尾？和现在的行为比，缺了哪个保证？
- MCP Server 的 docstring 会被发给模型这件事，如果被团队里不了解这个机制的人当成"随手写的注释"改动，可能造成什么后果？
