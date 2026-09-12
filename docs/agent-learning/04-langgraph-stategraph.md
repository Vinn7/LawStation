# LangGraph StateGraph 多 Agent 编排

> 面向：写过状态机/工作流代码，但没用过专门的 Agent 编排框架的后端工程师。
> 目标：搞清楚"多 Agent"到底是什么意思、State 和 Context 两个容器的边界在哪、路由函数怎么写，以及这套单机编排离生产级多 Agent 平台还差什么。

## 0. 前置知识

不需要提前读其他篇，这是整个 Agent 编排系列的入口。读完本篇后，[05-tool-calling-mcp.md](05-tool-calling-mcp.md) 讲的是本篇 §3.1 提到的"内层子循环"，[02-langgraph-checkpoint.md](02-langgraph-checkpoint.md) 讲的是本篇 State 怎么被持久化。

## 1. 要解决的问题

如果只用一次 LLM 调用，让它"读案情、检索法条、写意见、自己检查"全部一次做完，会遇到几个实际问题：

- Prompt 越写越长、职责越堆越多，模型容易顾此失彼
- 没法对"检索"和"生成意见"分别做质量控制（比如检索失败了要不要继续生成？生成的意见有没有真的用到检索出来的证据？）
- 调用工具的时机、次数、失败后的行为都混在一个大 Prompt 里，没法单独设限、单独审计
- 出了问题很难定位是"理解案情错了"还是"检索错了"还是"写意见时编造了"

拆成多个各司其职的角色（Agent），每个角色输入输出都结构化、职责边界清晰，出问题时容易定位到具体哪一环，也方便对每一环单独加确定性校验兜底。这就是"多 Agent 编排"要解决的问题。

## 2. 核心机制原理

### 2.1 三条常见路线

- **手写状态机**：自己用 `if/elif` 或者一个 `while` 循环 + 状态枚举来控制"现在该跑哪一步、下一步跑什么"。灵活，但状态多了之后条件分支会迅速变得难以维护，持久化/恢复/可视化都要自己另外实现——如果本项目走这条路，`review_gate`→`reviewer`→`after_review` 这条带循环回退的分支逻辑得自己维护一套状态转移表，还要自己实现 Checkpoint 落盘。
- **通用工作流引擎**（Airflow、Dagster、Temporal）：定义 DAG（有向无环图）描述任务依赖关系，引擎负责调度、重试、持久化。这类工具更偏"离线批处理/长周期业务流程"，不是为"一次 LLM 多轮交互"这种细粒度场景设计的，硬套上来会觉得笨重——它们的最小调度单元通常是"一个任务"，而不是"一次模型调用返回后决定下一步去哪"这种毫秒到秒级的细粒度决策。
- **Agent 编排框架**（LangGraph、CrewAI、AutoGen 等）：专门为"LLM 节点 + 工具节点 + 条件路由"这种细粒度场景设计，内置 State 管理、Checkpoint 持久化（见 [02-langgraph-checkpoint.md](02-langgraph-checkpoint.md)）、和 LLM Provider/工具生态的集成。本项目用的是 **LangGraph 的 `StateGraph`**。CrewAI 更偏"角色扮演式"的高层抽象（定义 Agent 的角色、目标、工具，框架决定怎么协作），AutoGen 更偏"多个 Agent 互相对话"的模式；LangGraph 相对更底层、更接近"图 + 状态机"的原始形态，换来的是路由逻辑（比如本项目的循环上限控制）可以写得更精确，代价是要自己声明每一条边。

### 2.2 核心抽象：节点、边、共享 State

`StateGraph` 的核心抽象只有三样：

- **节点（node）**：一个函数，接收当前 State，返回一个"增量" dict，框架负责把它合并回 State
- **边（edge）**：节点执行完之后该去哪个节点，分**普通边**（无条件，跑完 A 一定去 B）和**条件边**（`add_conditional_edges`，跑完 A 后调用一个路由函数决定去哪）
- **共享 State**：一个在所有节点之间传递、可以被 Checkpoint 持久化的数据结构

编排的过程 = 声明有哪些节点、节点之间怎么连（普通边/条件边），然后调用 `compile()` 得到一个可执行对象。

## 3. 本项目具体实现（函数级）

### 3.1 两层"Agent"抽象，容易搞混

本项目代码注释里专门强调了这一点（[graph/orchestrator.py:1-22](../../backend/app/agent/graph/orchestrator.py#L1)）：

1. **外层 `StateGraph`**——本文档讲的这一层，负责三个业务角色之间的路由，它本身**不是**大模型，`compile()` 这一步也不会发起任何模型调用。
2. **`legal_researcher` 节点内部**又用了 LangChain 的 `create_agent`——这是另一个更小的"模型自主决定要不要调用工具"的子循环，属于 [05-tool-calling-mcp.md](05-tool-calling-mcp.md) 的内容，不要和外层 StateGraph 混为一谈。

（拆分后，六个节点的业务逻辑按阶段分别放进了 `graph/nodes/` 目录下的独立文件——`legal_researcher` 在 `nodes/research.py`、`review_gate` 在 `nodes/review.py` 等——通过多继承组合进 `orchestrator.py` 的 `LegalConsultationGraph`；`orchestrator.py` 本身只负责图拓扑组装和跨节点共享的基础设施，不含任何一个节点的业务逻辑。）

### 3.2 声明拓扑：`_compile()`

[graph/orchestrator.py:114-152](../../backend/app/agent/graph/orchestrator.py#L114)：

```python
graph = StateGraph(LegalConsultationState, context_schema=AgentInvocationContext)
graph.add_node("case_analyst", self.case_analyst)
graph.add_node("legal_researcher", self.legal_researcher)
graph.add_node("legal_counsel", self.legal_counsel)
graph.add_node("review_gate", self.review_gate)
graph.add_node("reviewer", self.reviewer)
graph.add_node("finalize", self.finalize)

graph.add_edge(START, "case_analyst")
graph.add_conditional_edges("case_analyst", self.after_analysis,
                             {"finish": "finalize", "research": "legal_researcher"})
graph.add_edge("legal_researcher", "legal_counsel")
graph.add_edge("legal_counsel", "review_gate")
graph.add_conditional_edges("review_gate", self.after_review_gate,
                             {"review": "reviewer", "finish": "finalize"})
graph.add_conditional_edges("reviewer", self.after_review,
                             {"finish": "finalize", "research": "legal_researcher", "revise": "legal_counsel"})
graph.add_edge("finalize", END)
```

- **`StateGraph(LegalConsultationState, context_schema=AgentInvocationContext)`**：构造函数的第一个参数是这个图的 **State 类型**（一个 `TypedDict`），第二个是 **Context 类型**（见 §3.3，两者的区别是本文档最重要的一个知识点）。
- **`graph.add_node(name, func)`**：注册一个节点，`name` 是给路由/边引用用的字符串标识，`func` 是真正执行的异步函数（签名统一是 `async def node(self, state, runtime) -> dict[str, Any]`）。
- **`graph.add_edge(a, b)`**：无条件边，`a` 跑完必定去 `b`。`START`/`END` 是 LangGraph 内置的两个特殊哨兵节点，标记图的入口和出口。
- **`graph.add_conditional_edges(name, router_func, mapping)`**：`name` 节点跑完后，调用 `router_func(state)` 拿到一个字符串，再用 `mapping` 这个字典把字符串翻译成真正要去的节点名——**路由函数只读 State、返回一个 key，不关心目标节点具体叫什么名字**，`mapping` 才是"key → 真实节点名"的映射，这样路由函数和图拓扑解耦，改拓扑不用改路由函数内部逻辑。
- **`graph.compile(...)`**：把上面声明的节点/边固化成一个 `CompiledStateGraph`，这一步只是结构组装，不发模型请求。

### 3.3 State vs Context：这两个容器该放什么

这是本项目里最容易被新手搞混的一对概念。

**`LegalConsultationState`**（[state.py:18-45](../../backend/app/agent/state.py#L18)，`TypedDict`）——**业务数据，会被 Checkpoint 持久化**：

```python
class LegalConsultationState(TypedDict):
    messages: list[BaseMessage]
    case_analysis: CaseAnalysis | None
    evidence_packet: EvidencePacket | None
    counsel_draft: CounselDraft | None
    review_result: ReviewResult | None
    retry_count: int
    revision_count: int
    final_answer: str
    citations: list[Citation]
    ...
```

只能放**普通消息、Pydantic 模型、基本类型/容器**——因为它要能被序列化写进 Checkpoint 数据库（参见 [02-langgraph-checkpoint.md](02-langgraph-checkpoint.md)）。

**`AgentInvocationContext`**（[state.py:67-88](../../backend/app/agent/state.py#L67)，普通 `dataclass`）——**请求级依赖，不会被持久化**：

```python
@dataclass
class AgentInvocationContext:
    identity: AgentInvocationIdentity
    metrics: AgentInvocationMetrics = field(default_factory=AgentInvocationMetrics)
    langsmith_trace_id: str | None = None
    trace_config: dict[str, Any] | None = None
    persist_tool_audit: bool = True
    ...
```

放的是**这一次 HTTP 请求/这一次 Graph 调用**才有意义的东西：谁发起的（`identity`）、这次调用累计了多少次模型/工具调用（`metrics`）、追踪配置。这些东西**不能**进 Checkpoint——它们要么不该跨请求复用（比如 `metrics` 本次统计），要么根本不是可序列化的业务数据。

判断一个字段该放 State 还是 Context 的简单标准：**"如果进程重启、从 Checkpoint 恢复，这个值还应该是原来的值吗？"**——是，放 State；"不是，它应该重新构造"，放 Context。

一个体现两者边界微妙之处的细节：`context.metrics.tool_call_count`（本次调用的实时计数，在 Context 里）在每个节点执行完后，会**同步写回** `state["tool_call_count"]`（[graph/nodes/](../../backend/app/agent/graph/nodes) 下各节点文件的 `return` 语句里都有这一行）——这是因为 Checkpoint 恢复后 Context 会被重新构造成初始值（计数器归零），但业务上需要"恢复后这个计数不能真的归零"，所以真正权威的历史累计值要落在能被持久化的 State 里，Context 里的 `metrics` 只是"当前进程内跑这一段时的实时统计"。

### 3.4 路由函数：把结构化产物翻译成"下一步去哪"

路由函数是纯函数，不调用模型/工具，只读 State。以 `review_gate` 的确定性快速路径为例（[graph/nodes/review.py:103-128](../../backend/app/agent/graph/nodes/review.py#L103)）：

```python
skip_reason = ""
if self.settings.agent_review_mode == "always-llm":
    skip_reason = "评测配置要求始终执行模型复核"
elif state["review_result"] is not None:
    skip_reason = "修订后的草稿必须再次复核"
elif not analysis or analysis.risk_level != "low":
    skip_reason = "中高风险问题必须进行模型复核"
elif not packet or packet.retrieval_status != "no_match":
    skip_reason = "存在法规依据或检索异常，必须进行模型复核"
elif not draft or draft.confidence != "low":
    skip_reason = "无法条回答的置信度边界未满足"
elif _no_match_violations(draft.answer):
    skip_reason = "无法条回答未通过确定性边界校验"
```

这不是路由函数本身（路由函数是紧接着的 `after_review_gate`），而是给路由函数准备判断依据的节点——它把"要不要走一次真正的 LLM Reviewer"这个决定，尽量用**确定性规则**（风险等级、检索状态、置信度、正则规则）来做，只有低风险 + 无证据 + 低置信度 + 通过安全检查的简单问题才允许跳过 LLM 复核，省掉一次模型调用的延迟，同时不放松安全边界——这是"能用代码判断的就不要交给模型判断"这条设计原则的具体体现。

再看真正的路由函数 `after_review`（[graph/orchestrator.py:197-223](../../backend/app/agent/graph/orchestrator.py#L197)），它把 `ReviewResult.next_action`（模型的建议）和两个循环计数器（`retry_count`/`revision_count`）结合起来决定走向：

```python
if not review or review.approved or review.next_action == "finalize":
    return "finish"
if packet and packet.retrieval_status in {"no_match", "tool_unavailable", "tool_error"}:
    return "revise" if review.next_action == "revise_draft" and state["revision_count"] < 1 else "finish"
if review.next_action == "research_again" and state["retry_count"] < 1:
    return "research"
if review.next_action == "revise_draft" and state["revision_count"] < 1:
    return "revise"
return "finish"
```

关键点：**模型的建议不是唯一决定因素，代码里的计数器上限（`< 1`，即最多一次）才是最终的强制约束**——即使模型一直建议"再检索一次"或"再改一次稿"，计数器用完了也会被路由函数强制收敛到 `finish`，这是防止 Graph 陷入无限循环的硬性兜底，不依赖模型"自觉停下来"。

## 4. 设计取舍

**为什么用 LangGraph 而不是自己手写状态机？** `StateGraph` 免费带来了 Checkpoint 持久化（见 [02-langgraph-checkpoint.md](02-langgraph-checkpoint.md)）、流式事件输出（`astream`）、和 State/Context 分离的最佳实践建议——不用自己从零实现这些基础设施，代价是要按框架的节点/边模型组织代码，不能完全自由地写控制流。

**为什么拆成三个业务角色，而不是一个大 Prompt？** 拆分角色后，每一步的输入输出都是结构化对象（`CaseAnalysis`/`EvidencePacket`/`CounselDraft`/`ReviewResult`），可以针对每一步单独做 Pydantic 校验、单独做确定性安全检查（比如 `review_gate` 的规则判断），出问题时看是哪个节点、哪个字段不对，比"一大段自然语言里想办法猜哪里错了"好定位得多。代价是多了几次模型调用的延迟和成本，需要用 [05-tool-calling-mcp.md](05-tool-calling-mcp.md) 讲的调用上限来控制。

**为什么循环上限硬编码在路由函数里，而不是做成可配置项？** 改这个阈值需要直接改 `after_review` 的代码，没有做成 `.env` 可配置项——这是当前的一个具体设计取舍：循环上限直接关联到安全边界（防止无限循环消耗资源），把它做成运行时可调的配置反而增加了"被误配置成一个危险值"的风险面，硬编码换来的是这个边界不会在部署配置里被意外改动。

## 5. 易错点

- **把 State 和 Context 的边界搞混**——具体场景：以为某个计数器"恢复后应该还是对的"，但如果不小心把它放进了 Context 而不是 State，恢复后会发现计数被悄悄重置成初始值，等于绕过了循环上限。判断标准见 §3.3 结尾那句话，记住它就能避免这类隐蔽 Bug。
- **让路由函数调用外部服务（模型/工具）**——具体场景：如果路由函数内部又发起一次模型调用来"帮忙判断该走哪条边"，会让"从哪个节点该去哪个节点"这个决定本身变得不确定、不可预测，而且这次调用不会被计入正常的模型调用计数里，容易造成审计和预算统计的偏差。
- **忘记给条件边的每一个可能返回值都配好 `mapping`**——如果路由函数返回了一个 `mapping` 里没有的 key，LangGraph 会在运行时报错而不是静默忽略，写新的路由分支时要同步检查调用方的 `add_conditional_edges` 映射表是否已经覆盖新增的返回值。

## 6. 生产化差距与面试应对

这套三 Agent 编排在单机场景下已经把路由、状态隔离、循环上限这几件事做扎实了，但离生产级多 Agent 平台还有几处差距：

| 维度 | 本项目现状 | 生产环境通常怎么做 | 面试怎么答 |
|---|---|---|---|
| 部署形态 | 单进程内的 `CompiledStateGraph`，随应用启动加载 | LangGraph 官方有 LangGraph Platform/Cloud，提供托管的 Graph 部署、版本管理、水平扩展 | "当前是嵌入应用进程里的编排，验证阶段足够；如果要独立扩展编排层，会评估拆成独立服务或托管平台" |
| 可视化与调试 | 靠日志和 LangSmith Trace 排查执行路径，没有专门的 Graph 可视化工具接入 | 生产环境常配合 LangGraph Studio 或类似的可视化调试工具，实时查看 Graph 执行到哪个节点、State 长什么样 | "当前排查依赖 Trace 和日志；引入可视化调试工具能让路由问题定位更直观，这是下一步可以补的工具链" |
| 节点级重试策略 | 循环上限统一硬编码在路由函数里，没有按节点区分的重试/超时策略 | 生产级编排通常允许按节点配置独立的重试次数、超时、降级策略 | "当前的循环控制是业务语义层面的（'最多修订一次'），不是节点级的通用重试机制；如果节点本身因为网络原因失败，目前依赖的是 Checkpoint 恢复而不是节点内重试" |
| 多 Graph 编排 | 单一顶层 Graph，没有嵌套子图 | 复杂系统有时会用子图封装可复用的子流程（比如"检索+校验"作为一个可在多个 Graph 里复用的子图） | "当前业务复杂度还不需要子图抽象；如果未来出现多个业务场景复用同一段编排逻辑，会考虑拆成子图，但要注意 `checkpoint_ns` 这类子图专属机制的正确用法（见 [02-langgraph-checkpoint.md](02-langgraph-checkpoint.md)）" |

## 7. 动手验证方式

1. 照着 §3.2 的 `_compile()` 内容，自己手画一张节点-边的流程图（或者用 Mermaid 语法画在一个 Markdown 文件里）。
2. 跑 [tests/test_agent_runtime.py](../../tests/test_agent_runtime.py) 里几个不同分支的用例（搜 `matched`/`no_match`/`tool_error` 相关的测试函数名），对照断言里期望的 `stages` 序列，验证自己画的流程图和真实路由是否一致。
3. 故意构造一个"Reviewer 一直要求修订"的场景（可以临时改小 `revision_count < 1` 的判断做实验，但注意别把改动提交），观察 `after_review` 是否真的会在达到上限后强制走向 `finish`，而不是无限循环。

**自测题：**

- 如果把 `retry_count`/`revision_count` 这两个计数器错放进了 `AgentInvocationContext` 而不是 `LegalConsultationState`，会发生什么？（提示：想想服务重启后从 Checkpoint 恢复这次执行会发生什么）
- `review_gate` 节点和 `after_review_gate` 路由函数是两个不同的东西，分别对应 `add_node` 和 `add_conditional_edges` 里的哪个参数？为什么不能把两者合并成一个函数？
