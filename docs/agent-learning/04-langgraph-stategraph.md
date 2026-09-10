# LangGraph StateGraph 多 Agent 编排

> 面向：写过状态机/工作流代码，但没用过专门的 Agent 编排框架的后端工程师。
> 目标：搞清楚"多 Agent"到底是什么意思、State 和 Context 两个容器的边界在哪、路由函数怎么写。

## 1. 要解决的问题

如果只用一次 LLM 调用，让它"读案情、检索法条、写意见、自己检查"全部一次做完，会遇到几个实际问题：

- Prompt 越写越长、职责越堆越多，模型容易顾此失彼
- 没法对"检索"和"生成意见"分别做质量控制（比如检索失败了要不要继续生成？生成的意见有没有真的用到检索出来的证据？）
- 调用工具的时机、次数、失败后的行为都混在一个大 Prompt 里，没法单独设限、单独审计
- 出了问题很难定位是"理解案情错了"还是"检索错了"还是"写意见时编造了"

拆成多个各司其职的角色（Agent），每个角色输入输出都结构化、职责边界清晰，出问题时容易定位到具体哪一环，也方便对每一环单独加确定性校验兜底。这就是"多 Agent 编排"要解决的问题。

## 2. 行业内一般怎么做

- **手写状态机**：自己用 `if/elif` 或者一个 `while` 循环 + 状态枚举来控制"现在该跑哪一步、下一步跑什么"。灵活，但状态多了之后条件分支会迅速变得难以维护，持久化/恢复/可视化都要自己另外实现。
- **通用工作流引擎**（Airflow、Dagster、Temporal）：定义 DAG（有向无环图）描述任务依赖关系，引擎负责调度、重试、持久化。这类工具更偏"离线批处理/长周期业务流程"，不是为"一次 LLM 多轮交互"这种细粒度场景设计的，硬套上来会觉得笨重。
- **Agent 编排框架**（LangGraph、CrewAI、AutoGen 等）：专门为"LLM 节点 + 工具节点 + 条件路由"这种细粒度场景设计，内置 State 管理、Checkpoint 持久化（见 [02-langgraph-checkpoint.md](02-langgraph-checkpoint.md)）、和 LLM Provider/工具生态的集成。本项目用的是 **LangGraph 的 `StateGraph`**。

## 3. 核心机制原理

`StateGraph` 的核心抽象只有三样：

- **节点（node）**：一个函数，接收当前 State，返回一个"增量" dict，框架负责把它合并回 State
- **边（edge）**：节点执行完之后该去哪个节点，分**普通边**（无条件，跑完 A 一定去 B）和**条件边**（`add_conditional_edges`，跑完 A 后调用一个路由函数决定去哪）
- **共享 State**：一个在所有节点之间传递、可以被 Checkpoint 持久化的数据结构

编排的过程 = 声明有哪些节点、节点之间怎么连（普通边/条件边），然后调用 `compile()` 得到一个可执行对象。

## 4. 本项目具体实现（函数级）

### 4.1 两层"Agent"抽象，容易搞混

本项目代码注释里专门强调了这一点（[graph/orchestrator.py:1-22](../../backend/app/agent/graph/orchestrator.py#L1)）：

1. **外层 `StateGraph`**——本文档讲的这一层，负责三个业务角色之间的路由，它本身**不是**大模型，`compile()` 这一步也不会发起任何模型调用。
2. **`legal_researcher` 节点内部**又用了 LangChain 的 `create_agent`——这是另一个更小的"模型自主决定要不要调用工具"的子循环，属于 [05-tool-calling-mcp.md](05-tool-calling-mcp.md) 的内容，不要和外层 StateGraph 混为一谈。

（拆分后，六个节点的业务逻辑按阶段分别放进了 `graph/nodes/` 目录下的独立文件——`legal_researcher` 在 `nodes/research.py`、`review_gate` 在 `nodes/review.py` 等——通过多继承组合进 `orchestrator.py` 的 `LegalConsultationGraph`；`orchestrator.py` 本身只负责图拓扑组装和跨节点共享的基础设施，不含任何一个节点的业务逻辑。）

### 4.2 声明拓扑：`_compile()`

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

- **`StateGraph(LegalConsultationState, context_schema=AgentInvocationContext)`**：构造函数的第一个参数是这个图的 **State 类型**（一个 `TypedDict`），第二个是 **Context 类型**（见 4.3 节，两者的区别是本文档最重要的一个知识点）。
- **`graph.add_node(name, func)`**：注册一个节点，`name` 是给路由/边引用用的字符串标识，`func` 是真正执行的异步函数（签名统一是 `async def node(self, state, runtime) -> dict[str, Any]`）。
- **`graph.add_edge(a, b)`**：无条件边，`a` 跑完必定去 `b`。`START`/`END` 是 LangGraph 内置的两个特殊哨兵节点，标记图的入口和出口。
- **`graph.add_conditional_edges(name, router_func, mapping)`**：`name` 节点跑完后，调用 `router_func(state)` 拿到一个字符串，再用 `mapping` 这个字典把字符串翻译成真正要去的节点名——**路由函数只读 State、返回一个 key，不关心目标节点具体叫什么名字**，`mapping` 才是"key → 真实节点名"的映射，这样路由函数和图拓扑解耦，改拓扑不用改路由函数内部逻辑。
- **`graph.compile(...)`**：把上面声明的节点/边固化成一个 `CompiledStateGraph`，这一步只是结构组装，不发模型请求。

### 4.3 State vs Context：这两个容器该放什么

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

### 4.4 路由函数：把结构化产物翻译成"下一步去哪"

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

## 5. 对比：本项目 vs 行业常规方案

- 相比手写状态机：`StateGraph` 免费带来了 Checkpoint 持久化（见 [02-langgraph-checkpoint.md](02-langgraph-checkpoint.md)）、流式事件输出（`astream`）、和 State/Context 分离的最佳实践建议——不用自己从零实现这些基础设施，代价是要按框架的节点/边模型组织代码，不能完全自由地写控制流。
- 相比"一个大 Prompt 全包了"：拆分角色后，每一步的输入输出都是结构化对象（`CaseAnalysis`/`EvidencePacket`/`CounselDraft`/`ReviewResult`），可以针对每一步单独做 Pydantic 校验、单独做确定性安全检查（比如 `review_gate` 的规则判断），出问题时看是哪个节点、哪个字段不对，比"一大段自然语言里想办法猜哪里错了"好定位得多。
- 本项目的一个额外设计（框架本身不会替你想）：路由函数里的"最多一次"这类循环上限，是业务层面自己加的安全阀——`StateGraph` 本身不会阻止你写出一个会无限循环的图，防止死循环是设计路由函数时必须自己考虑的责任。

## 6. 本项目内部的关键设计取舍与易错点

- State 和 Context 的边界（见 4.3）——搞混了会导致"以为恢复后的计数还是对的，结果被重置了"这类隐蔽 bug。
- 路由函数只读 State、不调用外部服务——保证"从哪个节点该去哪个节点"这个决定本身是确定性的、可预测的，不会因为路由过程中又发一次模型请求而引入新的不确定性。
- 循环上限硬编码在路由函数里（`< 1`），不是框架层面的通用配置——改这个阈值需要直接改 `after_review` 的代码，没有做成 `.env` 可配置项，这是当前的一个具体设计取舍，不是遗漏。

## 7. 动手验证方式

1. 照着 4.2 节的 `_compile()` 内容，自己手画一张节点-边的流程图（或者用 Mermaid 语法画在一个 Markdown 文件里）。
2. 跑 [tests/test_agent_runtime.py](../../tests/test_agent_runtime.py) 里几个不同分支的用例（搜 `matched`/`no_match`/`tool_error` 相关的测试函数名），对照断言里期望的 `stages` 序列，验证自己画的流程图和真实路由是否一致。
3. 故意构造一个"Reviewer 一直要求修订"的场景（可以临时改小 `revision_count < 1` 的判断做实验，但注意别把改动提交），观察 `after_review` 是否真的会在达到上限后强制走向 `finish`，而不是无限循环。
