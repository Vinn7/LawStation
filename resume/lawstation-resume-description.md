# LawStation 简历项目说明（AI Agent 开发岗位）

> 使用方式：第 1 节可直接复制到中文简历；篇幅紧张时使用第 2 节。文中的实验数据均来自项目内冻结数据集和时间戳报告，不代表真实用户总体表现或律师评审的法律准确率。

## 1. 推荐简历成稿

### LawStation — 多 Agent 法律咨询与法规 RAG 系统

**个人项目｜AI Agent 开发**  
**技术栈：** Python、FastAPI、LangChain、LangGraph、MCP、DeepSeek、Ollama、FAISS、BM25、TEI、BGE Reranker、SQLite、SQLAlchemy、Alembic、React、TypeScript、LangSmith、Pytest

**项目简介：** 面向中国大陆法律咨询场景，设计并实现支持多用户并发、持久化任务恢复、分层记忆、法规证据约束和量化评测的三 Agent 系统；将案情分析、法规检索、法律意见生成与安全复核编排为可观测、可恢复的完整工作流。

**核心工作：**

- **LangGraph 多 Agent 编排：** 基于 StateGraph 拆分案情分析、法规检索、意见生成和复核流程，使用结构化状态传递结果，并通过 LangChain Middleware 统一控制模型调用、工具调用、超时和回流，解决长链路中职责混杂与失控循环问题。
- **Agent 长任务可靠执行：** 结合业务任务、LangGraph Checkpoint、任务租约、幂等写入和事件重放，将 Agent 执行与浏览器连接解耦，支持页面刷新、网络重连及服务重启后的任务续跑。
- **法律知识库混合检索与证据增强：**

  1. **检索结构：** 以 BM25 稀疏召回、Qwen Embedding + FAISS 稠密召回和 RRF 融合构建混合检索；将法规递归重叠切分为 55,374 个携带法名、条号和来源元数据的 Chunk，并实施片段级引用校验。
  2. **召回优化：** 开展组合消融实验，以 BM25-only 为基线，引入 Qwen Embedding + FAISS 稠密召回与 RRF 融合；在 300 条合成语义挑战集上，**Recall@5 从 86.67% 提升至 96.33%，MRR 从 0.7699 提升至 0.8705**。
  3. **BGE 精排实验：** 使用 TEI 部署 BGE Reranker，在 200 条合成排序挑战集上将 **Hit@1 从 95% 提升至 97%**；因平均延迟增加约 1.31 秒，将其设为可配置降级能力。
- **分层 Memory：** 设计原始消息、近期对话、会话摘要和长期记忆四层机制，按用户与案件范围隔离上下文，并异步完成结构化提取、增量摘要和冲突事实更新，解决长会话膨胀与跨案件事实污染，同时避免记忆整理阻塞主回答。
- **多用户并发与数据隔离：** 建立全局、用户和会话三级并发控制，并以短事务和服务端所有权校验约束数据访问，解决任务争抢和上下文串联问题；系统支持全局 6 个任务、单用户 2 个会话并发，同一会话保持单任务运行。
- **LangSmith 可观测与 Agent Eval：** 建立全链路 Trace、确定性 Evaluator 和结构化 LLM Judge，在 30 条合成分层样本上完成事实忠实度、Reviewer 有效性和回答质量评测：事实专项 Judge 四项均为 **5.00/5**，Reviewer 错误检出率为 **87.5%**、动作准确率为 **83.33%**，回答引用归属与无依据主张等安全检查均通过，相关性为 **4.50/5**、清晰度为 **4.75/5**，并据此定位 Reviewer 漏检与回答覆盖不足等优化方向。
- **Spec-Driven Development：** 通过 Living Spec 固化架构边界、接口契约、安全规则和验收指标，并以自动化测试、消融实验和文档回写验证变更，解决 Agent 项目迭代中实现与设计易偏离的问题，形成从需求到评测证据的可追溯闭环。

## 2. 一页简历精简版

### LawStation — 多 Agent 法律咨询与法规 RAG 系统

**技术栈：** LangGraph、LangChain、MCP、DeepSeek、FastAPI、Ollama、FAISS、BM25、SQLite、LangSmith、React

- **LangGraph 多 Agent：** 编排案情分析、法规检索、意见生成和复核流程，以结构化状态、工具权限和 Middleware 调用限制解决复杂链路职责混杂问题，形成可追踪、可复核的回答流程。
- **法律知识库混合检索：**

  1. **检索结构：** 以 BM25 稀疏召回、Qwen Embedding + FAISS 稠密召回和 RRF 融合构建混合检索；将法规切分为 55,374 个带法条元数据的 Chunk，并实施片段级引用校验。
  2. **召回优化：** 通过 BM25-only 与“Qwen Embedding + FAISS + RRF”混合检索的组合消融，在 300 条合成语义挑战集上将 Recall@5 从 86.67% 提升至 96.33%、MRR 从 0.7699 提升至 0.8705。
  3. **BGE 精排实验：** BGE 将 Hit@1 从 95% 提升至 97%，但增加约 1.31 秒平均延迟，因此保留为可配置降级能力。
- **Agent 可靠执行与并发：** 采用业务任务与 LangGraph Checkpoint 双层持久化，实现节点级续跑、消息幂等、SSE 重放以及全局 6、单用户 2、单会话 1 的并发控制。
- **分层 Memory：** 通过范围隔离、增量摘要和最新事实更新解决跨案件污染与上下文膨胀，使不同用户和案件上下文保持独立，后台整理不阻塞回答。
- **LangSmith Agent Eval：** 使用 Trace、确定性 Evaluator 和结构化 LLM Judge 对 30 条合成分层样本评估事实忠实度、Reviewer 有效性和回答质量；事实专项评分均为 **5.00/5**，Reviewer 检出率为 **87.5%**，回答安全检查均通过，相关性为 **4.50/5**、清晰度为 **4.75/5**。
- **SDD 工程闭环：** 固化架构、安全和验收指标，并以自动化测试、消融实验和报告归档解决 Agent 迭代缺少统一标准的问题，形成可复现的开发与评测流程。

## 3. 项目亮点关键词

适合填入招聘平台技能标签或 ATS 关键词：

```text
AI Agent / Multi-Agent / LangGraph / LangChain / StateGraph
Checkpoint / AsyncSqliteSaver / Agent Persistence / SSE Replay
MCP / Tool Calling / Structured Output / Pydantic
RAG / Hybrid Retrieval / BM25 / Embedding / FAISS / RRF
Cross-Encoder / BGE Reranker / Ollama / TEI
Evidence Grounding / Citation Grounding / no-match Safety
Long-term Memory / Incremental Summary / Multi-tenant Isolation
LangSmith / LLM-as-Judge / Deterministic Evaluator / Ablation Study
FastAPI / SQLite / SQLAlchemy / Alembic / React / TypeScript
Spec-Driven Development / Observability / Idempotency / Concurrency
```

## 4. 60 秒面试介绍

> LawStation 是我围绕法律咨询场景完成的多 Agent 项目。它不是简单套一层聊天界面，而是把案情分析、法规检索、法律意见生成和安全复核拆成 LangGraph 中的三个业务 Agent，并通过结构化状态和证据包限制各阶段的输入输出。Research Agent 只能通过 MCP 调用法规 RAG，法律意见只能引用本轮检索证据，最终出口还会执行确定性引用和空结果校验。
>
> 在工程层面，我使用 AgentRun 和 LangGraph Checkpoint 做双层持久化：前者管理任务所有权、租约、取消和 SSE 重放，后者保存节点执行状态，因此页面刷新或服务重启后可以继续任务。RAG 使用 BM25、Ollama Embedding、FAISS 和 RRF；在 300 条合成语义挑战集上，Recall@5 从 86.67% 提升到 96.33%。我还对 BGE 精排做了消融，发现排序收益有限但延迟增加约 1.31 秒，因此将其保留为可配置阶段，而不是为了技术堆栈强制启用。
>
> 最后，我使用 LangSmith、确定性 Evaluator 和结构化 LLM Judge 建立质量回归，并用 SDD 将 Spec、代码、测试、评测和简历证据串成闭环。这个项目重点体现的是 Agent 编排、证据约束、持久化、评测和工程取舍，而不只是 Prompt 调用。

## 5. 面试追问展开点

### 为什么使用三 Agent，而不是一个大 Prompt？

- Analyst 将自然语言案情转为结构化争议点和事实覆盖。
- Research 独占工具权限，隔离外部证据获取和回答生成。
- Counsel 只基于结构化案情与本轮证据形成草稿。
- Reviewer/Finalize 负责语义复核与确定性安全出口。
- 角色边界使工具轨迹、错误归因、调用预算和离线评测更清晰。

### 为什么 Agent 持久化不能只依赖 LangGraph Checkpoint？

- Checkpoint 保存 Graph State 和节点进度，但不负责用户所有权、排队、取消、租约、SSE 游标或最终 Message ID。
- `AgentRun/AgentRunEvent` 管业务任务和可重放输出；`AsyncSqliteSaver` 管 super-step 恢复。
- 两者结合后，Graph 节点允许至少一次执行，但用户消息、助手消息和最终事件保持幂等。

### 如何控制法律回答幻觉？

- MCP 返回候选后，服务端根据真实证据片段回填权威元数据，不信任模型自行生成的法名、条号和正文。
- 法律意见中的主张只能关联本轮已经检索并校验的证据片段。
- `no_match` 被视为成功空结果，要求低置信度、条件化分析和明确披露。
- Finalize 使用代码校验引用归属、旧事实边界和无法条表达；违规时输出安全模板。

### 如何评价 RAG，而不是只看几个成功案例？

- 固定 Dataset SHA256、随机种子、索引指纹和模型 revision。
- 分别评估 Recall@5、MRR、Hit@1、Gold 平均排名和 p50/p95 延迟。
- 使用相同样本进行 BM25/Hybrid、RRF/BGE 单变量消融。
- 保留失败和退化样本，同时报告效果收益与延迟代价。

### 为什么 BGE 精排没有直接成为必选链路？

- 200 条排序挑战集上 Hit@1 仅提升 2 个百分点、MRR 提升约 0.0113。
- 平均检索耗时增加约 1.31 秒。
- 因此保留 Provider、状态检查、超时、冷却和 RRF 降级能力，通过配置决定是否启用。
- 该决策体现以业务指标和延迟预算驱动架构，而不是为了展示模型而增加固定成本。

## 6. 数据口径与使用限制

简历可以使用：

- 300 条 Dense 合成语义挑战集的 Recall@5、MRR、Hit@1 和延迟。
- 200 条冻结排序挑战集的 BGE/RRF 排序与延迟结果。
- 30 条 Agent 分层实验的完整 Run/Judge 数，以及回答质量套件中有明确证据的指标。
- 项目真实存在的 LangGraph Checkpoint、AgentRun、MCP、记忆、并发和数据隔离设计。

简历中不要写：

- “真实用户准确率”“律师标注准确率”或“法律结论准确率”。
- “Reviewer 质量门禁全部通过”：当前错误草稿检出率为 87.5%，低于 90% 门禁。
- “BGE 大幅提升召回”：它主要改善排序，且延迟代价明显。
- “分布式任务队列”：当前是单进程 SQLite 持久化 Worker，不是 Celery/Redis 集群。
- “生产级认证”：当前 `X-User-ID` 是演示身份上下文，不是 JWT/RBAC。

## 7. 证据索引

- Agent/RAG 实测数字：`resume/eval-results.md`
- Agent 编排：`resume/agent.md`、`resume/three-agent-behavior.md`
- LangChain/LangGraph：`resume/langchain-langgraph.md`
- 任务持久化：`resume/agent-task-persistence.md`
- RAG：`resume/rag.md`
- Memory：`resume/memory.md`
- LangSmith：`resume/langsmith.md`
- 架构总览：`resume/architecture.md`
