# LawStation 简历项目说明（AI Agent 开发岗位）

> 使用方式：第 1 节可直接复制到中文简历；篇幅紧张时使用第 2 节。文中的实验数据均来自项目内冻结数据集和时间戳报告，不代表真实用户总体表现或律师评审的法律准确率。

## 1. 推荐简历成稿

### LawStation — 多 Agent 法律咨询与法规 RAG 系统

**个人项目｜AI Agent 开发**  
**技术栈：** Python、FastAPI、LangChain、LangGraph、MCP、DeepSeek、Ollama、FAISS、BM25、TEI、BGE Reranker、SQLite、SQLAlchemy、Alembic、React、TypeScript、LangSmith、Pytest

**项目简介：** 面向中国大陆法律咨询场景，设计并实现支持多用户并发、持久化任务恢复、分层记忆、法规证据约束和量化评测的三 Agent 系统；将案情分析、法规检索、法律意见生成与安全复核编排为可观测、可恢复的完整工作流。

**核心工作：**

- 基于 **LangGraph StateGraph** 编排 `CaseAnalyst → LegalResearch → LegalCounsel → Reviewer/Finalize` 链路，以 Pydantic Schema 约束 `CaseAnalysis`、`EvidencePacket`、`CounselDraft` 和 `ReviewResult`；仅允许 Research Agent 通过 LangChain `create_agent` 调用 MCP 工具，并以模型/工具/回流次数上限防止 Agent 无限循环。
- 设计 **AgentRun 业务任务层 + LangGraph AsyncSqliteSaver 执行层** 的双层持久化架构：使用独立 `thread_id` 在 super-step 边界保存 Graph State，通过任务租约、幂等消息写入和 sequence Event Log，实现页面刷新、SSE 断线重放及服务重启后的节点级续跑；明确区分 Checkpoint、业务消息和长期记忆的职责。
- 构建 **BM25 + Ollama Qwen Embedding + FAISS + RRF** 双路召回，法规库覆盖 55,348 条源记录、55,374 个 chunk；采用 `document_id + chunk_id` 证据追踪和 Finalize 确定性校验，保证最终引用只能映射到本轮 `EvidencePacket`，并将空检索建模为正常 `no_match`，阻止无依据法名、条号和引用进入回答。
- 在 300 条源法条约束的合成语义挑战集上完成 BM25/Hybrid 消融：**Recall@5 从 86.67% 提升至 96.33%，MRR 从 0.7699 提升至 0.8705，Hit@1 从 69.67% 提升至 80.33%**；同时记录平均检索延迟由 176 ms 增至 263 ms，量化效果与性能成本。
- 使用 TEI 本地部署 `BAAI/bge-reranker-v2-m3` 对 Hybrid Top 12 候选进行 Cross-Encoder 精排；在从 600 条候选按冻结规则筛选的 200 条合成排序挑战集上，将 **Hit@1 从 95% 提升至 97%、MRR 从 0.9725 提升至 0.9838**。由于平均总延迟由 262 ms 增至 1,569 ms，将精排设计为可配置、可降级至 RRF 的可选阶段，体现基于实验数据做架构取舍。
- 实现 **四层分级记忆**：原始消息、近期对话、结构化会话摘要和长期记忆；以 `user/conversation` 作用域隔离偏好与案件事实，使用独立非 Thinking JSON 模型异步提取记忆，并通过 `canonical_key + optimistic version + MemoryRevision` 原位覆盖冲突旧事实，避免跨案件污染且不阻塞主回答。
- 建立全局 6、单用户 2、单会话 1 的三级并发边界；将 SQLAlchemy Session 限制在短事务内，不跨越 LLM、MCP 或 SSE 生命周期，并在所有记忆、会话、消息和任务操作中使用 `tenant_id + user_id + conversation_id` 所有权条件，防止并发会话串联。
- 建立 **LangSmith 全链路追踪与 Agent 分层评测**：将三 Agent、LLM、MCP、BM25、Embedding、FAISS、RRF、Reviewer 和 Finalize 组织为嵌套 Trace；在 30 条合成分层样本上完成 30 个 Target Run 和 30 次结构化 LLM Judge，回答质量实验中的 chunk 引用归属、`no_match` 安全、最新事实边界和无依据主张控制均通过，Judge 相关性为 **4.50/5**、清晰度为 **4.75/5**。
- 采用 **Spec-Driven Development（SDD）** 管理 Agent 系统迭代：先在 Living Spec 中冻结状态语义、API/配置契约、安全边界和验收指标，再执行最小化代码修改、自动化回归、量化实验与架构文档回写，形成“Spec → 实现 → 测试 → Eval → Resume Evidence”的可追溯闭环。

## 2. 一页简历精简版

### LawStation — 多 Agent 法律咨询与法规 RAG 系统

**技术栈：** LangGraph、LangChain、MCP、DeepSeek、FastAPI、Ollama、FAISS、BM25、SQLite、LangSmith、React

- 使用 LangGraph 构建 Case Analyst、Legal Research、Legal Counsel 三 Agent 工作流，以结构化 State/EvidencePacket 传递案情和法规证据；Research 是唯一 MCP 工具调用者，Finalize 通过 chunk 级证据映射约束引用并处理正常 `no_match`。
- 构建 BM25 + Qwen Embedding + FAISS + RRF 混合检索，在 300 条源法条约束的合成语义挑战集上将 Recall@5 从 86.67% 提升至 96.33%、MRR 从 0.7699 提升至 0.8705。
- 采用 AgentRun/Event Log 与 LangGraph AsyncSqliteSaver 双层持久化，实现任务租约、节点级恢复、最终消息幂等和 SSE 断线重放；支持全局 6、单用户 2、单会话 1 的并发控制。
- 实现用户级/会话级分层记忆、增量摘要和最新事实原位替换；所有数据读写使用租户、用户、会话复合所有权条件，避免跨用户与跨案件污染。
- 建立 LangSmith 嵌套 Trace 与确定性 Evaluator + 专项 LLM Judge 评测体系；在 30 条合成 Agent 样本中完成 30 个 Target Run，回答质量安全前置项均通过，相关性 4.50/5、清晰度 4.75/5。
- 采用 SDD 固化状态机、接口、安全与验收标准，通过自动化测试、消融实验和时间戳报告驱动 Agent/RAG 迭代。

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

> LawStation 是我围绕法律咨询场景完成的多 Agent 项目。它不是简单套一层聊天界面，而是把案情分析、法规检索、法律意见生成和安全复核拆成 LangGraph 中的三个业务 Agent，并通过结构化 State 和 EvidencePacket 限制每个节点能看到和能输出的内容。Research Agent 只能通过 MCP 调用法规 RAG，Counsel 不能引用 EvidencePacket 之外的 chunk，Finalize 还会执行确定性引用和 no-match 校验。
>
> 在工程层面，我使用 AgentRun 和 LangGraph Checkpoint 做双层持久化：前者管理任务所有权、租约、取消和 SSE 重放，后者保存节点执行状态，因此页面刷新或服务重启后可以继续任务。RAG 使用 BM25、Ollama Embedding、FAISS 和 RRF；在 300 条合成语义挑战集上，Recall@5 从 86.67% 提升到 96.33%。我还对 BGE 精排做了消融，发现排序收益有限但延迟增加约 1.31 秒，因此将其保留为可配置阶段，而不是为了技术堆栈强制启用。
>
> 最后，我使用 LangSmith、确定性 Evaluator 和结构化 LLM Judge 建立质量回归，并用 SDD 将 Spec、代码、测试、评测和简历证据串成闭环。这个项目重点体现的是 Agent 编排、证据约束、持久化、评测和工程取舍，而不只是 Prompt 调用。

## 5. 面试追问展开点

### 为什么使用三 Agent，而不是一个大 Prompt？

- Analyst 将自然语言案情转为结构化争议点和事实覆盖。
- Research 独占工具权限，隔离外部证据获取和回答生成。
- Counsel 只基于案情与 EvidencePacket 形成草稿。
- Reviewer/Finalize 负责语义复核与确定性安全出口。
- 角色边界使工具轨迹、错误归因、调用预算和离线评测更清晰。

### 为什么 Agent 持久化不能只依赖 LangGraph Checkpoint？

- Checkpoint 保存 Graph State 和节点进度，但不负责用户所有权、排队、取消、租约、SSE 游标或最终 Message ID。
- `AgentRun/AgentRunEvent` 管业务任务和可重放输出；`AsyncSqliteSaver` 管 super-step 恢复。
- 两者结合后，Graph 节点允许至少一次执行，但用户消息、助手消息和最终事件保持幂等。

### 如何控制法律回答幻觉？

- MCP 返回候选后，服务端用真实 `chunk_id` 回填权威元数据，不信任模型自行生成的法名、条号和正文。
- Counsel Claim 只能声明本轮 EvidencePacket 中的 chunk。
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

