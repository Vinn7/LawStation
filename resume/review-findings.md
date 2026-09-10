# LawStation 全项目 Review 发现

## 1. 总体评价

LawStation 已不是简单的 LLM Chat Demo。当前代码形成了较完整的单机 Agent 工程：三 Agent 状态机、标准 MCP 边界、混合 RAG、分层记忆、多用户并发、审计、评测、前端后台会话和可恢复索引均有实际代码与自动测试。

它的优势是工程边界和失败语义清楚；主要短板集中在生产认证、真实流式体验、长任务持久化、单机并发边界和数据/评测质量，而不是“有没有 Agent”。

## 2. 已验证亮点

### 2.1 证据约束而非只靠 Prompt

`LegalResearchAgent` 只能选择工具真实返回的 chunk，服务端 `_authoritative_evidence` 重建 EvidenceItem，`_citation_errors` 和 `finalize` 再约束草稿与 citations。模型伪造 ID 不能进入最终引用。

关键 symbol：`backend/app/agent/graph/evidence.py::_authoritative_evidence`、`backend/app/agent/graph/nodes/finalize.py::finalize`。

### 2.2 no_match 具有正常业务语义

空检索继续进入 Counsel 和 Review，不重复搜索、不冒充工具故障，也不虚构法条。代码级安全模板是最后兜底。

关键 symbol：`EvidencePacket.retrieval_status`、`_no_match_violations`、`_no_match_safe_answer`。

### 2.3 MCP 工具发现按应用复用

首请求 single-flight 发现，后续缓存 Tool 定义；失效有 stale/failed 和冷却，不把 session、结果或用户状态放进缓存。

关键 symbol：`MCPToolRegistry.get_tools`、`refresh`、`invalidate`。

### 2.4 记忆安全由代码裁决

最新事实优先写入 Prompt；持久化替换时模型只能建议 memory ID，服务端重验所有权、作用域、active 状态和 version，并保存 revision。

关键 symbol：`MemoryTaskManager._persist_candidates`、`_replace_memory`。

### 2.5 并发与前端状态按会话隔离

服务端同会话 reservation，前端按 `userId:conversationId` 存 runtime 和 controller。页面切换不停止原任务，迟到事件由 request token 阻断。

关键 symbol：`AgentConcurrencyManager`、`frontend/src/App.tsx::isCurrentStream`。

### 2.6 RAG 建库具有恢复能力

数据 SHA、切分配置、Provider、模型 digest、维度和查询指令共同构成指纹；批次原子落盘，完整校验后才切换正式索引。

关键 symbol：`LawSearchEngine._fingerprint`、`_build`、`_read_valid_index`。

### 2.7 观测与业务失败解耦

JSONL 是本地审计，LangSmith 是可选观测；初始化、导出、反馈同步或 flush 失败都不阻塞咨询。评测保留显式上传确认和实际用量账本，但不再因月度累计值硬阻断；生产 Trace 仍独立限额。

关键 symbol：`LangSmithObservability`、`MonthlyResourceBudget`、`ReportRun`。

### 2.8 Skill 是受控编排能力，不是 Prompt 插件

此前运行时 Skill 会让 `case-intake` 在 Analyst 之后追加一次串行模型调用，并扩大后续节点 Prompt。该实验能力已从产品热路径移除；当前 Case Analyst 只做一次结构化调用，开发 Skill 继续保留在 `.agents/skills/`，但不进入线上 Agent。

### 2.9 场景观察具有显式能力探测和证据化断言

场景模式可通过`run.py --test-scenarios`按进程启用，并在外部模型启动前校验冻结数据。前端只有明确404时隐藏入口，其他异常提供顶部栏/侧栏诊断与重试。执行结果不再以“操作未抛错”作为通过条件：未知断言为`inconclusive`，并发、SSE重放和记忆隔离分别依赖Run重叠、sequence缺口及`source_message_id`证据。

## 3. 部分实现或语义边界

### 3.1 `MemorySnapshot` 已进入持久化任务主链路

`backend/app/services/memory_schemas.py::MemorySnapshot` 是冻结 dataclass，`MemoryService.snapshot()` 在 AgentRun 真正取得执行配额后构建快照。快照显式包含近期消息、会话摘要、选中记忆、当前用户消息和时间点；Graph 执行期间不会动态吸收其他会话后写入的记忆。旧的 `MemoryService.context()` 仅作为兼容入口保留。

### 3.2 线上评测配置多于实际调度

`LANGSMITH_ONLINE_EVAL_SAMPLE_RATE` 已进入 Settings，但没有独立线上 evaluator worker。生产 trace、反馈和离线评测已实现；“自动线上 LLM 评估”只能标记为部分实现。

### 3.3 用户确认不是当前默认记忆流

后端保留 confirm/reject，数据模型保留 pending 等状态；但 `MemoryTaskManager` 创建新候选时直接 `status=active`，前端只展示 active 和记忆修正/删除。

### 3.4 流式不是原始模型 token

状态和心跳实时，但正文等待 Graph 和 Reviewer 完成后才分片发送。这是“安全后流式呈现”，不能在简历中表述为“模型实时 token 全链路转发”。

### 3.5 场景控制状态仍是页面级

AgentRun和事件可跨刷新恢复，但`ScenarioSession`步骤游标、订阅epoch与预期/实际对照仍保存在浏览器内存。刷新后可从普通会话继续观察任务，不能自动恢复到场景面板原步骤。

## 4. 文档偏差

### 4.1 非默认端口不会自动调整 MCP URL

`run.py --port` 改变 Uvicorn，但 `Settings.mcp_law_server_url` 默认仍固定 8000。需要同步 `.env`，否则工具发现失败。

### 4.2 `tools/` 不属于正式 Agent 工具集

当前正式 MCP 只有 `search_laws` 和 `get_law_article`。`tools/case_tool.py` 等没有被生产模块导入，不应作为“已接入多工具能力”介绍。

### 4.3 三 Agent 不等于三个服务

三个角色在同一 `StateGraph`，共享一个 ChatOpenAI Provider 和同一进程资源。微服务化、多机容错均未实现。

## 5. 风险清单

### P0：生产使用前必须处理

1. **真实认证缺失**：任何客户端可选择已知演示用户 ID。涉及 `get_user_context` 和 `/api/users`。
2. **敏感数据边界**：LangSmith 默认 `capture_content=true` 时可上传真实咨询正文，生产必须经过法务、隐私与数据驻留评估。
3. **法规质量元数据不足**：当前法规记录缺少效力状态、生效/废止日期、机关和地域，可能召回失效或适用范围错误的法条。
4. **安全防护不足**：缺少限流、请求体上限、CSRF/真实 CORS 策略和外部访问 TLS 方案。

### P1：核心体验与一致性

1. **首正文延迟高**：正文等待完整 Graph 复核；可考虑安全的 draft buffer 或更精细快速路径。
2. **会话排序不准确**：新增消息未更新 `Conversation.updated_at`。
3. **记忆部分成功语义**：候选已提交但摘要失败时整个 Job 标失败；建议拆分阶段状态。
4. **Memory Worker 关闭与吞吐**：单 Worker、领取非多进程原子，关闭等待无硬超时。
5. **任务恢复仍是单机能力**：刷新后可通过 `active-run + Last-Event-ID` 重放，服务重启可从 LangGraph checkpoint 续跑；但 AgentRun Worker、租约和事件通知仍基于单机 SQLite，不支持多实例竞争领取和跨节点实时推送。
6. **反馈错误体验**：前端反馈失败没有独立可见提示。
7. **运行时 Skill 实验已经结束**：旧路由 fixture 与历史说明不再代表当前能力；后续评测应聚焦三 Agent 路由、检索证据边界、持久化恢复和端到端延迟。

### P2：性能与扩展

1. **索引合并可能阻塞事件循环**：批次 Embedding 是异步的，但 memmap 写入、FAISS 增量组装、manifest 和目录替换主要在事件循环线程执行。
2. **带 law_name 的 Dense 查询成本**：为保证过滤召回，当前先搜全量 FAISS 再过滤，候选法律较小时仍扫描整个向量索引。
3. **通用分词和精排环境差异**：Jieba 未加法律词典；BGE 已通过 TEI 批量精排，但 macOS Metal 与生产 GPU 的阈值和 p95 仍需分别标定。
4. **单机状态**：并发锁、Memory Worker、FAISS 和 SSE runtime 均不能跨进程共享。
5. **依赖可复现性**：前端使用 `latest`，Python 没有完全冻结 lock。

## 6. 简历表述边界

### 可以说

- 实现三 Agent LangGraph 编排和 MCP 法律检索。
- 使用 BM25 + Ollama Dense + FAISS + RRF 混合召回。
- 设计 chunk 级 EvidencePacket 与确定性引用边界。
- 实现多用户逻辑隔离、分层记忆、最新事实覆盖和并发会话。
- 使用 `AsyncSqliteSaver` 持久化 LangGraph super-step，并以 AgentRun 租约、事件日志和幂等消息写入实现刷新/断线/单机服务重启恢复。
- 建立 JSONL 审计、LangSmith 可选追踪和低资源评测。
- 使用仓库级开发 Skill 固化 SDD 变更闭环与评测真实性审核；产品运行时 Skill 经延迟分析后已从 Agent 热路径移除。
- 实现可恢复全量索引、单入口和单端口部署。

### 不能说

- 已有生产级认证、RBAC 或真正多租户 SaaS。
- 三个 Agent 是独立微服务。
- 已接入 TEI `BAAI/bge-reranker-v2-m3` Cross-Encoder，并建立不读取 Candidate 结果的相邻法条挑战集资格流水线；在200条冻结集完成同样本消融前，仍不能宣称真实基准优于 RRF。
- 已实现多机、多副本环境中的任务迁移或分布式恢复。
- 所有回答都有律师人工校验。
- 评测小样本分数代表生产准确率。
- 前端展示的是 DeepSeek 原始实时 token。

## 7. 建议继续阅读的 10 个文件

| 顺序 | 文件 | 原因 |
|---|---|---|
| 1 | `backend/app/api/routes/` | 看清用户上下文、SSE、短事务和主链路 |
| 2 | `backend/app/agent/graph/` | 三 Agent、状态路由和证据安全核心 |
| 3 | `mcp_servers/law_rag/engine.py` | 全量建库与在线混合检索 |
| 4 | `backend/app/services/memory_tasks/` | 异步记忆、最新事实覆盖和增量摘要 |
| 5 | `frontend/src/App.tsx` | 多用户后台流与客户端隔离 |
| 6 | `backend/app/main.py` | 应用级对象和生命周期边界 |
| 7 | `backend/app/agent/middleware.py` | 工具审计、超时和调用限制 |
| 8 | `backend/app/services/repositories.py` | 数据所有权安全边界 |
| 9 | `backend/app/observability/langsmith.py` | Trace、隐私、采样和 fail-open |
| 10 | `tests/test_agent_runtime.py` | 从测试理解关键行为与设计意图 |

运行时 Skill 实现文件已经删除；如需理解项目开发约束，应阅读 `.agents/skills/lawstation-spec-change/SKILL.md` 和 `.agents/skills/lawstation-eval-review/SKILL.md`，二者不进入产品运行时。

## 8. 建议面试讲解顺序

1. 先画单进程总体架构和三 Agent 主链路。
2. 用 chunk EvidencePacket 解释如何控制法条幻觉。
3. 用 no_match 解释业务状态与异常状态的区别。
4. 用 MemorySnapshot 语义和原位替换解释上下文工程。
5. 用 ConversationKey 和三层配额解释并发隔离。
6. 用 staging/checkpoint/指纹解释工程可靠性。
7. 用 LangSmith + 本地 evaluator 解释质量闭环。
8. 主动说明认证、数据质量和跨刷新恢复尚未生产化。
# 新增：多轮对话场景生成与观察

**已验证**：历史v1使用18类确定性蓝图冻结36条合成多轮场景；移除运行时Skill后，默认v2确定性保留12类、24条场景。动作和事件断言不由模型生成，并通过敏感信息、Prompt Injection及法规来源泄漏校验。

**已验证**：前端已增加配置开关控制的场景观察面板，可为Actor创建隔离会话并逐步执行真实AgentRun，覆盖取消、SSE sequence重放、消息/记忆检查与安全清理。后端仅加载项目内冻结白名单，outcome不暴露Graph State和Prompt。

**当前不足**：它仍是每次点击一步的人工观察器，不会自动连续跑完24条场景；步骤游标也不跨刷新恢复。简历可描述“可交互场景观察与预期/实际对照”，不能写“24条端到端测试全部通过”。
