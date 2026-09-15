# Agent / RAG 评测方法论

> 面向：知道"要给 AI 系统做评测"这个大方向，但没亲手设计过消融实验、没被 LLM-as-Judge 的偏差坑过的后端工程师。
> 目标：搞清楚"这个功能有没有用"这句话背后需要多严谨的实验设计才能回答，以及为什么"跑一次、看着还行"和"能写进简历的量化结论"之间隔着一整套方法论。

## 0. 前置知识

建议在读完 [06-rag-hybrid-retrieval.md](06-rag-hybrid-retrieval.md) 和 [04-langgraph-stategraph.md](04-langgraph-stategraph.md) 之后再看本篇——本篇讲的是"怎么证明前面两篇讲的检索优化和 Agent 编排真的有效"，需要先知道被评测的对象是什么。本篇是整个系列的收尾：前面几篇讲"怎么做"，这篇讲"怎么知道做得好不好"。

## 1. 要解决的问题

一个多 Agent + RAG 系统改了一个组件（比如加了 Reranker、换了 Reviewer 策略），怎么知道这个改动是真的变好了，而不是"改完之后随手试了三个问题、看起来还行"？这里面藏着几个真实的坑：

- **只看几个成功案例**：挑出来展示的往往是效果最好的样本，天然带幸存者偏差。
- **换了一个组件，同时也换了别的东西**：如果对比实验里数据集、随机种子、模型版本没有全部固定，看到的差异可能来自无关变量，不是被测试的那个改动。
- **拿小样本结果说大话**：3 条样本测出来"调用率下降了 66%"，这句话本身没错，但脱离样本量语境说出来，很容易被理解成生产环境的稳定收益。
- **只测最终答案，看不出问题出在哪一环**：一个多阶段系统（检索 → 编排 → 生成 → 复核）出问题，如果只测端到端结果，很难定位是哪一层的锅。

## 2. 核心机制原理

### 2.1 分层测试：把系统拆成独立可测的层

行业里对多组件系统的通用做法是分层隔离测试，而不是每次都测整条链路：

- 只测最底层的组件本身（这里是检索器），排除上层逻辑的干扰；
- 用固定/模拟的下游输入测中间层（这里是三 Agent 编排），排除底层组件的波动；
- 最后再测完整链路做最终确认，成本最高、频率最低。

这和普通软件工程里"单元测试 / 集成测试 / 端到端测试"金字塔是同一个思路在 AI 系统评测上的映射，只是每一层测的不是"函数返回值对不对"，而是"检索到的候选对不对""生成的回答符合安全规则吗"。

### 2.2 两类评测方法：确定性规则 vs LLM-as-Judge

AI 系统的输出往往没有唯一"标准答案"，但这不代表所有维度都只能靠模型打分：

- **确定性 Evaluator**：能用代码逻辑判断对错的维度（引用是否可追溯、是否越权访问了别人的数据、循环次数是否超限）应该用规则判断——结果可复现、可以无限扩展到大样本，没有模型本身的偏差。
- **LLM-as-Judge**：只有代码规则确实判断不了的语义维度（回答是否清晰、是否切题、语气是否得当）才交给另一个模型打分。行业里 RAGAS、DeepEval、TruLens、Langfuse 都是把这两类方法组合起来的评测框架。

举一个具体场景帮助建立直觉，而不是停留在抽象描述上：假设要评测"回答里引用的法条 chunk，是不是都来自这一轮真实检索到的证据包"——这句话读起来像是需要理解语义，但拆开看，它其实只是一次集合包含关系判断（`引用的 chunk ID 集合 ⊆ 证据包里的 chunk ID 集合`），一行代码就能给出确定性的对错，交给模型判断反而是多余的成本，还可能因为模型自己"觉得像是真的"而放过一次真实的编造。但换一个场景——"这段回答有没有把用户案情里的争议点都讲清楚、有没有遗漏关键风险提示"——两个字面完全不同的回答可能同样合格，也可能同样不合格，没有一条能写成代码的规则能覆盖"讲清楚"这个语义判断，只能交给语言模型去理解上下文再打分。这组对比背后的判断标准是：**能不能把"对不对"表达成一个不依赖语言理解、任何人复算都会得到同一个结果的谓词**——能，就该用确定性规则；不能，才值得花钱调用 Judge。

一个常见的方法论错误是"什么都交给 LLM 打分"——不仅成本高，还引入了下面这条要专门讲的偏差问题。

### 2.3 RAG 评测的四个经典维度：RAGAS 框架

开源框架 RAGAS 把 RAG 质量拆成四个可以独立衡量的维度，是这个领域里被引用最多的分解方式：

- **Faithfulness（忠实度）**：回答里的每个论断能不能由检索到的上下文支撑，衡量"有没有编造"。
- **Answer Relevancy（回答相关性）**：回答是否切题，衡量"有没有答非所问"。
- **Context Precision（上下文精确率）**：检索到的上下文里有多少是真正有用的，排在前面的是否更相关。
- **Context Recall（上下文召回率）**：标准答案所需的信息是否都出现在检索结果里。

这四个维度的价值在于**把"回答好不好"这个笼统问题拆成了检索质量和生成质量两个独立的、可以分别优化的子问题**——如果 Context Recall 低，问题在检索环节；如果 Context Recall 高但 Faithfulness 低，问题在生成环节没有忠实使用检索到的证据。

### 2.4 LLM-as-Judge 的已知偏差

用一个模型评价另一个模型的输出，天然带偏差，行业里总结出几类最常见的：

| 偏差 | 表现 | 常见缓解 |
|---|---|---|
| 位置偏差（Position Bias） | 成对比较时，Judge 倾向偏爱排在前面（或后面）的候选 | 交换两个候选的顺序各跑一次，取一致的结果 |
| 冗长偏差（Verbosity Bias） | 更长的回答容易被打更高分，即使信息量没有增加 | 打分标准里明确惩罚无关展开，或做长度归一化 |
| 自我偏好偏差（Self-preference Bias） | Judge 模型倾向偏爱和自己风格/训练分布接近的回答 | Judge 与被测模型使用不同家族/厂商的模型 |

这些偏差不是"用了 LLM Judge 就一定会犯"，而是设计评测时必须主动防范的已知风险点。

### 2.5 消融实验（Ablation Study）的基本纪律

消融实验的核心是**单变量控制**：固定其余全部条件（数据集、随机种子、模型版本、其他配置），只切换被测试的那一个变量，对比前后差异。行业里做严谨消融还会额外做**盲选/预注册**——决定"哪些样本进入最终对比"的规则,不能读取被比较双方的实际表现，否则等于在挑对自己有利的数据,这也是学术界"p-hacking"问题的工程对应版本。

## 3. 本项目具体实现（函数级）

> 本节所有代码引用都对照当前工作区源码逐行核实过（包括几个 §2 版本文档里只被文件级带过、这次重新通读全文才发现的关键机制）；行号会随代码演进漂移，读到行号对不上时以源文件为准。

### 3.1 三层检测对象，以及层内可变的"图起点"

LawStation 把"测什么"拆成三层，分别隔离不同的干扰源：

```text
retrieval  只测检索器（BM25/Hybrid/RRF/BGE），不经过 Agent
component  用固定 Fixture 工具结果测三 Agent 编排，屏蔽真实检索波动
live       真实 MCP + RAG + 完整 API 链路，成本高、用于最终确认
```

但通读 [backend/app/evaluation/targets.py](../../backend/app/evaluation/targets.py) 全文（382 行）后发现，实际落地比这张三层表格更细——真正暴露给评测脚本使用的是六个 Target 构造器，其中四个在用，两个是没有任何调用方的历史代码：

| 构造器 | 状态 | 图的起点 | 干扰源隔离方式 |
|---|---|---|---|
| `component_target`（116-121行） | **未被任何脚本或测试引用**，全仓 grep 确认是死代码 | 完整三 Agent 图 | Fixture 工具 |
| `live_target`（124-126行） | 同上，未被引用 | 完整三 Agent 图 | 真实 MCP |
| `agent_target(mode, review_mode)`（129-151行） | 使用中，`run_langsmith_eval.py`/`run_resume_rag_challenge_eval.py` 的 `--mode component/live` 走这里 | 完整三 Agent 图（Case Analyst → Researcher → Counsel → Reviewer → Finalize） | `mode="component"` 时用 `FixtureToolRegistry`，`mode="live"` 时用真实 `MCPToolRegistry` |
| `reviewer_target()`（154-236行） | 使用中，`run_agent_quality_eval.py` 的 `reviewer-effectiveness` 套件专用 | **跳过 Case Analyst 和 Researcher**，直接从冻结的 `CaseAnalysis`/`EvidencePacket`/`CounselDraft` 开始，只跑 `review_gate → reviewer →（按需）legal_counsel → reviewer → finalize` | 三个上游产物全部是测试固定的 Fixture，Reviewer 是唯一在评测中真正被调用的 Agent |
| `counsel_quality_target()`（239-313行） | 使用中，`factual-fidelity`/`answer-quality` 套件专用 | **跳过 Case Analyst 和 Researcher**，从冻结的 `CaseAnalysis`/`EvidencePacket` 开始，跑 `legal_counsel → review_gate →（按需）reviewer → finalize` | 同上，Counsel（以及可能触发的 Reviewer）是唯一真正执行的部分 |
| `RetrievalTarget`（316-382行） | 使用中，专供 RAG 消融 | 不经过图，直接调用 `LawSearchEngine.search` | 无 Agent，纯检索 |

也就是说，"三层"描述的是**工具/证据来源**这一个维度（fixture / 真实 MCP / 无 Agent），但 `component` 层内部还有第二个可变维度——**图从哪个节点开始跑**：`agent_target` 从头跑完整三 Agent 链路；`reviewer_target`/`counsel_quality_target` 把上游产物直接冻结成输入，只让被测的那一个 Agent 在**生产代码路径**（不是 Mock）上真正执行。这是本项目做"单 Agent 消融"的关键设计——想单独测 Reviewer 的检出能力，不需要 Mock 掉 Reviewer 本身，只需要冻结它的上游输入。

`FixtureToolRegistry`（22-65行）是 `component` 层的核心：它实现了和生产 `MCPToolRegistry` 相同的接口（`search_laws`/`get_law_article`/`get_tools`/`status`/`invalidate`/`close`），但 `search_laws` 直接对 `self.documents[:top_k]` 切片返回，不发起任何真实检索；构造时传入 `fail=True` 还能模拟工具调用异常（对应 `tool-error-degradation` 这类场景）。三 Agent 编排逻辑（路由判断、要不要调用工具、Reviewer 要不要介入）因此可以在完全确定性的输入下反复跑，不受检索质量波动干扰——这正是"component 层排除底层组件波动"这句话的具体所指。

公共的 `_invoke`（77-113行）把 Settings 拷贝一份、关闭 `langsmith_enabled`（避免评测内部的 Agent 调用又产生一层不受控的 Trace）、构造 `AgentRuntime`，用 `context.evaluation_output` 收集结果——这是所有 `component`/`live` Target 共用的执行外壳，保证不同评测套件看到的输出结构一致。

`RetrievalTarget.create`（323-335行）还带了两条"拒绝静默降级"的硬校验：要求 `hybrid` 模式时 Dense 索引必须已启用，要求开启精排时 Reranker 状态必须是 `ready`——不满足就直接抛异常终止评测，而不是让评测在"其实已经退化成纯 BM25"的情况下悄悄跑完再产出一份好看的报告。

### 3.2 十七个确定性 Evaluator：分别在检查什么

[backend/app/evaluation/evaluators.py](../../backend/app/evaluation/evaluators.py)（237 行）里 `DETERMINISTIC_EVALUATORS` 列表实际注册了 **17 个**函数，不是笼统的"覆盖路由正确性、Schema 校验..."几个类别就能概括的。按检查对象分组：

**路由与结构（3个）**：`route_correctness`（31-34行）比较 `case_analysis.next_action` 和期望路由；`schema_validity`（37-44行）检查 `final_answer` 是字符串、`case_analysis` 是字典，路由是 `research` 时还要求 `evidence_packet` 也是字典——这是最基础的"没崩、字段齐全"检查；`retrieval_status_correctness`（47-53行）比较证据包（或纯检索输出）里的 `retrieval_status`（`matched`/`no_match`/`tool_error`）和参考里的期望状态，是判断"这一轮检索到底算不算有结果"这个分类问题是否判对的入口。

**检索质量（6个，只在 `retrieval` 层有意义）**：`retrieval_recall_at_k`（85-90行，Top-5 内命中的期望文档比例）、`retrieval_mrr`（93-100行，第一个命中文档排名的倒数）、`retrieval_hit_at_1`/`retrieval_hit_at_3`（103-116行，共用 `_document_hit_at` helper）、`retrieval_gold_rank`（119-129行，Gold 文档的名次，未命中记为"列表长度+1"）、`exact_article_hit`（132-137行，要求期望的 **chunk** 级 ID 全部出现在 Top-5，比文档级 Recall 更严格）。这几个函数都通过 `_ranked_documents`/`_ranked_chunks`（56-73行）优先读 `evidence_packet.evidence_items`，读不到才退回 `retrieval_results`——同一套函数因此既能评测纯检索 Target，也能评测经过 Agent 编排后的输出。

**引用安全（2个）**：`citation_grounding`（140-149行）要求 `citations` 里出现的每个 `chunk_id`/`document_id` 都必须能在 `_evidence_keys`（76-82行，取自证据包）里找到，是一个纯粹的集合包含关系判断；`citation_precision`（152-162行）算的是"引用里有多少比例是有根据的"（分母是引用总数），和 Grounding 的"只要有一条引用越界就判失败"不同，Precision 是比例度量,可以用来观察越界程度而不只是有没有越界。

**`no_match_safety`（165-174行）**：不是简单检查"有没有返回结果"，而是三个条件同时成立才算安全——`citations` 为空、正文里没有用正则 `《[^》]+》|第[...]条` 匹配到的法条/条号字符串（防止模型在无证据时凭空编造出一个看起来像引用的表述）、并且正文里出现了"未检索到"加上"可引用法条"或"可直接引用"这类披露性措辞（防止模型不给引用但也不告诉用户"这是因为没查到"）。三个条件都是字符串/正则层面的判断，不涉及语义理解。

**其他安全约束（4个）**：`tool_trajectory`（177-183行，要求工具轨迹里的每一步 `agent` 字段都是 `legal_researcher`——检索工具不能被其他 Agent 越权调用）；`loop_limit`（186-194行，`tool_call_count<=4`、`model_call_count<=10`、`retry_count<=1`、`revision_count<=1` 四个上限同时满足）；`latest_fact_priority`（197-203行，回答必须包含参考里标注的"最新事实"、不能包含"旧事实"）；`tenant_isolation`（206-210行，回答里不能出现属于其他租户的禁用词）。

**`completion_success`（213-216行）**：兜底检查——`final_answer` 非空且 `errors` 列表为空，是"这一轮跑完了没崩"的最后一道信号。

### 3.3 两套结构化 LLM Judge：一次 JSON 请求覆盖多个维度

本项目实际有两套独立的 Judge 实现，服务不同的评测目的，都用同一个核心技巧——`ChatOpenAI(...).bind(response_format={"type": "json_object"})`，走的是 OpenAI 兼容的 **JSON 模式**（不是工具调用/Function Calling 那条结构化输出路径），配合 Pydantic Schema 在收到响应后做 `model_validate` 校验。

**`LegalQualityJudge`**（[backend/app/evaluation/judge.py](../../backend/app/evaluation/judge.py)，57 行全文）：`JudgeScores`（10-19行）定义了 8 个 1-5 分的字段——`legal_issue_coverage`（争议点覆盖）、`evidence_consistency`（证据一致性）、`factual_fidelity`（事实忠实度）、`risk_calibration`（风险校准）、`completeness`（完整性）、`actionability`（行动性）、`clarity`（清晰度）、`helpfulness`（帮助程度），外加一个 `comment` 字段。构造时（23-35行）显式设置 `extra_body={"thinking": {"type": "disabled"}}` 关闭模型的思维链输出（省 Token，也避免 Judge 用推理内容"说服自己"打高分）、`streaming=False`；调用时（37-57行）把 `inputs`/`outputs`/`reference` 一起序列化成一条 human 消息，模型返回的 JSON 校验通过后，8 个字段各自除以 5 归一化成 `[0,1]`，前缀 `judge_` 拼成 LangSmith Feedback Key。

**`AgentQualityJudge`**（[backend/app/evaluation/agent_quality.py](../../backend/app/evaluation/agent_quality.py)，270 行全文）是更晚加入、专门服务 §3.5 提到的三套 Agent 质量套件的实现，比 `LegalQualityJudge` 多两处工程考量：

1. **按套件切换 Schema**：`SUITE_SCHEMAS`（168-172行）把 `factual-fidelity`/`reviewer-effectiveness`/`answer-quality` 三个套件分别映射到 `FactualJudgeResult`（4个打分维度 + 逐条 `FactAssessment` 事实状态标注）、`ReviewerJudgeResult`（`instruction_specificity`/`decision_quality` 两个打分 + 三个错误标签列表）、`AnswerJudgeResult`（8个打分维度 + 覆盖/遗漏的争议点 ID 列表）——同一个 Judge 类，靠构造参数 `suite` 决定用哪个 Schema、哪套字段。
2. **失败重试但不重复计费评测契约**：`__call__`（199-226行）最多尝试 2 次，只有 JSON 解析或 Schema 校验失败才重试，重试仍然算作"同一次评测调用"（不会被上游计成两次 Judge 调用）。

`reviewer-effectiveness` 套件还有一个容易被忽略的设计（`_feedback` 方法 228-263行的代码注释里明确写了"preserves the one-Judge-call-per-example contract"）：`error_category_recall`（该套件独有的指标，衡量 Judge 识别出的错误类别和参考里预先注入的错误类别有多少重合）**不是靠第二次 Judge 调用算出来的**，而是从同一次结构化响应里已经返回的 `detected_error_labels` 字段事后在 Python 里和 `injected_error_labels` 做集合交集计算——保证"每个样本只调用一次 Judge"这条成本纪律不会因为多算一个指标就被打破。测试 [`test_reviewer_judge_derives_error_category_recall_without_second_call`](../../tests/test_agent_quality_eval.py#L125) 专门覆盖了这一点。

### 3.4 RAG 消融：盲选资格筛选的真实实现在脚本层，不在模块层

这是本次重新核实里发现和旧版文档描述偏差最大的一处：**"盲选"的判定逻辑（Hybrid Top12 + 至少 2 个预声明干扰项命中）并不在 `backend/app/evaluation/challenge_datasets.py` 里**，这个模块只负责"生成候选"（`prepare_source_pack`，120-161行：用确定性轮询算法 `_round_robin_sources` 在各部法律间均匀抽取源 chunk，为每条 Reranker 候选预先声明最多 3 个同法条相邻 chunk 作为"硬干扰项"）和"校验响应格式"（`validate_and_build`，244-290行：检查问题长度、法律名称/条号/连续原文泄漏）。真正执行盲选筛选、把候选变成正式挑战集的代码在 [scripts/create_resume_challenge_datasets.py::qualify](../../scripts/create_resume_challenge_datasets.py#L487)（487-604行）。

完整流程（对照 `resume/eval-results.md` 第 1 节的实测记录核实过）：

1. **候选池扩容**：`DEFAULT_DENSE_SIZE=300`、`DEFAULT_RERANK_CANDIDATE_SIZE=600`、`DEFAULT_RERANK_SIZE=200` 三个常量定义在 `challenge_datasets.py`（20-22行）。第一次只生成 300 条 Reranker 候选时，实测只有 115 条能通过资格筛选，不够冻结 200 条正式集，于是把候选池扩容到 600 条——`build()`（`create_resume_challenge_datasets.py` 261-333行）用 `assert_unchanged_prefix`（`challenge_datasets.py` 164-199行）保证旧的 300 条前缀原样保留、只在后面追加新的 300 条，扩容不会让已经生成过的内容发生变化。
2. **盲选筛选配置**（`_qualification_configuration`，352-368行）：`retrieval_mode="hybrid"`、`rerank_enabled=False`（**筛选阶段强制关闭 BGE**，这是"盲选"的核心——用不带精排的 Hybrid 检索去筛，BGE 的表现完全不参与筛选过程）、`top_k=12`、`required_distractors=2`。
3. **逐条判定**（`qualify` 487-604行）：对每条候选跑一次 `retrieval_mode="hybrid"`、`top_k=12` 的检索，`qualified = gold_in_top_12 and len(retrieved_distractors) >= 2`——Gold chunk 必须进入 Hybrid Top12，且预先声明的干扰 chunk 里至少有 2 个也真的被检索到（证明这确实是一个"容易和干扰项混淆"的困难样本，不是随手能命中的简单样本）。
4. **实测结果**：600 条候选里 212 条通过（`resume/eval-results.md` 记录拒绝原因中 385 条是"干扰项不足"，3 条是"Gold 未进入 Top12"，通过率 35.33%），按冻结顺序取前 200 条（`_select_qualified_candidates`，458-484行）写入正式的 `lawstation-reranker-challenge-v1` 数据集，manifest 里额外记录 `rerank_enabled_during_qualification: false`，供 `run_resume_rag_challenge_eval.py::_dataset_status`（72-121行）在正式跑对比实验前做一次"资格清单确实是在关闭 BGE 状态下生成的"完整性校验。

这套设计完整对应 §2.5 讲的"预注册"原则：**生成候选、声明干扰项、执行筛选**三步全部不读取 BGE 的真实检索结果，只有筛选完成、正式挑战集冻结之后，才第一次拿这 200 条去分别跑 RRF 和 BGE 两组实验做效果对比。

固定实验环境的另一半做法是把数据集 SHA256、法规索引指纹、Embedding digest、Reranker model revision、Git Commit、随机种子全部写进报告——`resume/eval-results.md` 里每一组实验都能顺着这些标识追溯回具体的代码和数据版本。

### 3.5 Reviewer 消融：8 条错误草稿 + 4 条安全草稿具体怎么构造

`Reviewer 有效性`实验的思路值得单独拎出来讲：它不是简单问"Reviewer 通过率是多少"，而是构造一批**已知包含错误的草稿**，衡量 Reviewer 能不能正确识别（检出率）、会不会错杀正常草稿（误拒率）、给出的修正指令是否具体（Judge 打分）——把"审核环节"当成一个二分类器，用检出率/误拒率这套混淆矩阵语言去评测。

数据集本身在 [scripts/create_agent_quality_datasets.py::reviewer_cases](../../scripts/create_agent_quality_datasets.py#L110)（110-147行）构造，一共 12 条，写入 `lawstation-reviewer-effectiveness-v1`（对应 `resume/eval-results.md` 里的"12条"）：

- **8 条注入错误的"unsafe"草稿**（117-125行），每条都手工构造了一个具体的错误类型标签：`forged_chunk`（引用了不存在的伪造 chunk ID）、`irrelevant_evidence`（用不相关的证据支持无关结论）、`unsupported_claim`（下了"一定胜诉"这种没有证据支持的断言）、`invented_amount`（编造赔偿金额）、`stale_fact`（在已有更新事实覆盖的情况下仍使用旧金额）、`no_match_hallucination`（检索无结果时仍编造法条引用）、`certainty_overclaim`（把"可能承担责任"夸大成"已经证明承担全部责任"）、`tool_error_verified`（工具调用失败的情况下却声称"已完成法规核验"）。
- **4 条正常的"safe"草稿**（126-131行）：分别对应"有证据支持的正常回答"、"无匹配结果时的规范披露"、"工具异常时的规范披露"和"低置信度回答仍保持安全表述"四种应该被 Reviewer 放行的正常情形。

评测端由 [backend/app/evaluation/agent_quality.py::reviewer_effectiveness_evaluators](../../backend/app/evaluation/agent_quality.py#L62)（62-87行）把这 12 条草稿的实际 Reviewer 处理结果映射成指标：`detected = not review.approved and action != "finalize"`（Reviewer 判定了"需要处理"）；`reviewer_detection_recall` 只在 `unsafe` 样本上算（8 条里 Reviewer 真的检出了几条）；`reviewer_detection_precision` 只在"被检出"的样本上算（检出的里有多少确实是真的 unsafe）；`reviewer_false_rejection_rate` 只在 `safe` 样本上算（4 条正常草稿里有几条被错误拦下）；`unsafe_draft_escape_rate` 检查修订后的最终回答是否仍然包含每条草稿预先标注的 `forbidden_terms`（比如 `invented_amount` 那条的 `forbidden_terms=["十万元"]`）——这是判断 Reviewer 拦下来之后，问题是否真的被修正掉，而不只是"拦下来了但没改对"。这几个函数返回 `None`（而不是 0 或 1）来表示"这个指标在这条样本上不适用"（比如 Precision 在没有检出任何东西的样本上不适用），`resume/eval-results.md` 里的检出率 `87.5%`（8 条里检出 7 条）和动作准确率 `83.33%` 都来自这套逐条映射后的汇总。

### 3.6 尚未接入评测闭环的机制：多轮对话场景蓝图生成器

通读 [backend/app/evaluation/conversation_scenarios.py](../../backend/app/evaluation/conversation_scenarios.py)（404 行全文）和 [backend/app/evaluation/scenario_catalog.py](../../backend/app/evaluation/scenario_catalog.py)（161 行全文）后发现，这是一套此前文档完全没有提到、且**目前还没有真正跑过 Agent** 的独立机制，值得单独记录清楚它的定位和现状，而不是含糊归进已有小节。

这套机制解决的问题和前面几节不同：前面的确定性 Evaluator/Judge 测的是"单轮问答的回答质量"，而 `conversation_scenarios.py` 生成的是**多轮、可能涉及并发和断线重连的操作序列**——比如"用户在等待第一轮检索结果时切换到另一个会话，原任务要在后台继续跑"（`concurrency-user-switch`）、"SSE 连接断开后重连，事件序列要能正确重放且不重复"（`sse-reconnect-replay`）、"同一会话内重复发起生成请求应该收到 409"（`same-conversation-busy`）。`blueprint_definitions()`（105-233行）定义了 12 个这样的确定性蓝图，覆盖 `routing`/`rag`/`memory`/`concurrency`/`durable_run` 五个类别，蓝图本身（包括每一步的具体动作序列、期望的事件类型、期望的终态）完全由代码写死，**LLM 只负责填空——为蓝图里预留的 `message_slot` 生成用户会说的具体话术**（`generation_prompt`，259-278行），不会生成测试逻辑本身。

生成出的候选要经过和 §3.4 类似强度的校验（`materialize_scenarios`，296-382行）：敏感信息正则（手机号/身份证/银行卡/邮箱/密钥，29-35行 `_SENSITIVE_PATTERNS`）、提示注入关键词黑名单（36-42行 `_INJECTION_PATTERNS`）、和 §3.4 相同风格的"连续 7 字符源文本泄漏"检测（`_source_leak`，285-293行）。校验通过的场景最终由 `ScenarioCatalog`（`scenario_catalog.py`）在启动时加载——但加载本身也不是"文件存在就能用"：必须有对应的 `.manifest.json`，SHA256 必须匹配，`status` 字段必须是 `"frozen"`，路径还被 `_resolve_path`（41-53行）强制限定在 `evals/conversations/` 目录下（防止配置误指到项目外任意路径）。

`resume/eval-results.md` 第 12 节明确记录了这套机制目前的真实状态：**2026-08-31 已经用 `deepseek-v4-flash` 基于 18 类蓝图生成了 36 条合成场景并通过了确定性校验，但这 36 条场景截至目前还没有被拿去驱动 Agent 真实运行**——报告原文写明"该结果只证明生成、约束和冻结流程工作正常……不能计为端到端通过率"。这是一个诚实的"半成品"状态：数据生成和校验管线是完整、可复现的，但它和 §3.1-3.5 的评测循环之间还缺最后一环——目前没有代码把这些场景喂给一个能模拟 SSE 断线、并发用户切换的执行器去驱动真实 Agent 调用并收集结果。

### 3.7 报告归档、确定性抽样与资源预算：分级执行的真实实现边界

四级评测分别是什么、由什么脚本触发，`scripts/run_langsmith_eval.py` 里 `argparse` 对 `--profile` 的合法取值定义得很明确（916行）：`choices=["learn", "smoke", "compare", "release"]`——文档之前用的这几个名字拼写和代码完全一致，不是简化或转译。但四级之间的边界，代码里实际强制的部分比"四级递进"这个说法暗示的要窄：

**`learn`（零成本）是真的不调用任何外部服务，甚至不调用 Agent**。`_run_learn`（302-354行左右）不会构造 `AgentRuntime` 或调用任何 LLM，而是用 `_learning_output`（267-299行）**直接从数据集里的期望字段拼出一份"看起来合理"的固定输出**（比如路由等于期望路由、`no_match` 时套用一段固定话术），再把这份自造输出喂给 §3.2 的确定性 Evaluator 跑一遍。测试 [`test_learn_profile_has_no_external_resource_usage`](../../tests/test_eval_resource_budget.py#L55) 断言 `planned_traces`/`agent_model_calls`/`judge_calls` 全部为 0——这解释了为什么 `learn` profile 产出的指标不能代表真实能力：它测的是"Evaluator 函数本身写对了没有"，不是"Agent 表现好不好"。

**`smoke`**（`_profile_args`，355-368行）和 `learn` 共享同一份 `PROFILE_DEFAULTS["smoke"]`（101-106行：`lawstation-agent-v3` 数据集、`component` 模式、6 个样本、覆盖 6 个类别），会强制 `upload_results=False`——真实调用 Agent（和真实的 DeepSeek 模型），但结果只留在本地，不产生 LangSmith Trace。

**`compare` 和 `release` 在代码里没有任何差异化处理**——除了被当作字符串标签写进 `_profile_args` 的 `else` 分支（走默认数据集 `lawstation-e2e-v2`、默认模式 `component`）和 `ReportRun` 的运行目录命名（`{run_id}-{profile}`）之外，`run_langsmith_eval.py` 没有任何逻辑因为 `profile=="release"` 而收紧阈值、放大样本量或触发额外校验——`--max-examples`、是否 `--upload-results`、跑多大样本，全部由调用方手动传的命令行参数决定。"`compare` 是小样本云端对比、`release` 是重大变更后的扩大样本"这句话描述的是**团队约定的使用方式**，不是代码强制的行为差异。

**资源预算的记账和拦截是两件独立的事，目前只接了前者**。[backend/app/core/resource_budget.py::MonthlyResourceBudget](../../backend/app/core/resource_budget.py)（143 行全文）同时提供了"先检查、超限就抛 `ResourceBudgetExceeded` 拒绝执行"的拦截接口（`reserve`/`reserve_many`/`check_many`，67-114行）和"事后如实记录用量、不做限制"的记账接口（`record_many`，125-134行）。但 `run_langsmith_eval.py` 里 `ledger.record_many(...)`（743-747行）只在评测**跑完之后**记录本次实际消耗的 `evaluation_traces`/`agent_model_calls`/`judge_calls`，评测开始前算出的 `_budget_plan`（395-407行）只是写进预览 JSON 里的 `planned` 字段（`preview` 里显式标注 `"monthly_budget_enforced": False`），**并没有拿这份计划去调用 `reserve_many`/`check_many` 做真正的执行前拦截**——换句话说，`learn`/`smoke`/`compare`/`release` 四级之间的成本升级目前完全依赖人工按顺序手动执行、自己看着预算走，不是代码强制"预算不够就跑不动"。真正会在代码层面用 `reserve()` 做强制拦截的，是生产环境里 `backend/app/observability/langsmith.py::LangSmithObservability._production_slot`（280-308行）对**生产流量 Trace 采样预算**（`production_traces`）的控制——这是给线上可观测性用的预算门，和评测分级执行是同一个 `MonthlyResourceBudget` 类的两种不同用法。

支撑这套分级执行可复现的另外两块基础设施：[backend/app/evaluation/profiles.py](../../backend/app/evaluation/profiles.py)（111 行全文）的 `select_cases`（48-79行）用"`sha256(种子:样本内容哈希)` 排序取前 N"的方式做确定性抽样——同一个种子、同一份数据集，任何时候重跑都会选出完全相同的样本子集，这是 `resume/eval-results.md` 里"Baseline/Candidate 样本内容哈希一致"这条校验能够自动完成的基础；[backend/app/evaluation/reporting.py](../../backend/app/evaluation/reporting.py)（260 行全文）的 `ReportRun` 用"先写 `.tmp` 再 `os.replace` 原子替换"加上 `fcntl` 文件锁（237-260行）保证每次评测跑的报告目录不会被并发写坏，`run-manifest.json`/`latest.json`/`latest-success.json` 三份索引让"最近一次成功的评测在哪"始终有据可查。

## 4. 设计取舍

**为什么确定性规则和 LLM Judge 要分开，而不是都交给 Judge？** 安全类硬规则（引用是否越界、有没有编造法条）必须是确定性的、可以扩展到大样本的，如果交给 LLM Judge 打分,会引入本不必要的不确定性,而且成本高得多。只有真正无法用规则表达的语义维度才值得花钱调用 Judge。

**为什么资格筛选和效果测量要严格分离？** 如果筛选规则本身依赖被比较双方的表现,相当于"先看了答案再决定考不考这道题",消融实验的结论会失去说服力——这是本项目消融设计里最容易被忽略、但也是最关键的一条纪律。

**为什么坚持报告未达标的结果？** Reviewer 检出率 87.5% 低于 90% 门禁，报告选择直接写"不能宣称质量门禁通过"而不是换个说法回避。这个取舍背后的逻辑是：一份会隐藏坏消息的评测报告,慢慢会失去被信任的资格,不如老实报告问题、把它列为下一步优化方向。

**为什么小样本实验（n=3、n=6）还要保留在文档里？** 它们不能用来外推生产表现,但可以用来验证"消融设计本身是否work"、发现工程 Bug、演示方法论——报告把这类结果和大样本结果物理分区,而不是删掉或混在一起,是为了让读者知道每个数字的可信边界。

**为什么要提供 `reviewer_target`/`counsel_quality_target` 这种"从图中间冻结开始跑"的 Target，而不是永远跑完整三 Agent 图？** 如果永远从 Case Analyst 跑到 Finalize，Reviewer 检出率的波动里会混进 Case Analyst 判断路由是否正确、Researcher 检索是否命中这两个上游环节的噪声，出了问题很难判断锅在哪一层。把上游产物冻结成固定输入，让被测的那个 Agent 在**真实生产代码路径**上跑（不是 Mock 它的行为，而是不产生它的上游输入的随机性），是"单 Agent 消融"和"单元测试"这两种思路在 Agent 系统里的结合——测的是这一个组件本身，同时又不是脱离生产代码逻辑的纯 Mock。

**为什么 `learn` profile 不调用 Agent、而是直接拼一份"应该长这样"的假输出？** `learn` 的目的是让开发者在写新的确定性 Evaluator 函数时能够零成本、秒级反馈地验证"这个函数的判断逻辑本身写对了没有"，而不是等一次真实的模型调用。用期望字段直接拼输出，相当于给 Evaluator 函数造了一份"标准答案输入"，如果 Evaluator 在这种输入下都判不对，说明问题出在 Evaluator 自己的代码里，不需要先花一次真实调用去排除"是不是 Agent 表现不好"这个变量。代价是 `learn` 产出的分数完全不能代表 Agent 真实能力——见 §3.7。

**为什么资源预算类同时提供了拦截接口和只记账的接口，评测脚本却只用了后者？** `reserve`/`check_many` 这类"先检查、超限就拒绝执行"的接口，在生产线上路径（`LangSmithObservability._production_slot`）上是必需的——线上流量不能因为一次意外的预算超支就把服务打挂，必须在真正发起追踪调用之前就决定"这次要不要上报"。但评测脚本目前的做法是先离线估算一次成本（`_budget_plan`），再手动决定要不要跑，跑完之后才用 `record_many` 如实记账——这更接近"人工审批 + 事后台账"的流程，而不是自动化门禁。这是一个明确可以改进但目前还没有接上的地方，详见 §6。

## 5. 易错点

- **把门禁目标值当成实际成绩**："要求 Recall@5 不回退"是门禁条件，不是"Recall@5 提升了多少"这个实际结果——混用这两者会让报告显得比实际情况更好。
- **样本量相同就假设两组实验可比**：n=100 的两组实验可能测的是完全不同的变量（比如一组测检索模式、另一组测精排选择），不能只凭样本量一样就直接横向比较数字。
- **只报告收益、不报告代价**：BGE 精排提升排序的同时带来约 1.3 秒的延迟增加，报告如果只说"提升了排序质量"而不提延迟代价，读者会做出错误的成本估算。
- **字符串匹配指标和语义 Judge 结论冲突时，直接采信更好看的那个**：本项目事实忠实度实验里就出现过字符串检查把"旧记录已被正确更正"误判为"泄漏旧事实"的假阳性，正确做法是记录这个冲突并标注"需要升级为 Claim/Negation-aware evaluator"，而不是挑一个更好看的数字写进结论。
- **测试样本或提示词泄漏进了评测/提示词本身**：如果评测样本的措辞出现在系统提示词或 few-shot 示例里，模型等于"提前看过题"，分数会虚高但不代表真实能力,这是评测设计时要主动检查的数据泄漏风险。
- **以为 `targets.py` 里能看到的函数都在实际评测路径上**：`component_target`/`live_target` 这两个独立函数目前没有任何脚本或测试引用，是历史遗留的死代码；真正在用的是 `agent_target`/`reviewer_target`/`counsel_quality_target`/`RetrievalTarget`。读这类协作模块时，先确认调用方，再判断这段代码是不是"在跑的那条路径"。
- **把 `compare` 和 `release` 当成代码里有真实差异的两个门禁级别**：目前这两个 profile 在 `run_langsmith_eval.py` 里走的是完全相同的代码分支，样本量、是否上传全部由命令行参数决定，"`compare` 小样本、`release` 扩大样本"只是团队约定的使用方式，不是代码强制的行为。
- **以为资源预算的 `record_many` 会在超支时拦截评测继续执行**：它只是事后如实记账，不会抛异常、不会阻止下一次调用；`MonthlyResourceBudget` 里真正会拦截执行的 `reserve`/`reserve_many`/`check_many` 接口目前只接在生产环境的 Trace 采样路径上，没有被评测脚本使用。

## 6. 生产化差距与面试应对

本项目的评测体系在方法论纪律上（分层、消融、盲选、诚实披露）已经相当完整，但离生产级的持续评测体系还有明确差距：

| 维度 | 本项目现状 | 生产环境通常怎么做 | 面试怎么答 |
|---|---|---|---|
| 数据标注来源 | 全部为合成数据或源数据派生集，`human_verified=false` | 人工标注平台（Label Studio、Scale AI）产出的真人标注基准，通常配合标注者间一致性（inter-annotator agreement）指标 | "当前样本明确标注为合成/派生数据，不冒充人工标注；生产化第一步是投入真实法律专业人士标注一批黄金集" |
| 评测触发方式 | 手动执行分级脚本（learn/smoke/compare/release）；`compare`/`release` 在代码里没有差异化逻辑，只是同一批代码手动传不同参数跑出来的约定 | 接入 CI/CD，代码或 Prompt 变更时自动触发回归评测（类似 promptfoo、DeepEval 在 CI 里跑），阻断有回归的合并 | "当前评测是手动分级执行，且分级之间的边界靠团队约定而不是代码强制；生产化会把确定性 Evaluator 那一层接入 CI，用代码而不是约定去区分级别、防止有安全回归的改动被合并" |
| 评测资源预算的强制力 | `MonthlyResourceBudget` 已经实现了 `reserve`/`check_many` 这类"先检查、超限就拒绝"的拦截接口，但评测脚本（`run_langsmith_eval.py`）只调用了事后记账的 `record_many`，执行前的预算计划不会真正拦截超支的评测跑起来 | CI/CD 里的评测网关在发起真实调用前就用配额系统做拦截，超限直接失败并给出明确原因，而不是跑完才知道超了 | "预算记账已经做了，但还没有接成'超支就跑不动'的强制门禁；这是从'人工看着预算跑'到'系统自动拦截'之间的差距" |
| 在线评测 | 无持续在线抽样评估——`LANGSMITH_ONLINE_EVAL_SAMPLE_RATE` 只在 `Settings` 里声明了默认值 `0.0`，全仓搜索确认代码里没有任何地方读取或使用这个配置项，是一个完全未接线的占位符，比"配置了但没有调度器"更进一步——目前连读取的代码都不存在 | 对生产流量做抽样异步评测，配合隐式反馈（点赞/点踩/复制/重新提问）持续监控质量漂移 | "线下消融证明了改动有效，但线上还没有持续监控；甚至连采样开关的读取逻辑都还没写，这是从'一次性验证'到'持续质量保障'之间实打实的差距" |
| 统计显著性 | 报告绝对/相对变化，未做显著性检验 | 大样本场景通常会做假设检验或置信区间估计，判断观察到的差异是否显著 | "当前样本量下，主要通过控制变量和盲选来保证结论可信，还没有引入正式的统计检验；样本量足够大之后应该补上" |
| Golden Set 漂移监控 | 无 | 定期在固定标准样本集上重跑，观察分数是否随模型/索引更新退化 | "当前每次实验是独立跑一次；生产化应该建一个固定的回归基准集，每次发布前自动重跑对比" |
| 人工审核回流 | 用户反馈（赞/踩）落库但未形成标注回流闭环 | 低分/点踩样本进入人工审核队列，审核结果回流成新的评测/训练数据 | "反馈已经持久化，但还没有形成'低分样本 → 人工审核 → 回流评测集'的完整闭环，这是生产化的下一步" |
| 多轮/并发场景评测 | 已经建好一套确定性蓝图 + LLM 填话术的生成管线（`conversation_scenarios.py`），并冻结了 36 条通过校验的场景，但截至目前没有代码把这些场景接到一个能驱动真实 Agent、模拟并发用户和 SSE 断线重连的执行器上——生成完就停在原地 | 场景数据接入真实的集成测试执行器，在 CI 或定期任务里驱动真实的多 Agent 并发/断线重连流程，观察终态是否符合预期 | "多轮场景的数据生成和冻结管线已经打通并通过了自检，但执行器这一环还没接上；这是当前评测体系里最明显的'半成品'——诚实说清楚差在哪一步，比含糊说'已支持多轮场景测试'更重要" |

## 7. 动手验证方式

1. 分别跑一次 `learn`（零成本）和 `smoke`（本地真实调用）profile，对比两次报告里 `missing_required_metrics`、`resume_eligible` 等字段的差异，理解为什么前者产出的数字不能用于结论——再读一下 `_learning_output`（`scripts/run_langsmith_eval.py`），确认 `learn` 是直接拼装期望字段而不是调用 Agent。
2. 找一组本项目已经跑过的消融实验（比如 BM25 vs Hybrid），检查两份报告的 Dataset SHA256、索引指纹、`top_k` 是否完全一致——这是判断"这组对比是不是真正的单变量消融"的第一步。
3. 故意构造一个"字符串匹配指标和语义判断会冲突"的样本（比如回答里出现了"旧记录已被更正为新值"这类否定语境表述），观察确定性 Evaluator 和 LLM Judge 是否给出不一致的判断，体会为什么两种方法要并存而不是只选一种。
4. 读一遍 `scripts/create_resume_challenge_datasets.py::qualify`，找到 `rerank_enabled=False` 这一行，思考：如果把它改成 `True`（筛选时也打开精排），盲选还成立吗？为什么"筛选配置"本身也要作为盲选纪律的一部分被检查，而不只是看筛选规则的文字描述。
5. 在 `backend/app/core/resource_budget.py` 里对比 `reserve_many` 和 `record_many` 两个方法的实现，再去 `scripts/run_langsmith_eval.py` 里确认实际调用的是哪一个——体会"预算记账"和"预算拦截"这两件事在代码里可以是完全独立的，接了前者不代表接了后者。

**自测题：**

- 如果一组消融实验的"资格筛选规则"会读取 Candidate（比如 BGE 精排后）的实际排序结果来决定样本要不要进入最终对比集，这个消融实验的结论还可信吗？为什么？
- Reviewer 检出率 87.5%、误拒率、动作准确率这三个指标，分别对应经典分类问题里的哪些概念（提示：想想 Precision/Recall/混淆矩阵）？
- `reviewer_target` 和 `counsel_quality_target` 都会跳过 Case Analyst 和 Researcher、从冻结的中间产物开始跑图。如果评测发现 Reviewer 检出率很低，这个结果能不能说明"三 Agent 编排整体表现差"？为什么单 Agent 消融的结论边界比端到端消融更窄，同时又比纯 Mock 更可信？
- `compare` 和 `release` 在代码里走的是同一个分支，区别只在人为传参。如果面试官问"你们怎么保证 release 级别的评测不会被误当成 compare 级别的小样本跑掉"，诚实的回答应该承认什么、又该强调哪一部分是有代码保障的（提示：想想 Dataset SHA256、`resume_eligible`、`missing_required_metrics` 这几个字段谁负责兜底）？
