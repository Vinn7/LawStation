# LawStation 量化评测实测结果

> 最新实验时间：2026-08-26（Asia/Singapore）。本页只记录实际运行结果，不使用门禁目标值替代成绩。

> 最新运行目录：`evals/reports/runs/20260826-095648-160455-compare/`。六组正式RAG消融与六类Agent Fixture均已完成。旧预算拦截产生的失败manifest保持原样，恢复结果记录在`POST_RECOVERY_SUMMARY.json`。

## 0. 最新套件状态

`scripts/run_resume_rag_challenge_eval.py` 完成了模块测试、数据校验和 100/300/200 三组 Baseline/Candidate，共 1,200 次本地检索。六份 RAG 报告均为 `missing_required_metrics=[]`、`local_reproducible=true`、`resume_eligible=true`。测试不上传 LangSmith、不调用 Judge；Ollama 与 TEI 为本地推理。

首次300条候选实测只有115条满足资格：其中183条的Gold已进入Top12但只命中0～1个预声明干扰项，
仅2条Gold未进入Top12。2026-08-26已保留旧300条不可变前缀并追加到600条候选；使用固定Hybrid Top12、
至少2个预声明干扰项且关闭BGE的规则完成全量资格筛选，共212条通过、388条拒绝，通过率35.33%。
拒绝原因中385条为干扰项不足、3条为Gold未进入Top12；最终按冻结顺序选择前200条，三个数据集校验均为
`errors=0`。候选SHA为`358852283a4c74d2df5d948152a4605033d65f0dc9f77b9b13f8cc37c4bcd097`，
资格指纹为`e9c316218cdd4051fab3868f715ba1b359e781314b6796b8813b650d1c99d4b9`。

正式运行使用固定的 200 条挑战集完成了 RRF/BGE 对比；资格过程与效果实验仍严格分离。

## 1. 最新实验快照

- Git Commit：`f13ee18c2ac6dac769aa5358123f4747d7a96a10`；正式运行前 tracked worktree clean。
- 法规数据 SHA256：`8d1b4832c56c90d0cf8782aae70ee9961199b3d16ea4946933639485c21b8f86`。
- 索引指纹：`9c95a6fe9f682fd3ca41d097e8b7463180503620d39e4b724a381f9fde718f0d`。
- 索引规模：55,374 chunks；Embedding 为 Ollama `qwen3-embedding:0.6b`、1024 维。
- 固定随机种子：`42`。
- 正式实验前指定模块回归：60 passed；预算改造与实验补跑后的全量离线回归：139 passed、2 条第三方 warning（Pydantic Settings 与 LangSmith），无项目测试失败。
- 持久化 AgentRun、准确性闭环与运行时 Skill 改造后的离线回归：后端 153 passed、前端 14 passed，生产构建通过。新增200条检索校准集、30条事实冲突集和24条 Skill 路由 fixture 只完成冻结/结构与确定性边界校验，尚未运行真实模型校准或 Skill 路由基准，因此不得将目标门禁写成实测成绩。

## 2. Dense 挑战：BM25 vs Hybrid（n=300）

数据集 SHA256：`31bda76e9e14719c42b4cf6baff864e721620797d563c34ca3ceac8f81f77f68`。

| 指标 | BM25 | Hybrid | 绝对变化 | 相对变化 |
|---|---:|---:|---:|---:|
| Recall@5 | 86.67% | 96.33% | +9.67 pp | +11.15% |
| MRR | 0.7699 | 0.8705 | +0.1006 | +13.07% |
| Hit@1 | 69.67% | 80.33% | +10.67 pp | +15.31% |
| Hit@3 | 84.00% | 94.00% | +10.00 pp | +11.90% |
| Gold 平均排名 | 1.93 | 1.44 | -0.49 | 改善 25.56% |
| 平均检索耗时 | 176.36 ms | 263.17 ms | +86.81 ms | +49.22% |
| p95 检索耗时 | 233.54 ms | 332.34 ms | +98.79 ms | +42.30% |

结论：Hybrid 对口语化和语义改写查询产生了明确召回与排序收益，同时引入约 87 ms 平均耗时。

## 3. Reranker 挑战：RRF vs BGE（n=200）

数据集 SHA256：`931d05a1d10ba4c4082d0134d30d568235c3da5406782030b33205cbc7a086b2`。正式集从 600 条候选按预先冻结的 Hybrid Top12 双干扰项规则选出，不读取 BGE 结果。

| 指标 | RRF | BGE | 绝对变化 |
|---|---:|---:|---:|
| Recall@5 | 99.50% | 100.00% | +0.50 pp |
| MRR | 0.9725 | 0.9838 | +0.0113 |
| Hit@1 | 95.00% | 97.00% | +2.00 pp |
| Hit@3 | 99.50% | 99.50% | 0 |
| Gold 平均排名 | 1.07 | 1.04 | -0.03 |
| 平均总检索耗时 | 262.02 ms | 1568.97 ms | +1306.96 ms |
| BGE 精排 p95 | — | 1740.61 ms | — |

BGE `rerank_applied_rate=100%`、`rerank_degraded_rate=0%`，固定 revision 为 `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`。质量门禁和 3 秒 p95 目标均通过，但收益以约 1.31 秒平均额外耗时为代价。

## 4. 通用回归：RRF vs BGE（n=100）

| 指标 | RRF | BGE | 绝对变化 |
|---|---:|---:|---:|
| Recall@5 | 98.00% | 100.00% | +2.00 pp |
| MRR | 0.9242 | 0.9750 | +0.0508 |
| Hit@1 | 88.00% | 96.00% | +8.00 pp |
| Gold 平均排名 | 1.30 | 1.10 | -0.20 |
| Retrieval Status Accuracy | 80.00% | 80.00% | 0 |

BGE 没有造成召回回退，但 20 条预设 no-match 均被当前阈值返回候选。该问题属于召回阈值/空结果判别，不是 Reranker 排序问题，必须作为后续优化项披露。

## 5. Agent Fixture 冒烟（n=6，工程门禁）

- 覆盖 casual、clarification、matched、no_match、tool_error、memory 各1条。
- Route、Schema、Citation Grounding、no-match Safety、Tool Trajectory、Loop Limit、Tenant Isolation、Completion Success均为100%，项目配置门禁通过。
- 平均模型调用数3.0，平均工具调用数1.33，LLM Reviewer调用率33.33%，实际消耗18次模型调用。
- 非门禁诊断：matched Fixture被Research Agent判为no_match，Retrieval Status Correctness为83.33%；memory Fixture未体现“最新一万元覆盖旧八千元”，Latest Fact Priority为83.33%。
- 该组件报告`resume_eligible=false`，只证明固定工程约束，不代表法律回答总体准确率。

证据：`evals/reports/runs/20260826-095648-160455-compare/agent-smoke.json`。

## 6. 当前可用于简历的表述

> 构建 BM25 + Ollama Qwen Embedding + FAISS + RRF 混合检索，在 300 条源法条约束的合成语义挑战集上，将 Recall@5 从 86.67% 提升至 96.33%，MRR 从 0.7699 提升至 0.8705；平均检索耗时从 176 ms 增至 263 ms。

> 使用 TEI 本地部署 `BAAI/bge-reranker-v2-m3`，对 Hybrid Top12 执行批量 Cross-Encoder 精排；在从 600 条候选按固定双干扰项规则冻结的 200 条合成排序挑战集上，将 Hit@1 从 95% 提升至 97%、MRR 从 0.9725 提升至 0.9838，精排 p95 为 1.74 秒，应用率 100%、降级率 0%。

> 建立可复现的 RAG 消融体系，以 Dataset SHA256、法规索引指纹、Embedding digest、Reranker revision、Git Commit 和时间戳报告固定实验环境，输出逐样本排名、分组指标、绝对/相对变化和失败边界。

> 基于LangGraph构建三Agent法律咨询链路，以casual、clarification、matched、no_match、tool_error、memory六类Fixture验证路由、Schema、chunk引用归属、no-match防幻觉、循环上限、租户隔离和完成状态，配置的关键工程门禁全部通过；平均模型调用3次，LLM Reviewer调用率33.33%。

不得写成真实用户或律师标注准确率；六条Agent Fixture只能表述为“关键工程门禁通过”，不能写“Agent准确率100%”。

## 7. 历史 100 条本地 RAG 消融（2026-08-25，保留作版本对照）

数据集为 `lawstation-live-retrieval-v1`，包含 80 条 matched 和 20 条 no_match。它使用真实法规 ID，但由源数据派生，`human_verified=false`，不能称为律师人工标注或真实用户查询集。

Baseline 与 Candidate 的 Dataset SHA256、100 个样本内容哈希、索引、代码和 `top_k` 一致，只改变检索模式。

| 指标 | BM25 | Hybrid | 绝对变化 | 相对变化 |
|---|---:|---:|---:|---:|
| Recall@5 | 0.9800 | 0.9800 | 0 | 0% |
| MRR | 0.9065 | 0.9242 | +0.0177 | +1.95% |
| Exact Article Hit | 0.9800 | 0.9800 | 0 | 0% |
| Retrieval Status Accuracy | 0.8000 | 0.8000 | 0 | 0% |
| Mean Retrieval Duration | 208.44 ms | 365.97 ms | +157.53 ms | +75.58% |

结论：Hybrid 没有提高前五召回率，但改善了正确法条的排序；代价是平均检索耗时增加约 75.6%。因此简历应表述为“MRR 提升”，不能表述为“Recall@5 提升”。

Retrieval Status Accuracy 只有 80%，说明当前 matched/no_match 判定仍有 20 条错误。现有汇总报告没有输出逐条失败归因，下一步应先增加错误样本明细，再标定阈值，不能直接为了提高分数修改基准数据。

证据报告：

- `evals/reports/runs/20260825-184145-290801-compare/rag-bm25-full.json`
- `evals/reports/runs/20260825-184225-827641-compare/rag-hybrid-full.json`

两份报告均满足：`sample_size=100`、`missing_required_metrics=[]`、`local_reproducible=true`、`resume_eligible=true`。

## 8. 历史 Agent Smoke（2026-08-25）

使用 `lawstation-agent-v3` 中固定的 matched/no_match Fixture，真实调用 DeepSeek，但不调用真实法规检索、不上传 LangSmith、不运行 Judge。

| 指标 | 结果 |
|---|---:|
| 样本数 | 2 |
| Route Correctness | 100% |
| Schema Validity | 100% |
| Citation Grounding | 100% |
| no_match Safety | 100% |
| Loop Limit | 100% |
| Tenant Isolation | 100% |
| Completion Success | 100% |
| 平均模型调用数 | 4.5 |
| 平均工具调用数 | 2.0 |
| LLM Reviewer 调用率 | 50% |

证据报告：`evals/reports/runs/20260825-184335-353180-smoke/smoke-component.json`。

由于 `n=2` 且 `resume_eligible=false`，这组结果只用于链路检查，不能作为总体 Agent 准确率或 Reviewer 优化收益写入简历。

## 9. 历史 LangSmith Reviewer 小样本对比

使用相同的 3 条 `matched/no_match/memory` Fixture、相同内容哈希和固定种子 42，对比
`always-llm` 与 `auto`。两组均完成 3 条根 Trace 和 3 次八维 Judge，
`missing_required_metrics=[]`、`langsmith_witness_complete=true`、`resume_eligible=true`。

| 指标 | Always-LLM | Fast Path | 变化 |
|---|---:|---:|---:|
| LLM Reviewer 调用率 | 100% | 33.33% | -66.67 个百分点 |
| 平均模型调用数 | 4.6667 | 4.3333 | -7.14% |
| LangSmith Total Tokens | 31,143 | 29,370 | -5.69% |
| 根 Run p50 延迟 | 36.24 s | 34.82 s | -3.92% |
| Citation Grounding | 100% | 100% | 持平 |
| no_match Safety | 100% | 100% | 持平 |
| Loop Limit | 100% | 100% | 持平 |
| Tenant Isolation | 100% | 100% | 持平 |
| Judge Evidence Consistency | 0.7333 | 0.7333 | 持平 |
| Judge Factual Fidelity | 0.8667 | 0.8667 | 持平 |
| Judge Risk Calibration | 0.8000 | 0.8000 | 持平 |
| Judge Helpfulness | 0.8667 | 0.8000 | -0.0667 |

结论：Fast Path 明显减少 Reviewer 进入率，但三 Agent 中 Research/Counsel 仍占主要延迟，所以
p50 总耗时只下降 3.92%。硬安全指标没有下降，Judge 帮助度却在 3 条样本上下降；受 `n=3/组`
限制，这一结果只用于证明消融设计和效率趋势，不能外推为生产质量提升。

有效证据报告：

- `evals/reports/runs/20260825-190001-050711-compare/reviewer-llm-baseline.json`
- `evals/reports/runs/20260825-190518-225859-compare/reviewer-fastpath-candidate.json`

实施期间发现云端 `aevaluate` 会继承 `LANGSMITH_TEST_CACHE`。第一次 Candidate 因 VCR 回放得到
不真实的毫秒级延迟，已明确排除。评测入口随后改为上传 Compare/Release 时强制关闭并恢复测试
缓存；上表 Candidate 来自修复后的真实模型调用。Baseline 是缓存首次生成方，其节点耗时和调用
日志均来自真实模型请求。

最初配置的 Workspace ID 不属于当前 PAT，导致 Dataset API 返回 403。通过 PAT 默认工作区确认
真实 tenant/workspace ID 后已修正 `.env`，数据集上传和两组实验均成功。

## 10. 历史简历表述（已被第 6 节最新结果替代）

> 在 100 条基于真实法规 ID 的源数据派生基准上完成 BM25 与 BM25 + Ollama Dense + FAISS + RRF 消融实验；两种方案 Recall@5 均为 98%，Hybrid 将 MRR 从 0.9065 提升至 0.9242（+1.95%），以约 75.6% 的平均检索延迟增长换取更优的正确法条排序。

> 设计确定性 Reviewer Fast Path，并在 3 条/组的固定分层见证样本上与 always-LLM 基线消融；在引用归属、no-match 安全、循环限制和租户隔离均保持 100% 的情况下，将 Reviewer 调用率从 100% 降至 33.3%，平均模型调用减少 7.1%，Token 减少 5.7%，根 Run p50 延迟降低 3.9%。

> 建立 retrieval/component/live 分层评测体系，通过 Dataset SHA256、索引指纹、Git Commit 和时间戳报告保证实验可复现；确定性指标覆盖 Recall@5、MRR、chunk 法条命中、引用归属、no-match 安全、循环上限和租户隔离。

不能写：p95 延迟降低。当前 LangSmith 小样本统计只返回 p50/p99，且 `n=3/组` 不足以形成稳定
p95；Judge Helpfulness 也没有提升。
# 多轮场景数据（未执行）

2026-08-31 使用 `deepseek-v4-flash` 基于18类确定性蓝图生成36条合成多轮对话场景。首次18次调用得到35条有效样例，1条因复用法规连续原文被拒绝；随后只对该变体执行1次定向修复，最终36/36通过确定性数据校验。数据集SHA256和真实19次模型调用记录在 `evals/conversations/lawstation-dialogue-scenarios-v1.manifest.json`。

该结果只证明生成、约束和冻结流程工作正常。36条场景尚未运行Agent，不能计为端到端通过率，也不能用于声明法律回答准确率。
