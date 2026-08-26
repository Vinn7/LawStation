# LawStation 量化评测实测结果

> 实验时间：2026-08-25（Asia/Singapore）。本页记录实际运行结果，不使用门禁目标值代替成绩。

> 当前文件中的效果数字来自既有实验。新的统一量化入口和挑战数据已经准备完成，但100/300/200三组
> RAG消融与6类Agent完整套件尚未运行；不得用门禁目标或模板替换正式实验结果。

## 0. 新量化套件（数据已就绪、正式实验待运行）

`scripts/run_resume_rag_challenge_eval.py` 现在会在同一个时间戳目录完成：模块回归、挑战数据校验、
600条候选完整时的 Hybrid Top12双干扰项资格冻结、100条通用回归、300条 Dense 挑战、200条 Reranker 挑战和
6类 Agent Fixture 冒烟。产物新增 `experiment-snapshot.json`、`suite-gates.json` 和根据实测数字
生成的 `EVAL_REPORT.md`。该套件不上传 LangSmith、不调用 Judge；只有 Agent 冒烟少量调用
DeepSeek。下文继续保留历史实测，直到新套件实际完成后再原地更新。

首次300条候选实测只有115条满足资格：其中183条的Gold已进入Top12但只命中0～1个预声明干扰项，
仅2条Gold未进入Top12。2026-08-26已保留旧300条不可变前缀并追加到600条候选；使用固定Hybrid Top12、
至少2个预声明干扰项且关闭BGE的规则完成全量资格筛选，共212条通过、388条拒绝，通过率35.33%。
拒绝原因中385条为干扰项不足、3条为Gold未进入Top12；最终按冻结顺序选择前200条，三个数据集校验均为
`errors=0`。候选SHA为`358852283a4c74d2df5d948152a4605033d65f0dc9f77b9b13f8cc37c4bcd097`，
资格指纹为`e9c316218cdd4051fab3868f715ba1b359e781314b6796b8813b650d1c99d4b9`。

该结果只证明正式排序挑战集的冻结过程已完成，不能替代RRF与BGE的效果对比成绩。

## 1. 实验快照

- Git Commit：`252ada6e6375d4bc7a2a0adcc245616349250057`。RAG 实验时 tracked worktree clean；云端实验期间只修改了评测缓存隔离和对应文档，Agent Graph、Prompt、Fixture 和索引未修改。
- 法规数据 SHA256：`8d1b4832c56c90d0cf8782aae70ee9961199b3d16ea4946933639485c21b8f86`。
- 索引指纹：`9c95a6fe9f682fd3ca41d097e8b7463180503620d39e4b724a381f9fde718f0d`。
- 索引规模：55,374 chunks；Embedding 为 Ollama `qwen3-embedding:0.6b`、1024 维。
- 固定随机种子：`42`。
- 后端回归：97 passed，2 条第三方依赖 warning。

## 2. 100 条本地 RAG 消融

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

## 3. Agent Smoke

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

## 4. LangSmith Reviewer 小样本对比

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

## 5. 当前可用于简历的表述

> 在 100 条基于真实法规 ID 的源数据派生基准上完成 BM25 与 BM25 + Ollama Dense + FAISS + RRF 消融实验；两种方案 Recall@5 均为 98%，Hybrid 将 MRR 从 0.9065 提升至 0.9242（+1.95%），以约 75.6% 的平均检索延迟增长换取更优的正确法条排序。

> 设计确定性 Reviewer Fast Path，并在 3 条/组的固定分层见证样本上与 always-LLM 基线消融；在引用归属、no-match 安全、循环限制和租户隔离均保持 100% 的情况下，将 Reviewer 调用率从 100% 降至 33.3%，平均模型调用减少 7.1%，Token 减少 5.7%，根 Run p50 延迟降低 3.9%。

> 建立 retrieval/component/live 分层评测体系，通过 Dataset SHA256、索引指纹、Git Commit 和时间戳报告保证实验可复现；确定性指标覆盖 Recall@5、MRR、chunk 法条命中、引用归属、no-match 安全、循环上限和租户隔离。

不能写：p95 延迟降低。当前 LangSmith 小样本统计只返回 p50/p99，且 `n=3/组` 不足以形成稳定
p95；Judge Helpfulness 也没有提升。
