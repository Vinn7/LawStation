# LawStation LangSmith Eval Report

- 状态：`blocked: LangSmith monthly unique traces usage limit exceeded`
- 生成时间：`2026-08-22T06:19:44.968221+00:00`
- 简历可引用：`否（LangSmith Trace 导出不完整）`

> 检索集由 law.json 源数据派生并通过真实 document_id/chunk_id 校验，尚未经过律师人工标注；
> 因此可用于工程检索回归，不能宣称为专家标注的法律准确率。

## RAG：BM25 vs Hybrid

| 指标 | Baseline | Candidate | 变化 |
|---|---:|---:|---:|
| exact_article_hit | 0.98 | 0.98 | 0.0 |
| retrieval_mrr | 0.9065 | 0.924167 | 0.017667 |
| retrieval_recall_at_k | 0.98 | 0.98 | 0.0 |
| retrieval_status_correctness | 0.8 | 0.8 | 0.0 |

平均检索耗时：BM25 `309.811ms`，Hybrid `388.909ms`。20 条预设 no-match 查询均被返回候选，
因此检索状态准确率只有 `0.80`；这说明当前阈值仍需标定。

## Reviewer Judge 冒烟验证

- 固定 CaseAnalysis 成功让样本进入预期研究/复核路径；
- 8 个 Judge 指标均成功产生；
- 实际模型调用 4 次、工具调用 2 次、LLM Reviewer 命中率 100%；
- 随后 LangSmith 返回月度 unique traces 配额耗尽，阶段 2/3 已安全停止。

以上 RAG 数值是本地 evaluator 的实际结果，但对应 LangSmith Trace 未完整导出，暂不能作为最终简历成绩。

## 简历表述规则

升级 LangSmith 配额或等待月度额度恢复后，重新运行 `scripts/run_staged_langsmith_eval.py`。
只有 `langsmith_export_complete=true` 的实验才能用于简历；不得把门禁阈值或本次不完整结果写成正式成绩。
