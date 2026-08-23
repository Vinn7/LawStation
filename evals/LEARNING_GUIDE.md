# LawStation 低资源评测学习指南

## 1. 六个基本概念

- **Dataset**：稳定保存的一组测试题。
- **Example**：一条输入、参考输出和分类元数据。
- **Target**：被评测的对象，可以是检索器、Agent 组件或完整应用。
- **Experiment**：同一 Target 配置在一个 Dataset 批次上的运行结果。
- **Trace**：一次 Target 调用的完整链路；子 Agent、模型和工具是同一根 Trace 的 Span。
- **Evaluator**：比较实际输出与参考要求并产生分数的函数。

先运行零资源学习模式：

```bash
conda run -n LawStation python scripts/run_langsmith_eval.py --profile learn \
  --report-name learn.json
```

报告中的 `details` 会并列展示输入、期望、固定输出、每个 evaluator 的分数和原因。该模式不访问 LangSmith、DeepSeek 或 Ollama。

命令默认把结果写入：

```text
evals/reports/runs/YYYYMMDD-HHMMSS-ffffff-learn/
├── learn.json
├── learn.csv
└── run-manifest.json
```

时间使用新加坡时区，微秒避免同一秒内运行冲突。`--plan-only` 完全只读，不创建报告；
`--no-report` 仅供自动测试使用。旧 `--output <path>` 仍兼容，但同时保留本次时间戳归档。

## 2. 关键指标

### Recall@5

前 5 个召回结果覆盖了多少期望文档：

```text
命中的期望 document_id 数 / 期望 document_id 总数
```

适合回答“应该找到的法规有没有被找到”，不衡量排序位置。

### MRR

第一个正确结果排名的倒数。正确结果排第 1 得 1，排第 2 得 0.5，完全没找到得 0。它比 Recall@5 更关注正确法规是否靠前。

### 引用归属率

最终回答中的每个 `chunk_id` 都必须存在于本轮 `EvidencePacket`。这项指标防止模型引用没有检索到的法条。

### no_match 安全率

检索正常但没有证据时，回答必须满足：引用为空、不编造具体法名或条号、明确披露未检索到可引用依据。

## 3. 两类 Evaluator

确定性 evaluator 是 Python 规则：便宜、稳定、适合 Schema、ID、上限和隔离等硬约束。LLM-as-judge 适合完整性、帮助程度和风险校准等语义质量，但会消耗模型额度且具有波动。

正确顺序是：先让确定性规则过滤硬错误，再只对少量边界样本调用 Judge。一次 Judge 请求同时返回 8 个评分维度，不为每个维度重复请求模型。

## 4. 公平比较

Baseline 和 Candidate 必须保持以下内容一致：

- 相同 example 内容哈希和顺序；
- 相同模型、Prompt 和索引版本，实验变量除外；
- 相同 Judge 版本；
- 相同重复次数。

例如比较 BM25 与 Hybrid 时，只改变 `rag_mode`；比较 Reviewer Fast Path 时，只改变 `review_mode`。

先查看资源计划：

```bash
conda run -n LawStation python scripts/run_staged_langsmith_eval.py \
  --profile compare --plan-only
```

真正上传必须显式增加 `--confirm-upload`。日常 Smoke 使用 `upload_results=False`，不会产生 LangSmith Trace。

分阶段 Compare/Release 只创建一个运行目录，所有 Baseline、Candidate、对比 CSV、
`experiment-manifest.json` 和 `EVAL_REPORT.md` 都归档在其中。中途失败或用户中断时，已完成阶段不会被删除，`run-manifest.json` 会记录失败阶段和安全错误摘要。最近运行可查看：

```text
evals/reports/runs/latest.json
evals/reports/runs/latest-success.json
```

## 5. 如何定位失败

1. `route_correctness` 失败：检查 Case Analyst 分类与路由 Prompt。
2. Recall/MRR 失败：检查分词、阈值、过滤条件和 RRF，不先改回答 Prompt。
3. `citation_grounding` 失败：检查 `chunk_id` 在 Research、Counsel、Finalize 间的传递。
4. `no_match_safety` 失败：检查无证据提示和确定性 Finalize 校验。
5. `tool_trajectory` 或 `loop_limit` 失败：检查 Graph 路由和调用上限。
6. 硬指标全通过但 Judge 低分：再检查回答结构、风险表达和行动建议。

## 6. 如何阅读报告和写简历

`local_reproducible=true` 表示本地数据、索引和代码版本足以复现指标；`langsmith_witness_complete=true` 表示小样本 Trace 已完整上传；`resume_eligible=true` 表示对应数字具备所声明范围内的证据。

完整本地 RAG 数据可以报告 Recall@5、MRR 和延迟。Agent/Judge 小样本必须同时写明样本数，不得把小样本结果描述成生产准确率，也不得把门禁阈值当作实测结果。
