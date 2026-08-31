---
name: lawstation-eval-review
description: 评估 LawStation 的 Agent、RAG、记忆或并发改造，并审核实验数字能否进入简历时使用。
---
# LawStation Eval Review

先确定变更影响和最低成本的评测层级。默认使用 mock、fixture 和本地确定性 evaluator；未经用户明确授权，不启动 LawStation、Ollama、TEI，不调用云端模型或上传 LangSmith。

正式对比必须固定 Dataset SHA256、样本 ID、随机种子、法规索引指纹、模型 revision、Prompt/Graph 版本和除实验变量外的配置。Baseline 与 Candidate 使用完全相同的样本，不得按 Candidate 结果删除失败项。

报告必须保留样本量、失败与降级记录、绝对变化、相对变化和限制。禁止用门禁目标冒充实测值，禁止把合成挑战集描述为真实用户或律师标注数据，禁止隐藏 Candidate 退化。

只有 `resume_eligible=true` 且关键指标完整的结果才能直接用于简历；工程安全 Fixture 只能描述为约束测试，不得描述为法律结论准确率。
