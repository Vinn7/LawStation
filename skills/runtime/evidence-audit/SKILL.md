---
name: evidence-audit
version: 1.0.0
description: 围绕待证明事实整理现有证据、缺口、保全动作和真实性风险。
allowed_agents:
  - legal_counsel
  - reviewer
allowed_tools: []
output_schema: EvidenceAuditResult
---
# Evidence Audit

把用户陈述的材料视为“用户称其持有”，不得描述为已经核验。围绕争议点列出待证明事实、现有证据、证据来源、缺失证据、保全动作和真实性风险。

证据强度只能基于当前材料作条件化判断。不得承诺证据一定被法院采纳，不得把 Agent 生成的法律意见作为证据。输出必须符合 `EvidenceAuditResult`，回答中应给出简洁、可执行的证据准备建议。
