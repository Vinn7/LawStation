---
name: procedure-roadmap
version: 1.0.0
description: 为起诉、仲裁、投诉、执行或申诉问题生成条件化程序路线和材料清单。
allowed_agents:
  - legal_researcher
  - legal_counsel
  - reviewer
allowed_tools:
  - search_laws
  - get_law_article
output_schema: ProcedureRoadmapResult
---
# Procedure Roadmap

先识别用户希望采取的程序和关键争议点。Legal Researcher 需要通过已授权的 MCP 法规工具核验管辖、期限和程序条件；其他 Agent 不得调用工具。

无法从 EvidencePacket 核验的具体期限、机关或前置条件必须标记为待确认，不得凭模型常识输出确定结论。输出必须符合 `ProcedureRoadmapResult`，并区分可能路径、关键步骤、材料、风险和下一步行动。
