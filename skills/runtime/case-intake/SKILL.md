---
name: case-intake
version: 1.0.0
description: 复杂案情的主体、法律关系、时间线、金额、诉求、冲突与缺失事实结构化。
allowed_agents:
  - case_analyst
allowed_tools: []
output_schema: CaseIntakeResult
---
# Case Intake

仅依据用户本轮消息、近期对话和服务端记忆快照整理案情。当前消息与历史冲突时采用当前消息，并把冲突明确列出。

不要补造姓名、日期、金额、证据或法律关系。疑问、假设和第三人说法不得标记为已确认事实。输出必须符合 `CaseIntakeResult`，供后续 Agent 作为结构化背景使用。
