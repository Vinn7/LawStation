---
name: document-readiness
version: 1.0.0
description: 检查起诉状、答辩状等法律文书是否具备起草所需事实、请求和证据材料。
allowed_agents:
  - legal_counsel
  - reviewer
allowed_tools: []
output_schema: DocumentReadinessResult
---
# Document Readiness

首期只做文书起草前的信息就绪检查，不生成可直接提交法院或仲裁机构的最终文书。识别文书类型、已有信息、缺失信息、事实与请求一致性、证据缺口和敏感信息处理要求。

不得要求用户在公开回答中展示完整身份证号、银行卡号等敏感信息。信息不足时 readiness 必须是 `not_ready` 或 `needs_information`。输出必须符合 `DocumentReadinessResult`。
