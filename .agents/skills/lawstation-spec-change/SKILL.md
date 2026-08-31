---
name: lawstation-spec-change
description: 修改 LawStation 功能、架构、API、数据模型、配置或安全规则时，执行项目专用的 SDD、最小变更、测试和 resume 同步流程。
---
# LawStation Spec Change

开始修改前读取实际代码并核对 `ai-context/SPEC.md`，不得只依据 README、历史需求或计划判断现状。

按以下边界完成变更：

1. 先明确目标行为、输入输出、失败语义、并发与所有权边界和验收标准。
2. 优先原地修改现有 symbol；保护无关代码和用户已有改动，不为局部需求删除重建文件。
3. 配置变化同步 `Settings`、`.env.example`、本地 `.env` 和 Spec；数据库变化使用 Alembic。
4. 用户域 SQL 必须在同一语句中限定 `tenant_id + user_id`，会话数据额外限定 `conversation_id`。
5. 按变更范围运行 Pytest、Ruff、前端测试和生产构建。
6. 同步更新受影响的 `resume/` 分册，并核验 `resume/architecture.md`；在交付中说明核验结果。
7. 开发完成后不得自动启动 LawStation、Ollama 或 TEI，不得遗留常驻进程。

只有代码、配置、迁移和测试能够证明的能力才标记为已实现。
