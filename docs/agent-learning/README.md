# Agent 开发学习笔记

面向中级后端工程师、Agent/前端知识较薄弱的学习材料，基于 LawStation 项目实际代码整理。每篇文档统一按以下结构组织：

1. 要解决的问题
2. 行业内一般怎么做
3. 核心机制原理
4. 本项目具体实现（函数级，含框架内部函数说明）
5. 对比：本项目 vs 行业常规方案
6. 本项目内部的关键设计取舍与易错点
7. 动手验证方式

## 目录

1. [SSE（Server-Sent Events）流式通信机制](01-sse.md)
2. [LangGraph Checkpoint 持久化机制](02-langgraph-checkpoint.md)
3. [分层记忆系统与 Context 工程](03-memory-context-engineering.md)
4. [LangGraph StateGraph 多 Agent 编排](04-langgraph-stategraph.md)
5. [Tool Calling / Function Calling 机制 + MCP 协议](05-tool-calling-mcp.md)

建议阅读顺序：4 → 5 → 2 → 3 → 1（先建立"什么是 Agent 编排"和"Agent 怎么用工具"的框架性认知，再看持久化和记忆这两块支撑性基础设施，最后看贯穿前后端的 SSE）。如果更想从已经熟悉的通信机制切入，也可以反过来从 1 开始。
