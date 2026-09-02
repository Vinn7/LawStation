"""LangGraph 可持久化 State 与不可持久化 Invocation Context 的类型边界。"""

from dataclasses import dataclass, field
from typing import Any, TypedDict

from langchain_core.messages import BaseMessage

from backend.app.agent.schemas import (
    AgentError,
    CaseAnalysis,
    Citation,
    CounselDraft,
    EvidencePacket,
    ReviewResult,
)


class LegalConsultationState(TypedDict):
    """一次 AgentRun 的 Graph State；每个节点返回的同名字段会合并到这里。

    该对象可以被 AsyncSqliteSaver 保存，因此只能包含消息、Pydantic 模型和普通
    容器，不能放数据库 Session、HTTP Client、MCP Session 或 LangSmith Client。
    """

    # 输入上下文：业务消息与按预算生成的只读记忆快照。
    messages: list[BaseMessage]
    memory_context: str
    # 三个业务阶段和 Reviewer 的结构化产物。
    case_analysis: CaseAnalysis | None
    evidence_packet: EvidencePacket | None
    counsel_draft: CounselDraft | None
    review_result: ReviewResult | None
    # 路由循环计数：恢复后必须沿用，不能重新获得补检索/改稿额度。
    retry_count: int
    revision_count: int
    # Finalize 写入的用户可见结果。
    final_answer: str
    citations: list[Citation]
    errors: list[AgentError]
    # Analyst 建议经服务端验证后的最新事实覆盖。
    current_fact_overrides: list[dict[str, Any]]
    # 同步进 Checkpoint 的实际调用计数，保证故障恢复后上限不重置。
    model_call_count: int
    tool_call_count: int
    tool_trajectory: list[dict[str, Any]]


@dataclass(frozen=True)
class AgentInvocationIdentity:
    """服务端创建的不可变所有权身份；模型和工具不能自行指定这些字段。"""
    request_id: str
    tenant_id: str
    user_id: str
    conversation_id: str


@dataclass
class AgentInvocationMetrics:
    """单次 invocation 的可变运行计数，不在共享 Graph 实例之间复用。"""
    tool_call_count: int = 0
    model_call_count: int = 0
    tool_trajectory: list[dict[str, Any]] = field(default_factory=list)
    review_mode: str = "not_applicable"


@dataclass
class AgentInvocationContext:
    """通过 LangGraph context_schema 注入节点的请求级依赖。

    Context 不属于 LegalConsultationState，因而不会自动成为 Checkpoint 或下一轮
    对话上下文；恢复所需的计数必须由节点同步写回 State。
    """
    identity: AgentInvocationIdentity
    metrics: AgentInvocationMetrics = field(default_factory=AgentInvocationMetrics)
    langsmith_trace_id: str | None = None
    trace_config: dict[str, Any] | None = None
    evaluation_output: dict[str, Any] = field(default_factory=dict)
    persist_tool_audit: bool = True
    evaluation_case_analysis: dict[str, Any] | None = None

    @property
    def audit_fields(self) -> dict[str, Any]:
        return {
            "request_id": self.identity.request_id,
            "tenant_id": self.identity.tenant_id,
            "user_id": self.identity.user_id,
            "conversation_id": self.identity.conversation_id,
        }
