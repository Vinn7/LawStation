"""API 路由包的对外门面：组合子路由 + 重新导出测试依赖的符号，不放任何逻辑。"""

from fastapi import APIRouter

from . import agent_runs, chat, conversations, feedback, index, memories, scenarios
from .feedback import message_feedback
from .helpers import with_sse_heartbeat

router = APIRouter()
for _module in (index, scenarios, conversations, memories, agent_runs, feedback, chat):
    router.include_router(_module.router)
del _module

__all__ = ["router", "message_feedback", "with_sse_heartbeat"]
