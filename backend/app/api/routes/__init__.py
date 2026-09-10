from fastapi import APIRouter

from . import agent_runs, chat, conversations, feedback, index, memories, scenarios
from .feedback import message_feedback
from .helpers import with_sse_heartbeat
from .scenarios import _delete_scenario_conversation, _scenario_catalog

router = APIRouter()
for _module in (index, scenarios, conversations, memories, agent_runs, feedback, chat):
    router.include_router(_module.router)

__all__ = ["router"]
