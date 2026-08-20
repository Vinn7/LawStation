import asyncio
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass

from backend.app.core.config import Settings, get_settings


class ConversationBusyError(RuntimeError):
    pass


class AgentQueueTimeoutError(RuntimeError):
    pass


@dataclass(frozen=True)
class ConcurrencyIdentity:
    request_id: str
    tenant_id: str
    user_id: str
    conversation_id: str

    @property
    def user_key(self) -> tuple[str, str]:
        return self.tenant_id, self.user_id

    @property
    def conversation_key(self) -> tuple[str, str, str]:
        return self.tenant_id, self.user_id, self.conversation_id


class AgentConcurrencyManager:
    """Process-local admission control without head-of-line global blocking."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if self.settings.agent_per_conversation_concurrency != 1:
            raise ValueError("AGENT_PER_CONVERSATION_CONCURRENCY 首版必须为 1")
        self._condition = asyncio.Condition()
        self._reserved_conversations: set[tuple[str, str, str]] = set()
        self._active_global = 0
        self._active_by_user: dict[tuple[str, str], int] = defaultdict(int)
        self._active_requests: set[str] = set()

    async def reserve(self, identity: ConcurrencyIdentity) -> None:
        async with self._condition:
            if identity.conversation_key in self._reserved_conversations:
                raise ConversationBusyError("该会话正在生成回答，请等待完成或先停止生成。")
            self._reserved_conversations.add(identity.conversation_key)

    async def release_reservation(self, identity: ConcurrencyIdentity) -> None:
        async with self._condition:
            self._reserved_conversations.discard(identity.conversation_key)
            self._condition.notify_all()

    async def would_queue(self, identity: ConcurrencyIdentity) -> bool:
        async with self._condition:
            return not self._has_capacity(identity)

    async def acquire(self, identity: ConcurrencyIdentity) -> None:
        deadline = time.monotonic() + self.settings.agent_queue_timeout_seconds
        async with self._condition:
            while not self._has_capacity(identity):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AgentQueueTimeoutError("系统当前咨询任务较多，请稍后重试。")
                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=remaining)
                except TimeoutError as exc:
                    raise AgentQueueTimeoutError("系统当前咨询任务较多，请稍后重试。") from exc
            self._active_global += 1
            self._active_by_user[identity.user_key] += 1
            self._active_requests.add(identity.request_id)

    async def release(self, identity: ConcurrencyIdentity) -> None:
        async with self._condition:
            if identity.request_id in self._active_requests:
                self._active_requests.remove(identity.request_id)
                self._active_by_user[identity.user_key] -= 1
                if self._active_by_user[identity.user_key] == 0:
                    del self._active_by_user[identity.user_key]
                self._active_global -= 1
            self._condition.notify_all()

    @asynccontextmanager
    async def slot(self, identity: ConcurrencyIdentity):
        acquired = False
        try:
            await self.acquire(identity)
            acquired = True
            yield
        finally:
            if acquired:
                await self.release(identity)

    def _has_capacity(self, identity: ConcurrencyIdentity) -> bool:
        return (
            self._active_global < self.settings.agent_global_concurrency
            and self._active_by_user[identity.user_key]
            < self.settings.agent_per_user_concurrency
        )
