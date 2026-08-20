from dataclasses import dataclass
from uuid import uuid4

from fastapi import Header, HTTPException
from sqlalchemy import select

from backend.app.db.models import User
from backend.app.db.session import SessionLocal


@dataclass(frozen=True)
class RequestUserContext:
    tenant_id: str
    user_id: str
    request_id: str


def get_user_context(
    x_user_id: str = Header(..., alias="X-User-ID"),
) -> RequestUserContext:
    # This dependency is also used by StreamingResponse routes. Open and close
    # the lookup session here so a DB connection is never retained for the
    # lifetime of an SSE stream.
    with SessionLocal() as db:
        user = db.scalar(select(User).where(User.id == x_user_id))
    if user is None:
        raise HTTPException(status_code=401, detail="未知演示用户")
    return RequestUserContext(user.tenant_id, user.id, str(uuid4()))
