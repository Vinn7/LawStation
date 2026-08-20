from dataclasses import dataclass
from uuid import uuid4

from fastapi import Depends, Header, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models import User
from backend.app.db.session import get_db


@dataclass(frozen=True)
class RequestUserContext:
    tenant_id: str
    user_id: str
    request_id: str


def get_user_context(
    x_user_id: str = Header(..., alias="X-User-ID"), db: Session = Depends(get_db)
) -> RequestUserContext:
    user = db.scalar(select(User).where(User.id == x_user_id))
    if user is None:
        raise HTTPException(status_code=401, detail="未知演示用户")
    return RequestUserContext(user.tenant_id, user.id, str(uuid4()))

