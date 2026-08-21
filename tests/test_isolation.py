from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.core.context import RequestUserContext
from backend.app.db.models import Conversation, Tenant, User, UserMemory
from backend.app.db.session import Base
from backend.app.services.repositories import OwnedRepository


def test_memory_isolation():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    tenant = Tenant(id="t", name="t")
    user_a = User(id="a", tenant_id="t", name="a")
    user_b = User(id="b", tenant_id="t", name="b")
    conversation_a = Conversation(id="ca", tenant_id="t", user_id="a")
    conversation_b = Conversation(id="cb", tenant_id="t", user_id="b")
    memory = UserMemory(
        id="secret",
        tenant_id="t",
        user_id="b",
        conversation_id="cb",
        content="B secret",
    )
    db.add_all([tenant, user_a, user_b, conversation_a, conversation_b, memory])
    db.commit()

    repository_a = OwnedRepository(db, RequestUserContext("t", "a", "r"))
    assert repository_a.memories() == []
    assert repository_a.update_memory("secret", "stolen", 1) == "missing"
    assert repository_a.set_memory_status("secret", 1, "active") == "missing"
    assert not repository_a.delete_memory("secret")
