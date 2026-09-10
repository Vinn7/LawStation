from fastapi import APIRouter

from mcp_servers.law_rag.server import get_index_status

router = APIRouter(prefix="/api")


@router.get("/index/status")
def index_status():
    return get_index_status()
