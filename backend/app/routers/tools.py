from fastapi import APIRouter

from ..tools import tool_registry

router = APIRouter(tags=["Tools"])


@router.get("/v1/tools")
async def list_tools():
    return {
        "tools": tool_registry.list_definitions(),
    }
