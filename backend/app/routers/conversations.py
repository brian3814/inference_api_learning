from fastapi import APIRouter, HTTPException

from ..schemas.chat import (
    ChatMessage,
    ConversationSummary,
    ConversationDetail,
    ConversationListResponse,
)
from ..services.memory import conversation_store

router = APIRouter(tags=["Conversations"])


@router.get("/v1/conversations")
async def list_conversations() -> ConversationListResponse:
    convs = conversation_store.list_all()
    return ConversationListResponse(
        data=[
            ConversationSummary(
                id=c.id,
                title=c.title,
                message_count=len(c.messages),
                created_at=c.created_at,
                updated_at=c.updated_at,
            )
            for c in convs
        ]
    )


@router.get("/v1/conversations/{conversation_id}")
async def get_conversation(conversation_id: str) -> ConversationDetail:
    conv = conversation_store.get(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationDetail(
        id=conv.id,
        title=conv.title,
        messages=[ChatMessage(**m) for m in conv.messages],
        created_at=conv.created_at,
        updated_at=conv.updated_at,
    )


@router.delete("/v1/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    if not conversation_store.delete(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "deleted", "id": conversation_id}
