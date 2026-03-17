import time
import uuid

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from ..schemas.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    ChatMessage,
    Choice,
    ChoiceDelta,
    Usage,
)
from ..services.model_manager import model_manager
from ..services.generation import GenerationService

router = APIRouter(tags=["Chat"])
generation_service = GenerationService(model_manager)


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not model_manager.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="No model loaded. Use /v1/models/load to load a model first.",
        )

    model_id = request.model or model_manager.current_model_id

    if request.stream:
        return EventSourceResponse(
            stream_response(request, model_id),
            media_type="text/event-stream",
        )
    else:
        return await generate_response(request, model_id)


async def generate_response(
    request: ChatCompletionRequest,
    model_id: str,
) -> ChatCompletionResponse:
    try:
        generated_text, prompt_tokens, completion_tokens = await generation_service.generate(
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=False,
        )

        return ChatCompletionResponse(
            model=model_id,
            choices=[
                Choice(
                    index=0,
                    message=ChatMessage(role="assistant", content=generated_text),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def stream_response(request: ChatCompletionRequest, model_id: str):
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    try:
        generator = await generation_service.generate(
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=True,
        )

        async for text in generator:
            chunk = ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=model_id,
                choices=[
                    ChoiceDelta(
                        index=0,
                        delta={"content": text},
                    )
                ],
            )
            yield {"data": chunk.model_dump_json()}

        final_chunk = ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=model_id,
            choices=[
                ChoiceDelta(
                    index=0,
                    delta={},
                    finish_reason="stop",
                )
            ],
        )
        yield {"data": final_chunk.model_dump_json()}
        yield {"data": "[DONE]"}

    except Exception as e:
        yield {"data": f'{{"error": "{str(e)}"}}'}
