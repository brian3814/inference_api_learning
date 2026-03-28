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
from ..services.agent import AgentService
from ..services.memory import conversation_store
from ..tools import tool_registry

router = APIRouter(tags=["Chat"])
generation_service = GenerationService(model_manager)
agent_service = AgentService(generation_service)


def _resolve_conversation(request: ChatCompletionRequest) -> tuple[str | None, list[ChatMessage]]:
    """Load or create a conversation and return (conv_id, effective_messages)."""
    conv_id = request.conversation_id
    if conv_id is None:
        return None, request.messages

    conv = conversation_store.get(conv_id)
    if conv is None:
        conv = conversation_store.create(conversation_id=conv_id)

    stored = [ChatMessage(**m) for m in conv.messages]
    return conv_id, stored + request.messages


def _save_to_conversation(
    conv_id: str | None,
    new_messages: list[ChatMessage],
    assistant_content: str,
) -> None:
    """Append the new user messages and assistant reply to the conversation store."""
    if conv_id is None:
        return
    to_save = [m.model_dump() for m in new_messages]
    to_save.append({"role": "assistant", "content": assistant_content})
    conversation_store.append_messages(conv_id, to_save)


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not model_manager.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="No model loaded. Use /v1/models/load to load a model first.",
        )

    model_id = request.model or model_manager.current_model_id
    conv_id, effective_messages = _resolve_conversation(request)
    use_agent = request.tools_enabled and len(tool_registry) > 0

    if use_agent:
        if request.stream:
            return EventSourceResponse(
                stream_agent_response(effective_messages, request, model_id, conv_id),
                media_type="text/event-stream",
            )
        else:
            return await generate_agent_response(effective_messages, request, model_id, conv_id)

    if request.stream:
        return EventSourceResponse(
            stream_response(effective_messages, request, model_id, conv_id),
            media_type="text/event-stream",
        )
    else:
        return await generate_response(effective_messages, request, model_id, conv_id)


async def generate_response(
    effective_messages: list[ChatMessage],
    request: ChatCompletionRequest,
    model_id: str,
    conv_id: str | None,
) -> ChatCompletionResponse:
    try:
        generated_text, prompt_tokens, completion_tokens = await generation_service.generate(
            messages=effective_messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=False,
        )

        _save_to_conversation(conv_id, request.messages, generated_text)

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
            conversation_id=conv_id,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def generate_agent_response(
    effective_messages: list[ChatMessage],
    request: ChatCompletionRequest,
    model_id: str,
    conv_id: str | None,
) -> ChatCompletionResponse:
    try:
        final_text, tool_history, prompt_tokens, completion_tokens = await agent_service.run(
            messages=effective_messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        _save_to_conversation(conv_id, request.messages, final_text)

        return ChatCompletionResponse(
            model=model_id,
            choices=[
                Choice(
                    index=0,
                    message=ChatMessage(role="assistant", content=final_text),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            conversation_id=conv_id,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def stream_response(
    effective_messages: list[ChatMessage],
    request: ChatCompletionRequest,
    model_id: str,
    conv_id: str | None,
):
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    collected: list[str] = []

    try:
        generator = await generation_service.generate(
            messages=effective_messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=True,
        )

        async for text in generator:
            collected.append(text)
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
                conversation_id=conv_id,
            )
            yield {"data": chunk.model_dump_json()}

        _save_to_conversation(conv_id, request.messages, "".join(collected))

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
            conversation_id=conv_id,
        )
        yield {"data": final_chunk.model_dump_json()}
        yield {"data": "[DONE]"}

    except Exception as e:
        yield {"data": f'{{"error": "{str(e)}"}}'}


async def stream_agent_response(
    effective_messages: list[ChatMessage],
    request: ChatCompletionRequest,
    model_id: str,
    conv_id: str | None,
):
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    collected: list[str] = []

    try:
        async for event in agent_service.run_streaming(
            messages=effective_messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        ):
            if event["type"] == "tool_activity":
                chunk = ChatCompletionChunk(
                    id=chunk_id,
                    created=created,
                    model=model_id,
                    choices=[
                        ChoiceDelta(
                            index=0,
                            delta={"tool_activity": event["data"]},
                        )
                    ],
                    conversation_id=conv_id,
                )
                yield {"data": chunk.model_dump_json()}

            elif event["type"] == "content_delta":
                collected.append(event["data"])
                chunk = ChatCompletionChunk(
                    id=chunk_id,
                    created=created,
                    model=model_id,
                    choices=[
                        ChoiceDelta(
                            index=0,
                            delta={"content": event["data"]},
                        )
                    ],
                    conversation_id=conv_id,
                )
                yield {"data": chunk.model_dump_json()}

            elif event["type"] == "done":
                _save_to_conversation(conv_id, request.messages, "".join(collected))

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
                    conversation_id=conv_id,
                )
                yield {"data": final_chunk.model_dump_json()}
                yield {"data": "[DONE]"}

    except Exception as e:
        yield {"data": f'{{"error": "{str(e)}"}}'}
