from pydantic import BaseModel, Field
from typing import Annotated, Literal, Optional, Union
import time
import uuid


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict


class ToolDefinition(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class ImageUrl(BaseModel):
    url: str


class TextContentPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageContentPart(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: ImageUrl


ContentPart = Annotated[
    Union[TextContentPart, ImageContentPart], Field(discriminator="type")
]
MessageContent = Union[str, list[ContentPart]]


def extract_text(content: MessageContent) -> str:
    """Extract the text portion from any content format."""
    if isinstance(content, str):
        return content
    return " ".join(p.text for p in content if isinstance(p, TextContentPart))


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: MessageContent
    tool_calls: Optional[list[ToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    max_tokens: Optional[int] = None
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    tools_enabled: bool = False
    conversation_id: Optional[str] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "tool_calls"]


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Usage
    conversation_id: Optional[str] = None


class ChoiceDelta(BaseModel):
    index: int
    delta: dict
    finish_reason: Optional[Literal["stop", "length", "tool_calls"]] = None


class ChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChoiceDelta]
    conversation_id: Optional[str] = None


class ConversationSummary(BaseModel):
    id: str
    title: Optional[str] = None
    message_count: int
    created_at: float
    updated_at: float


class ConversationDetail(BaseModel):
    id: str
    title: Optional[str] = None
    messages: list[ChatMessage]
    created_at: float
    updated_at: float


class ConversationListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ConversationSummary]
