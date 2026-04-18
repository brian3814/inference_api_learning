import time
import uuid
import threading
from dataclasses import dataclass, field


@dataclass
class Conversation:
    id: str
    messages: list[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    title: str | None = None


class ConversationStore:
    def __init__(self):
        self._conversations: dict[str, Conversation] = {}
        self._lock = threading.Lock()

    def create(self, conversation_id: str | None = None) -> Conversation:
        conv = Conversation(id=conversation_id or uuid.uuid4().hex[:16])
        with self._lock:
            self._conversations[conv.id] = conv
        return conv

    def get(self, conversation_id: str) -> Conversation | None:
        with self._lock:
            return self._conversations.get(conversation_id)

    def list_all(self) -> list[Conversation]:
        with self._lock:
            convs = list(self._conversations.values())
        return sorted(convs, key=lambda c: c.updated_at, reverse=True)

    def append_messages(self, conversation_id: str, messages: list[dict]) -> None:
        with self._lock:
            conv = self._conversations.get(conversation_id)
            if conv is None:
                return
            conv.messages.extend(messages)
            conv.updated_at = time.time()
            if conv.title is None:
                for m in conv.messages:
                    if m.get("role") == "user" and m.get("content"):
                        content = m["content"]
                        if isinstance(content, list):
                            content = " ".join(
                                p.get("text", "") for p in content if p.get("type") == "text"
                            )
                        conv.title = (content or "")[:80] or None
                        break

    def delete(self, conversation_id: str) -> bool:
        with self._lock:
            return self._conversations.pop(conversation_id, None) is not None

    def __len__(self) -> int:
        return len(self._conversations)


conversation_store = ConversationStore()
