import { useState, useEffect } from 'react';
import {
  listConversations,
  deleteConversation,
  type ConversationSummary,
} from '../api';

interface ConversationListProps {
  activeId: string | null;
  onSelect: (id: string) => void;
  onNewChat: () => void;
  refreshKey: number;
}

export default function ConversationList({
  activeId,
  onSelect,
  onNewChat,
  refreshKey,
}: ConversationListProps) {
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);

  useEffect(() => {
    listConversations().then(setConversations).catch(() => {});
  }, [refreshKey]);

  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    try {
      await deleteConversation(id);
      setConversations((prev) => prev.filter((c) => c.id !== id));
      if (activeId === id) onNewChat();
    } catch {
      // ignore
    }
  };

  return (
    <div className="conversation-list">
      <button className="new-chat-btn" onClick={onNewChat}>
        + New Chat
      </button>
      {conversations.length === 0 && (
        <div className="conv-empty">No conversations yet</div>
      )}
      {conversations.map((conv) => (
        <div
          key={conv.id}
          className={`conv-item ${conv.id === activeId ? 'active' : ''}`}
          onClick={() => onSelect(conv.id)}
        >
          <div className="conv-title">
            {conv.title || 'Untitled'}
          </div>
          <div className="conv-meta">
            {conv.message_count} messages
          </div>
          <button
            className="conv-delete"
            onClick={(e) => handleDelete(e, conv.id)}
            title="Delete conversation"
          >
            &times;
          </button>
        </div>
      ))}
    </div>
  );
}
