import { useState, useCallback } from 'react';
import ModelPanel from './components/ModelPanel';
import ChatPanel from './components/ChatPanel';
import ConversationList from './components/ConversationList';
import { getConversation, type ChatMessage } from './api';

function generateId() {
  return Math.random().toString(36).slice(2, 12);
}

export default function App() {
  const [modelLoaded, setModelLoaded] = useState(false);
  const [modelMultimodal, setModelMultimodal] = useState(false);
  const [conversationId, setConversationId] = useState<string | null>(
    () => generateId(),
  );
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [refreshKey, setRefreshKey] = useState(0);

  const handleStatusChange = useCallback((loaded: boolean, multimodal: boolean) => {
    setModelLoaded(loaded);
    setModelMultimodal(multimodal);
  }, []);

  const handleNewChat = useCallback(() => {
    setConversationId(generateId());
    setMessages([]);
    setRefreshKey((k) => k + 1);
  }, []);

  const handleSelectConversation = useCallback(async (id: string) => {
    try {
      const conv = await getConversation(id);
      setConversationId(id);
      setMessages(conv.messages);
    } catch {
      // ignore
    }
  }, []);

  const handleMessagesChange = useCallback(
    (next: ChatMessage[] | ((prev: ChatMessage[]) => ChatMessage[])) => {
      if (typeof next === 'function') {
        setMessages((prev) => {
          const result = next(prev);
          return result;
        });
      } else {
        setMessages(next);
      }
      // Refresh conversation list after each exchange to pick up new titles
      setRefreshKey((k) => k + 1);
    },
    [],
  );

  return (
    <div className="app-layout">
      <aside className="sidebar">
        <h2>Conversations</h2>
        <ConversationList
          activeId={conversationId}
          onSelect={handleSelectConversation}
          onNewChat={handleNewChat}
          refreshKey={refreshKey}
        />
      </aside>
      <div className="app">
        <header>
          <h1>Inference API</h1>
        </header>
        <ModelPanel onStatusChange={handleStatusChange} />
        <ChatPanel
          modelLoaded={modelLoaded}
          modelMultimodal={modelMultimodal}
          conversationId={conversationId}
          messages={messages}
          onMessagesChange={handleMessagesChange}
        />
      </div>
    </div>
  );
}
