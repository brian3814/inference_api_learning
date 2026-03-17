import { useState, useRef, useEffect } from 'react';
import { streamChat, type ChatMessage } from '../api';

interface ChatPanelProps {
  modelLoaded: boolean;
}

export default function ChatPanel({ modelLoaded }: ChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [streaming, setStreaming] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    const text = input.trim();
    if (!text || streaming || !modelLoaded) return;

    const userMsg: ChatMessage = { role: 'user', content: text };
    const updated = [...messages, userMsg];
    setMessages(updated);
    setInput('');
    setStreaming(true);

    const assistantMsg: ChatMessage = { role: 'assistant', content: '' };
    setMessages([...updated, assistantMsg]);

    try {
      await streamChat(
        updated,
        (chunk) => {
          assistantMsg.content += chunk;
          setMessages((prev) => {
            const copy = [...prev];
            copy[copy.length - 1] = { ...assistantMsg };
            return copy;
          });
        },
        () => setStreaming(false),
      );
    } catch {
      assistantMsg.content += '\n[Error: stream failed]';
      setMessages((prev) => {
        const copy = [...prev];
        copy[copy.length - 1] = { ...assistantMsg };
        return copy;
      });
      setStreaming(false);
    }
  };

  return (
    <div className="chat-panel">
      <div className="messages">
        {messages.length === 0 && (
          <div className="empty-state">
            {modelLoaded
              ? 'Send a message to start chatting'
              : 'Load a model to begin'}
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <div className="message-bubble">{msg.content || '...'}</div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      <div className="chat-input-row">
        <textarea
          placeholder={modelLoaded ? 'Type a message...' : 'Load a model first'}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              handleSend();
            }
          }}
          disabled={!modelLoaded || streaming}
          rows={1}
        />
        <button onClick={handleSend} disabled={!modelLoaded || streaming || !input.trim()}>
          Send
        </button>
      </div>
    </div>
  );
}
