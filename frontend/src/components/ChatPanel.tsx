import { useState, useRef, useEffect } from 'react';
import { streamChat, type ChatMessage, type ToolActivity } from '../api';

interface ChatPanelProps {
  modelLoaded: boolean;
}

export default function ChatPanel({ modelLoaded }: ChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [streaming, setStreaming] = useState(false);
  const [toolsEnabled, setToolsEnabled] = useState(false);
  const [toolActivity, setToolActivity] = useState<ToolActivity[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, toolActivity]);

  const handleSend = async () => {
    const text = input.trim();
    if (!text || streaming || !modelLoaded) return;

    const userMsg: ChatMessage = { role: 'user', content: text };
    const updated = [...messages, userMsg];
    setMessages(updated);
    setInput('');
    setStreaming(true);
    setToolActivity([]);

    const assistantMsg: ChatMessage = { role: 'assistant', content: '' };
    setMessages([...updated, assistantMsg]);

    try {
      await streamChat({
        messages: updated,
        onChunk: (chunk) => {
          assistantMsg.content += chunk;
          setMessages((prev) => {
            const copy = [...prev];
            copy[copy.length - 1] = { ...assistantMsg };
            return copy;
          });
        },
        onDone: () => setStreaming(false),
        toolsEnabled,
        onToolActivity: (activity) => {
          setToolActivity((prev) => [...prev, activity]);
        },
      });
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
            {msg.role === 'assistant' && i === messages.length - 1 && toolActivity.length > 0 && (
              <div className="tool-activity-list">
                {toolActivity.map((ta, j) => (
                  <details key={j} className="tool-activity">
                    <summary>
                      <span className="tool-icon">&#128295;</span>
                      <span className="tool-name">{ta.name}</span>
                      <span className="tool-status">
                        {ta.type === 'tool_call' ? 'Calling...' : 'Done'}
                      </span>
                    </summary>
                    <div className="tool-detail">
                      {ta.type === 'tool_call' && ta.arguments && (
                        <pre className="tool-arguments">
                          {typeof ta.arguments === 'object'
                            ? JSON.stringify(ta.arguments, null, 2)
                            : String(ta.arguments)}
                        </pre>
                      )}
                      {ta.type === 'tool_result' && ta.content && (
                        <pre className="tool-result">{ta.content}</pre>
                      )}
                    </div>
                  </details>
                ))}
              </div>
            )}
            <div className="message-bubble">{msg.content || '...'}</div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      <div className="chat-input-row">
        <label className="tools-toggle">
          <input
            type="checkbox"
            checked={toolsEnabled}
            onChange={(e) => setToolsEnabled(e.target.checked)}
            disabled={streaming}
          />
          <span className="tools-toggle-label">Tools</span>
        </label>
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
