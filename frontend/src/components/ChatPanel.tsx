import { useState, useRef, useEffect, useCallback } from 'react';
import {
  streamChat,
  type ChatMessage,
  type ContentPart,
  type ToolActivity,
} from '../api';

interface ChatPanelProps {
  modelLoaded: boolean;
  modelMultimodal: boolean;
  conversationId: string | null;
  messages: ChatMessage[];
  onMessagesChange: (messages: ChatMessage[]) => void;
}

export default function ChatPanel({
  modelLoaded,
  modelMultimodal,
  conversationId,
  messages,
  onMessagesChange,
}: ChatPanelProps) {
  const [input, setInput] = useState('');
  const [streaming, setStreaming] = useState(false);
  const [toolsEnabled, setToolsEnabled] = useState(false);
  const [toolActivity, setToolActivity] = useState<ToolActivity[]>([]);
  const [pendingImages, setPendingImages] = useState<string[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, toolActivity]);

  const addImageFiles = useCallback((files: File[]) => {
    files.forEach((file) => {
      if (!file.type.startsWith('image/')) return;
      const reader = new FileReader();
      reader.onload = () => {
        if (typeof reader.result === 'string') {
          setPendingImages((prev) => [...prev, reader.result as string]);
        }
      };
      reader.readAsDataURL(file);
    });
  }, []);

  const handleImageSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files) addImageFiles(Array.from(e.target.files));
      e.target.value = '';
    },
    [addImageFiles],
  );

  const handlePaste = useCallback(
    (e: React.ClipboardEvent) => {
      if (!modelMultimodal) return;
      const imageFiles = Array.from(e.clipboardData.items)
        .filter((item) => item.type.startsWith('image/'))
        .map((item) => item.getAsFile())
        .filter((f): f is File => f !== null);
      if (imageFiles.length > 0) {
        addImageFiles(imageFiles);
      }
    },
    [modelMultimodal, addImageFiles],
  );

  const handleSend = async () => {
    const text = input.trim();
    if ((!text && pendingImages.length === 0) || streaming || !modelLoaded) return;

    let content: ChatMessage['content'];
    if (pendingImages.length > 0) {
      const parts: ContentPart[] = [
        ...pendingImages.map((url) => ({
          type: 'image_url' as const,
          image_url: { url },
        })),
        ...(text ? [{ type: 'text' as const, text }] : []),
      ];
      content = parts;
    } else {
      content = text;
    }

    const userMsg: ChatMessage = { role: 'user', content };
    const updated = [...messages, userMsg];
    onMessagesChange(updated);
    setInput('');
    setPendingImages([]);
    setStreaming(true);
    setToolActivity([]);

    const assistantMsg: ChatMessage = { role: 'assistant', content: '' };
    onMessagesChange([...updated, assistantMsg]);

    // When using a conversation, only send the new user message
    const toSend = conversationId ? [userMsg] : updated;

    try {
      await streamChat({
        messages: toSend,
        onChunk: (chunk) => {
          assistantMsg.content += chunk;
          onMessagesChange((prev) => {
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
        conversationId: conversationId ?? undefined,
      });
    } catch {
      assistantMsg.content += '\n[Error: stream failed]';
      onMessagesChange((prev) => {
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
            <div className="message-bubble">
              {typeof msg.content === 'string' ? (
                msg.content || '...'
              ) : (
                msg.content.map((part, pi) =>
                  part.type === 'image_url' ? (
                    <img
                      key={pi}
                      src={part.image_url.url}
                      alt="uploaded"
                      className="message-image"
                    />
                  ) : (
                    <span key={pi}>{part.text}</span>
                  ),
                )
              )}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      {pendingImages.length > 0 && (
        <div className="pending-images">
          {pendingImages.map((url, i) => (
            <div key={i} className="pending-image-thumb">
              <img src={url} alt="pending" />
              <button
                className="remove-image"
                onClick={() => setPendingImages((prev) => prev.filter((_, j) => j !== i))}
              >
                &times;
              </button>
            </div>
          ))}
        </div>
      )}
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
        {modelMultimodal && (
          <>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              multiple
              onChange={handleImageSelect}
              style={{ display: 'none' }}
            />
            <button
              className="image-upload-btn"
              onClick={() => fileInputRef.current?.click()}
              disabled={!modelLoaded || streaming}
              title="Upload image"
            >
              &#128247;
            </button>
          </>
        )}
        <textarea
          placeholder={modelLoaded ? 'Type a message...' : 'Load a model first'}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onPaste={handlePaste}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              handleSend();
            }
          }}
          disabled={!modelLoaded || streaming}
          rows={1}
        />
        <button
          onClick={handleSend}
          disabled={!modelLoaded || streaming || (!input.trim() && pendingImages.length === 0)}
        >
          Send
        </button>
      </div>
    </div>
  );
}
