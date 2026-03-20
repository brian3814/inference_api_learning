export interface ModelStatus {
  loaded: boolean;
  model_id: string | null;
  device: string;
}

export async function loadModel(modelId: string): Promise<Record<string, unknown>> {
  const res = await fetch('/v1/models/load', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_id: modelId }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Failed to load model');
  }
  return res.json();
}

export async function getModelStatus(): Promise<ModelStatus> {
  const res = await fetch('/v1/models/status');
  if (!res.ok) throw new Error('Failed to get model status');
  return res.json();
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface ToolActivity {
  type: 'tool_call' | 'tool_result';
  name: string;
  arguments?: Record<string, unknown>;
  content?: string;
}

export interface StreamChatOptions {
  messages: ChatMessage[];
  onChunk: (text: string) => void;
  onDone: () => void;
  toolsEnabled?: boolean;
  onToolActivity?: (activity: ToolActivity) => void;
}

export async function streamChat({
  messages,
  onChunk,
  onDone,
  toolsEnabled = false,
  onToolActivity,
}: StreamChatOptions) {
  const res = await fetch('/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      messages,
      stream: true,
      ...(toolsEnabled && { tools_enabled: true }),
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Chat request failed');
  }

  const reader = res.body?.getReader();
  if (!reader) throw new Error('No response body');

  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || !trimmed.startsWith('data:')) continue;

      const data = trimmed.slice(5).trim();
      if (data === '[DONE]') {
        onDone();
        return;
      }

      try {
        const parsed = JSON.parse(data);
        const delta = parsed.choices?.[0]?.delta;
        if (!delta) continue;

        if (delta.tool_activity && onToolActivity) {
          onToolActivity(delta.tool_activity as ToolActivity);
        }

        if (delta.content) {
          onChunk(delta.content);
        }
      } catch {
        // skip unparseable lines
      }
    }
  }

  onDone();
}

export async function getAvailableTools(): Promise<{ tools: Record<string, unknown>[] }> {
  const res = await fetch('/v1/tools');
  if (!res.ok) throw new Error('Failed to get tools');
  return res.json();
}
