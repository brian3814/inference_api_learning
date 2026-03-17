import { useState, useEffect, useCallback } from 'react';
import { loadModel, getModelStatus, type ModelStatus } from '../api';

interface ModelPanelProps {
  onStatusChange: (loaded: boolean) => void;
}

export default function ModelPanel({ onStatusChange }: ModelPanelProps) {
  const [modelId, setModelId] = useState('');
  const [status, setStatus] = useState<ModelStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const fetchStatus = useCallback(async () => {
    try {
      const s = await getModelStatus();
      setStatus(s);
      onStatusChange(s.loaded);
    } catch {
      // backend might not be running yet
    }
  }, [onStatusChange]);

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  const handleLoad = async () => {
    if (!modelId.trim()) return;
    setLoading(true);
    setError('');
    try {
      await loadModel(modelId.trim());
      await fetchStatus();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load model');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="model-panel">
      <div className="model-input-row">
        <input
          type="text"
          placeholder="Model ID (e.g. microsoft/DialoGPT-medium)"
          value={modelId}
          onChange={(e) => setModelId(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleLoad()}
          disabled={loading}
        />
        <button onClick={handleLoad} disabled={loading || !modelId.trim()}>
          {loading ? 'Loading...' : 'Load Model'}
        </button>
      </div>
      {error && <div className="error">{error}</div>}
      {status && (
        <div className="model-status">
          {status.loaded ? (
            <span className="status-loaded">
              {status.model_id} — {status.device}
            </span>
          ) : (
            <span className="status-none">No model loaded</span>
          )}
        </div>
      )}
    </div>
  );
}
