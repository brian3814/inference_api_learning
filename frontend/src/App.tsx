import { useState, useCallback } from 'react';
import ModelPanel from './components/ModelPanel';
import ChatPanel from './components/ChatPanel';

export default function App() {
  const [modelLoaded, setModelLoaded] = useState(false);

  const handleStatusChange = useCallback((loaded: boolean) => {
    setModelLoaded(loaded);
  }, []);

  return (
    <div className="app">
      <header>
        <h1>Inference API</h1>
      </header>
      <ModelPanel onStatusChange={handleStatusChange} />
      <ChatPanel modelLoaded={modelLoaded} />
    </div>
  );
}
