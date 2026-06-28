import { useEffect } from 'react';

import { IdeaAnalyzer } from '@/components/IdeaAnalyzer';

function App() {
  // Apply dark mode on mount. shadcn's default token set is dark-first
  // (per Phase 1.9 spec). Toggle off the class to test light mode.
  useEffect(() => {
    document.documentElement.classList.add('dark');
  }, []);

  return (
    <div className="min-h-screen bg-background text-foreground antialiased">
      <IdeaAnalyzer />
    </div>
  );
}

export default App;