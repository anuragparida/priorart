import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'node:path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    host: '0.0.0.0',
    // PHASE-1.9 spec calls for :15173. Port 15173 is currently bound by
    // the clausecraft dev container (PID 2132774, since 2026-06-16).
    // Per Apollo's standing-permission rule (2026-06-28 12:00 audit),
    // run on :15174 with this documented deviation.
    port: 15174,
    strictPort: true,
    proxy: {
      // Proxy the FastAPI backend so the frontend can use a relative
      // URL in dev (avoids CORS preflight round-trips for the LLM
      // call). Production builds hit the API directly.
      '/api': {
        target: 'http://localhost:18001',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})