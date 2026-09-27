import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

const API_URL = process.env.VITE_API_PROXY_TARGET ?? 'http://localhost:8000'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    // The sample plan lives in the repo-level tests/fixtures, shared with the Python tests.
    fs: { allow: ['..'] },
    proxy: { '/api': API_URL },
    // Bind mounts in Docker on Windows/macOS do not deliver file events.
    watch: process.env.VITE_USE_POLLING === 'true' ? { usePolling: true, interval: 300 } : undefined,
  },
  test: {
    coverage: { include: ['src/**/*.ts'], exclude: ['src/**/*.test.ts'] },
  },
})
