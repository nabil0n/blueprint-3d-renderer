import { useEffect, useState } from 'react'

export type BackendStatus = 'checking' | 'online' | 'offline'

/** Pings the backend once on mount. `/api` is proxied to the backend by the Vite dev server. */
export function useBackendStatus(): BackendStatus {
  const [status, setStatus] = useState<BackendStatus>('checking')

  useEffect(() => {
    const controller = new AbortController()
    fetch('/api/health', { signal: controller.signal })
      .then((res) => setStatus(res.ok ? 'online' : 'offline'))
      .catch((err: unknown) => {
        if (!controller.signal.aborted) {
          console.warn('Backend health check failed:', err)
          setStatus('offline')
        }
      })
    return () => controller.abort()
  }, [])

  return status
}
