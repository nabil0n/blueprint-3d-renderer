// @vitest-environment jsdom
import { renderHook, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { useBackendStatus } from './useBackendStatus'

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('useBackendStatus', () => {
  it('is online when the health check succeeds', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{}', { status: 200 })))
    const { result } = renderHook(() => useBackendStatus())
    expect(result.current).toBe('checking')
    await waitFor(() => expect(result.current).toBe('online'))
  })

  it('is offline on an error status', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('', { status: 502 })))
    const { result } = renderHook(() => useBackendStatus())
    await waitFor(() => expect(result.current).toBe('offline'))
  })

  it('is offline when the request fails', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => {})
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('fetch failed')))
    const { result } = renderHook(() => useBackendStatus())
    await waitFor(() => expect(result.current).toBe('offline'))
  })
})
