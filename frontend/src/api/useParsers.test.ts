// @vitest-environment jsdom
import { renderHook, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { useParsers } from './useParsers'

const PARSERS = [
  { name: 'opencv', available: true, default: true, description: 'Classical.' },
  { name: 'cubicasa', available: false, default: false, description: 'Needs setup.' },
]

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('useParsers', () => {
  it('lists the parsers the backend offers', async () => {
    const fetchMock = vi.fn().mockResolvedValue(Response.json(PARSERS))
    vi.stubGlobal('fetch', fetchMock)
    const { result } = renderHook(() => useParsers())
    expect(result.current).toEqual([])
    await waitFor(() => expect(result.current).toEqual(PARSERS))
    expect(fetchMock.mock.calls[0][0]).toBe('/api/parsers')
  })

  it('offers nothing when the backend cannot say', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => {})
    const fetchMock = vi.fn().mockResolvedValue(Response.json({ unexpected: true }))
    vi.stubGlobal('fetch', fetchMock)
    const { result } = renderHook(() => useParsers())
    await waitFor(() => expect(fetchMock).toHaveBeenCalled())
    await waitFor(() => expect(console.warn).toHaveBeenCalled())
    expect(result.current).toEqual([])
  })
})
