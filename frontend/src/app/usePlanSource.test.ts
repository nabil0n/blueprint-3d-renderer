// @vitest-environment jsdom
import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import sample from '../../../tests/fixtures/two_room_apartment.json'
import * as api from '../api/parseFloorPlan'
import { parsePlan } from '../plan/schema'
import { usePlanSource, type LoadedPlan } from './usePlanSource'

const INITIAL: LoadedPlan = { plan: parsePlan({}), source: 'initial', version: 0 }
const meta: api.ParseMeta = {
  parser: 'opencv',
  cm_per_px: 1.5,
  scale_source: 'doors',
  scale_detail: '',
  image_width: 10,
  image_height: 10,
  origin_px: [0, 0],
  warnings: [],
}
const png = new File([new Uint8Array(4)], 'plan.png', { type: 'image/png' })

beforeEach(() => {
  URL.createObjectURL = vi.fn(() => 'blob:preview')
  URL.revokeObjectURL = vi.fn()
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('usePlanSource', () => {
  it('uploads an image and shows the parsed plan', async () => {
    vi.spyOn(api, 'parseFloorPlan').mockResolvedValue({ plan: parsePlan(sample), meta })
    const { result } = renderHook(() => usePlanSource(INITIAL))

    act(() => result.current.uploadImage(png))
    expect(result.current.busy).toBe(true)
    await waitFor(() => expect(result.current.busy).toBe(false))

    expect(result.current.loaded).toMatchObject({ source: 'plan.png', version: 1 })
    expect(result.current.loaded.plan.rooms).toHaveLength(2)
    expect(result.current.image).toEqual({ file: png, url: 'blob:preview' })
    expect(result.current.meta).toBe(meta)
    expect(result.current.error).toBeNull()
  })

  it('reports parse failures and keeps the current plan', async () => {
    vi.spyOn(api, 'parseFloorPlan').mockRejectedValue(new Error('No walls found.'))
    const { result } = renderHook(() => usePlanSource(INITIAL))

    act(() => result.current.uploadImage(png))
    await waitFor(() => expect(result.current.error).toBe('No walls found.'))

    expect(result.current.loaded).toBe(INITIAL)
    expect(result.current.busy).toBe(false)
  })

  it('re-parses the uploaded image with a new scale', async () => {
    const parse = vi.spyOn(api, 'parseFloorPlan').mockResolvedValue({ plan: parsePlan(sample), meta })
    const { result } = renderHook(() => usePlanSource(INITIAL))
    act(() => result.current.uploadImage(png))
    await waitFor(() => expect(result.current.image).not.toBeNull())

    act(() => result.current.reparse(2))
    await waitFor(() => expect(result.current.loaded.version).toBe(2))

    expect(parse).toHaveBeenLastCalledWith(png, expect.objectContaining({ cmPerPx: 2 }))
  })

  it('ignores re-parse requests before any image is uploaded', () => {
    const parse = vi.spyOn(api, 'parseFloorPlan')
    const { result } = renderHook(() => usePlanSource(INITIAL))
    act(() => result.current.reparse(2))
    expect(parse).not.toHaveBeenCalled()
  })

  it('loads a plan JSON and clears the image', async () => {
    vi.spyOn(api, 'parseFloorPlan').mockResolvedValue({ plan: parsePlan(sample), meta })
    const { result } = renderHook(() => usePlanSource(INITIAL))
    act(() => result.current.uploadImage(png))
    await waitFor(() => expect(result.current.image).not.toBeNull())

    const json = new File([JSON.stringify(sample)], 'plan.json', { type: 'application/json' })
    act(() => result.current.loadJson(json))
    await waitFor(() => expect(result.current.loaded.source).toBe('plan.json'))

    expect(result.current.image).toBeNull()
    expect(result.current.meta).toBeNull()
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:preview')
  })

  it('reports invalid JSON', async () => {
    const { result } = renderHook(() => usePlanSource(INITIAL))
    act(() => result.current.loadJson(new File(['{"walls": 3}'], 'bad.json')))
    await waitFor(() => expect(result.current.error).toMatch(/Invalid plan/))
  })

  it('cancels an in-flight upload when a new one starts', async () => {
    const signals: AbortSignal[] = []
    vi.spyOn(api, 'parseFloorPlan').mockImplementation(async (_file, options) => {
      signals.push(options!.signal!)
      if (signals.length === 1) {
        await new Promise((_, reject) => options!.signal!.addEventListener('abort', () => reject(new Error('aborted'))))
      }
      return { plan: parsePlan(sample), meta }
    })
    const { result } = renderHook(() => usePlanSource(INITIAL))

    act(() => result.current.uploadImage(png))
    act(() => result.current.uploadImage(png))
    await waitFor(() => expect(result.current.busy).toBe(false))

    expect(signals[0].aborted).toBe(true)
    expect(result.current.error).toBeNull()
    expect(result.current.loaded.version).toBe(1)
  })
})
