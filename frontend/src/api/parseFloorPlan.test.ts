import { afterEach, describe, expect, it, vi } from 'vitest'
import sample from '../../../tests/fixtures/two_room_apartment.json'
import { MAX_IMAGE_BYTES, parseFloorPlan, validateImageFile } from './parseFloorPlan'

const meta = {
  parser: 'opencv',
  cm_per_px: 1.5,
  scale_source: 'doors',
  scale_detail: 'Assumed 80 cm doors.',
  image_width: 1000,
  image_height: 800,
  warnings: [],
}

const png = (bytes = 10) => new File([new Uint8Array(bytes)], 'plan.png', { type: 'image/png' })

function mockFetch(response: Response) {
  const fetchMock = vi.fn().mockResolvedValue(response)
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('validateImageFile', () => {
  it('accepts PNG, JPEG and WebP', () => {
    for (const type of ['image/png', 'image/jpeg', 'image/webp']) {
      expect(validateImageFile(new File([], 'x', { type }))).toBeNull()
    }
  })

  it('rejects other types and oversized files', () => {
    expect(validateImageFile(new File([], 'plan.pdf', { type: 'application/pdf' }))).toMatch(/PNG, JPEG or WebP/)
    expect(validateImageFile(png(MAX_IMAGE_BYTES + 1))).toMatch(/10 MB/)
  })
})

describe('parseFloorPlan', () => {
  it('posts the image and returns a validated plan and meta', async () => {
    const fetchMock = mockFetch(Response.json({ plan: sample, meta }))

    const result = await parseFloorPlan(png())

    expect(result.plan.rooms).toHaveLength(2)
    expect(result.meta.cm_per_px).toBe(1.5)
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('/api/plans/parse')
    expect(init.method).toBe('POST')
    expect((init.body as FormData).get('file')).toBeInstanceOf(File)
    expect((init.body as FormData).has('cm_per_px')).toBe(false)
  })

  it('sends a scale override when given', async () => {
    const fetchMock = mockFetch(Response.json({ plan: sample, meta: { ...meta, cm_per_px: 2, scale_source: 'user' } }))
    await parseFloorPlan(png(), { cmPerPx: 2 })
    expect((fetchMock.mock.calls[0][1].body as FormData).get('cm_per_px')).toBe('2')
  })

  it('rejects invalid files without calling the backend', async () => {
    const fetchMock = mockFetch(Response.json({}))
    await expect(parseFloorPlan(new File([], 'a.gif', { type: 'image/gif' }))).rejects.toThrow(/PNG, JPEG or WebP/)
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('surfaces the backend error message', async () => {
    mockFetch(Response.json({ detail: 'No walls found.' }, { status: 422 }))
    await expect(parseFloorPlan(png())).rejects.toThrow('No walls found.')
  })

  it('summarises FastAPI validation errors', async () => {
    mockFetch(Response.json({ detail: [{ msg: 'Input should be greater than 0' }] }, { status: 422 }))
    await expect(parseFloorPlan(png())).rejects.toThrow('Input should be greater than 0')
  })

  it('falls back to the status code for non-JSON errors', async () => {
    mockFetch(new Response('Bad gateway', { status: 502 }))
    await expect(parseFloorPlan(png())).rejects.toThrow(/HTTP 502/)
  })

  it('explains when the backend is unreachable', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('fetch failed')))
    await expect(parseFloorPlan(png())).rejects.toThrow(/reach the backend/)
  })

  it('rejects malformed responses', async () => {
    mockFetch(Response.json({ plan: sample }))
    await expect(parseFloorPlan(png())).rejects.toThrow(/Unexpected response/)
  })
})
