import { z } from 'zod'
import { parsePlan, type Plan } from '../plan/schema'

export const ACCEPTED_IMAGE_TYPES = ['image/png', 'image/jpeg', 'image/webp'] as const
export const MAX_IMAGE_BYTES = 10 * 1024 * 1024

const metaSchema = z.object({
  parser: z.string(),
  cm_per_px: z.number().positive(),
  scale_source: z.enum(['user', 'scale_bar', 'page_format', 'doors', 'wall_thickness']),
  scale_detail: z.string(),
  image_width: z.number().int().positive(),
  image_height: z.number().int().positive(),
  /** Where plan (0, 0) lies in the uploaded image: image px = origin_px + plan cm / cm_per_px. */
  origin_px: z.tuple([z.number(), z.number()]).default([0, 0]),
  warnings: z.array(z.string()),
})

const responseSchema = z.object({ plan: z.unknown(), meta: metaSchema })

export type ParseMeta = z.infer<typeof metaSchema>

export interface ParsedFloorPlan {
  readonly plan: Plan
  readonly meta: ParseMeta
}

interface ParseOptions {
  /** Known scale in centimetres per pixel of the uploaded image; estimated by the backend if omitted. */
  readonly cmPerPx?: number
  /** Which backend parser to use (see GET /api/parsers); the backend's default if omitted. */
  readonly parser?: string
  readonly signal?: AbortSignal
}

/** Returns a user-facing problem with the file, or null if it can be uploaded. */
export function validateImageFile(file: File): string | null {
  if (!(ACCEPTED_IMAGE_TYPES as readonly string[]).includes(file.type)) {
    return `${file.name} is not a PNG, JPEG or WebP image.`
  }
  if (file.size > MAX_IMAGE_BYTES) {
    return `${file.name} is larger than ${MAX_IMAGE_BYTES / (1024 * 1024)} MB.`
  }
  return null
}

export async function parseFloorPlan(file: File, options: ParseOptions = {}): Promise<ParsedFloorPlan> {
  const problem = validateImageFile(file)
  if (problem) throw new Error(problem)

  const body = new FormData()
  body.append('file', file)
  if (options.cmPerPx !== undefined) body.append('cm_per_px', String(options.cmPerPx))
  if (options.parser !== undefined) body.append('parser', options.parser)

  const response = await post(body, options.signal)
  if (!response.ok) throw new Error(await errorMessage(response))

  const parsed = responseSchema.safeParse(await response.json())
  if (!parsed.success) throw new Error('Unexpected response from the backend.')
  return { plan: parsePlan(parsed.data.plan), meta: parsed.data.meta }
}

async function post(body: FormData, signal?: AbortSignal): Promise<Response> {
  try {
    return await fetch('/api/plans/parse', { method: 'POST', body, signal })
  } catch (err) {
    if (signal?.aborted) throw err
    throw new Error('Could not reach the backend. Is it running?', { cause: err })
  }
}

async function errorMessage(response: Response): Promise<string> {
  try {
    const { detail } = (await response.json()) as { detail?: unknown }
    if (typeof detail === 'string') return detail
    if (Array.isArray(detail)) {
      return detail.map((d: { msg?: string }) => d.msg ?? 'Invalid request').join('; ')
    }
  } catch {
    // Not JSON; fall through to the generic message.
  }
  return `Parsing failed (HTTP ${response.status}).`
}
