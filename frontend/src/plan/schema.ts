/**
 * Runtime mirror of the Python plan schema (src/blueprint3d/schema.py).
 * Structural validation and defaults only; cross-reference checks live in the backend.
 * Coordinates are plan space: centimetres, origin top-left, x right, y down.
 */
import { z } from 'zod'

export const DEFAULTS = {
  wallThickness: 15,
  wallHeight: 250,
  doorHeight: 210,
  windowHeight: 120,
  windowSill: 90,
} as const

const point = z.strictObject({ x: z.number(), y: z.number() })

const wall = z.strictObject({
  id: z.string(),
  start: point,
  end: point,
  thickness: z.number().positive().default(DEFAULTS.wallThickness),
  height: z.number().positive().default(DEFAULTS.wallHeight),
  exterior: z.boolean().default(false),
})

const opening = z
  .strictObject({
    id: z.string(),
    wall_id: z.string(),
    kind: z.enum(['door', 'window']),
    offset: z.number().nonnegative(),
    width: z.number().positive(),
    height: z.number().positive().nullish(),
    sill_height: z.number().nonnegative().nullish(),
  })
  .transform((o) => {
    const isDoor = o.kind === 'door'
    return {
      ...o,
      height: o.height ?? (isDoor ? DEFAULTS.doorHeight : DEFAULTS.windowHeight),
      sill_height: o.sill_height ?? (isDoor ? 0 : DEFAULTS.windowSill),
    }
  })

export const ROOM_KINDS = [
  'living_room',
  'bedroom',
  'kitchen',
  'bathroom',
  'hallway',
  'closet',
  'balcony',
  'other',
] as const

const room = z.strictObject({
  id: z.string(),
  polygon: z.array(point).min(3),
  kind: z.enum(ROOM_KINDS).default('other'),
  name: z.string().nullish(),
})

const plan = z.strictObject({
  version: z.literal(1).default(1),
  units: z.literal('cm').default('cm'),
  walls: z.array(wall).default([]),
  openings: z.array(opening).default([]),
  rooms: z.array(room).default([]),
})

export type Point = z.infer<typeof point>
export type Wall = z.infer<typeof wall>
export type Opening = z.infer<typeof opening>
export type RoomKind = (typeof ROOM_KINDS)[number]
export type Room = z.infer<typeof room>
export type Plan = z.infer<typeof plan>

export function parsePlan(raw: unknown): Plan {
  const result = plan.safeParse(raw)
  if (!result.success) {
    throw new Error(`Invalid plan:\n${z.prettifyError(result.error)}`)
  }
  return result.data
}
