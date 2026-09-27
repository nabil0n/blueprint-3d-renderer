import { describe, expect, it } from 'vitest'
import type { Opening, Wall } from '../plan/schema'
import { pieceTransform, wallLength, wallPieces } from './walls'

const wall = (overrides: Partial<Wall> = {}): Wall => ({
  id: 'w1',
  start: { x: 0, y: 0 },
  end: { x: 400, y: 0 },
  thickness: 20,
  height: 250,
  exterior: false,
  ...overrides,
})

const opening = (overrides: Partial<Opening> = {}): Opening => ({
  id: 'o1',
  wall_id: 'w1',
  kind: 'door',
  offset: 200,
  width: 100,
  height: 210,
  sill_height: 0,
  ...overrides,
})

describe('wallLength', () => {
  it('is the distance between endpoints', () => {
    expect(wallLength(wall({ end: { x: 300, y: 400 } }))).toBe(500)
  })
})

describe('wallPieces', () => {
  it('returns one solid piece, extended by half the thickness at both ends', () => {
    expect(wallPieces(wall(), [])).toEqual([{ u0: -10, u1: 410, v0: 0, v1: 250 }])
  })

  it('splits around a door and adds a lintel above it', () => {
    expect(wallPieces(wall(), [opening()])).toEqual([
      { u0: -10, u1: 150, v0: 0, v1: 250 },
      { u0: 150, u1: 250, v0: 210, v1: 250 },
      { u0: 250, u1: 410, v0: 0, v1: 250 },
    ])
  })

  it('adds both a sill piece and a lintel for a window', () => {
    const pieces = wallPieces(wall(), [opening({ kind: 'window', sill_height: 90, height: 120 })])
    expect(pieces).toContainEqual({ u0: 150, u1: 250, v0: 0, v1: 90 })
    expect(pieces).toContainEqual({ u0: 150, u1: 250, v0: 210, v1: 250 })
  })

  it('handles openings in any order', () => {
    const a = opening({ id: 'a', offset: 100, width: 50 })
    const b = opening({ id: 'b', offset: 300, width: 50 })
    expect(wallPieces(wall(), [b, a])).toEqual(wallPieces(wall(), [a, b]))
  })

  it('does not produce overlapping pieces for overlapping openings', () => {
    const pieces = wallPieces(wall(), [opening({ offset: 150 }), opening({ id: 'o2', offset: 180 })])
    const full = pieces.filter((p) => p.v0 === 0)
    for (let i = 1; i < full.length; i++) {
      expect(full[i].u0).toBeGreaterThanOrEqual(full[i - 1].u1)
    }
  })

  it('ignores an opening that lies entirely inside another', () => {
    const big = opening({ width: 200 })
    expect(wallPieces(wall(), [big, opening({ id: 'o2', width: 50 })])).toEqual(wallPieces(wall(), [big]))
  })

  it('clamps everything to the cut height and drops pieces above it', () => {
    const pieces = wallPieces(wall(), [opening()], 120)
    expect(pieces).toEqual([
      { u0: -10, u1: 150, v0: 0, v1: 120 },
      { u0: 250, u1: 410, v0: 0, v1: 120 },
    ])
  })

  it('returns nothing when the cut height is zero', () => {
    expect(wallPieces(wall(), [], 0)).toEqual([])
  })
})

describe('pieceTransform', () => {
  it('centres a piece in world space, in metres, with plan y mapped to world z', () => {
    const t = pieceTransform(wall({ start: { x: 100, y: 200 }, end: { x: 100, y: 600 } }), {
      u0: 0,
      u1: 400,
      v0: 0,
      v1: 250,
    })
    expect(t.position[0]).toBeCloseTo(1)
    expect(t.position[1]).toBeCloseTo(1.25)
    expect(t.position[2]).toBeCloseTo(4)
    expect(t.size).toEqual([4, 2.5, 0.2])
  })

  it('rotates so the box x-axis follows the wall direction', () => {
    const along = pieceTransform(wall({ end: { x: 0, y: 400 } }), { u0: 0, u1: 400, v0: 0, v1: 250 })
    // Rotating local +x by rotationY about +Y must give plan +y, i.e. world +z.
    expect(Math.cos(along.rotationY)).toBeCloseTo(0)
    expect(-Math.sin(along.rotationY)).toBeCloseTo(1)
  })
})
