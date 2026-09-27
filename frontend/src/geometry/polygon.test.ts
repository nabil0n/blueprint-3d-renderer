import { describe, expect, it } from 'vitest'
import type { Plan } from '../plan/schema'
import { planBounds, polygonArea, polygonCentroid } from './polygon'

const square = [
  { x: 0, y: 0 },
  { x: 400, y: 0 },
  { x: 400, y: 200 },
  { x: 0, y: 200 },
]

describe('polygonArea', () => {
  it('is positive regardless of winding', () => {
    expect(polygonArea(square)).toBe(80_000)
    expect(polygonArea([...square].reverse())).toBe(80_000)
  })
})

describe('polygonCentroid', () => {
  it('finds the centre of a rectangle regardless of winding', () => {
    expect(polygonCentroid(square)).toEqual({ x: 200, y: 100 })
    expect(polygonCentroid([...square].reverse())).toEqual({ x: 200, y: 100 })
  })

  it('weights by area for an L-shape', () => {
    const l = [
      { x: 0, y: 0 },
      { x: 200, y: 0 },
      { x: 200, y: 100 },
      { x: 100, y: 100 },
      { x: 100, y: 200 },
      { x: 0, y: 200 },
    ]
    const c = polygonCentroid(l)
    expect(c.x).toBeCloseTo(250 / 3)
    expect(c.y).toBeCloseTo(250 / 3)
  })

  it('falls back to the vertex average for degenerate polygons', () => {
    expect(polygonCentroid([{ x: 0, y: 0 }, { x: 2, y: 0 }, { x: 4, y: 0 }])).toEqual({ x: 2, y: 0 })
  })
})

describe('planBounds', () => {
  const plan = (walls: Plan['walls']): Plan => ({ version: 1, units: 'cm', walls, openings: [], rooms: [] })
  const w = (id: string, x0: number, y0: number, x1: number, y1: number) => ({
    id,
    start: { x: x0, y: y0 },
    end: { x: x1, y: y1 },
    thickness: 15,
    height: 250,
    exterior: false,
  })

  it('spans all wall endpoints', () => {
    expect(planBounds(plan([w('a', 100, 50, 500, 50), w('b', 500, 50, 500, 350)]))).toEqual({
      minX: 100,
      maxX: 500,
      minY: 50,
      maxY: 350,
      center: { x: 300, y: 200 },
    })
  })

  it('is centred on the origin for an empty plan', () => {
    expect(planBounds(plan([])).center).toEqual({ x: 0, y: 0 })
  })
})
