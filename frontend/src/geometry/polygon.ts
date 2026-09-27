import type { Plan, Point } from '../plan/schema'

export interface Bounds {
  readonly minX: number
  readonly maxX: number
  readonly minY: number
  readonly maxY: number
  readonly center: Point
}

const edgesOf = (points: readonly Point[]) => points.map((a, i) => [a, points[(i + 1) % points.length]] as const)

const signedTwiceArea = (points: readonly Point[]) =>
  edgesOf(points).reduce((sum, [a, b]) => sum + a.x * b.y - b.x * a.y, 0)

/** Unsigned area via the shoelace formula, in the input units squared. */
export function polygonArea(points: readonly Point[]): number {
  return Math.abs(signedTwiceArea(points)) / 2
}

/** Area-weighted centroid; falls back to the vertex average when the area is ~0. */
export function polygonCentroid(points: readonly Point[]): Point {
  const edges = edgesOf(points)
  const twiceArea = signedTwiceArea(points)

  if (Math.abs(twiceArea) < 1e-9) {
    return {
      x: points.reduce((s, p) => s + p.x, 0) / points.length,
      y: points.reduce((s, p) => s + p.y, 0) / points.length,
    }
  }

  const sums = edges.reduce(
    (acc, [a, b]) => {
      const cross = a.x * b.y - b.x * a.y
      return { x: acc.x + (a.x + b.x) * cross, y: acc.y + (a.y + b.y) * cross }
    },
    { x: 0, y: 0 },
  )
  return { x: sums.x / (3 * twiceArea), y: sums.y / (3 * twiceArea) }
}

export function planBounds(plan: Plan): Bounds {
  const points = plan.walls.flatMap((w) => [w.start, w.end])
  if (points.length === 0) {
    return { minX: 0, maxX: 0, minY: 0, maxY: 0, center: { x: 0, y: 0 } }
  }
  const xs = points.map((p) => p.x)
  const ys = points.map((p) => p.y)
  const minX = Math.min(...xs)
  const maxX = Math.max(...xs)
  const minY = Math.min(...ys)
  const maxY = Math.max(...ys)
  return { minX, maxX, minY, maxY, center: { x: (minX + maxX) / 2, y: (minY + maxY) / 2 } }
}
