/**
 * Turns a wall with openings into solid box pieces.
 *
 * Pieces are described in wall-local coordinates (cm):
 *   u — distance along the wall from its start point
 *   v — height above the floor
 * Each wall is extended by half its thickness at both ends so corners close without gaps.
 */
import type { Opening, Wall } from '../plan/schema'
import { CM_TO_M } from './units'

export interface WallPiece {
  readonly u0: number
  readonly u1: number
  readonly v0: number
  readonly v1: number
}

export interface PieceTransform {
  readonly position: [number, number, number]
  readonly rotationY: number
  readonly size: [number, number, number]
}

const EPSILON = 1e-6

export function wallLength(wall: Wall): number {
  return Math.hypot(wall.end.x - wall.start.x, wall.end.y - wall.start.y)
}

export function wallPieces(wall: Wall, openings: readonly Opening[], cutHeight = Infinity): WallPiece[] {
  const top = Math.min(wall.height, cutHeight)
  const half = wall.thickness / 2
  const sorted = [...openings].sort((a, b) => a.offset - b.offset)

  const { pieces, cursor } = sorted.reduce<{ pieces: WallPiece[]; cursor: number }>(
    (acc, o) => {
      const left = Math.max(o.offset - o.width / 2, acc.cursor)
      const right = o.offset + o.width / 2
      if (right <= left) return acc
      return {
        cursor: right,
        pieces: [
          ...acc.pieces,
          { u0: acc.cursor, u1: left, v0: 0, v1: top },
          { u0: left, u1: right, v0: 0, v1: o.sill_height },
          { u0: left, u1: right, v0: o.sill_height + o.height, v1: top },
        ],
      }
    },
    { pieces: [], cursor: -half },
  )

  return [...pieces, { u0: cursor, u1: wallLength(wall) + half, v0: 0, v1: top }]
    .map((p) => ({ ...p, v1: Math.min(p.v1, top) }))
    .filter((p) => p.u1 - p.u0 > EPSILON && p.v1 - p.v0 > EPSILON)
}

/** World transform for a box piece: metres, plan (x, y) mapped to world (x, z), y up. */
export function pieceTransform(wall: Wall, piece: WallPiece): PieceTransform {
  const length = wallLength(wall)
  const dx = (wall.end.x - wall.start.x) / length
  const dy = (wall.end.y - wall.start.y) / length
  const uMid = (piece.u0 + piece.u1) / 2

  return {
    position: [
      (wall.start.x + dx * uMid) * CM_TO_M,
      ((piece.v0 + piece.v1) / 2) * CM_TO_M,
      (wall.start.y + dy * uMid) * CM_TO_M,
    ],
    rotationY: -Math.atan2(dy, dx),
    size: [(piece.u1 - piece.u0) * CM_TO_M, (piece.v1 - piece.v0) * CM_TO_M, wall.thickness * CM_TO_M],
  }
}
