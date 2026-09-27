import type { RoomKind } from '../plan/schema'

export const ROOM_COLORS: Record<RoomKind, string> = {
  living_room: '#e6d3ae',
  bedroom: '#a9c4e6',
  kitchen: '#f2c88e',
  bathroom: '#9fd9d2',
  hallway: '#cfd9e4',
  closet: '#b4c0cf',
  balcony: '#b5d39f',
  other: '#c8d1dc',
}

export const ROOM_LABELS: Record<RoomKind, string> = {
  living_room: 'Living room',
  bedroom: 'Bedroom',
  kitchen: 'Kitchen',
  bathroom: 'Bathroom',
  hallway: 'Hallway',
  closet: 'Closet',
  balcony: 'Balcony',
  other: 'Room',
}

/** Matches the cyanotype theme in index.css. */
export const WALL_COLORS = {
  interior: '#f3f7fc',
  exterior: '#dbe6f2',
  /** Top faces, where the dollhouse cut exposes the wall section: drawn like a section cut. */
  cap: '#ff7a1a',
  ground: '#12345c',
  /** Sky and fog: the ground fades into it, so the plane has no visible edge. */
  horizon: '#081b33',
} as const

export const GROUND_GRID = {
  cell: '#2a5585',
  section: '#4f7fb5',
} as const

export const LABEL_COLORS = {
  card: 'rgba(8, 27, 51, 0.92)',
  mark: '#ff7a1a',
  title: '#eaf2fb',
  subtitle: '#9fb8d4',
} as const
