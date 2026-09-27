import type { RoomKind } from '../plan/schema'

export const ROOM_COLORS: Record<RoomKind, string> = {
  living_room: '#d9c7a7',
  bedroom: '#b9c8d8',
  kitchen: '#e3cfa0',
  bathroom: '#a9d1cc',
  hallway: '#d6d0c6',
  closet: '#c9c1b6',
  balcony: '#bcc9a8',
  other: '#d2d2d2',
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

export const WALL_COLORS = {
  interior: '#f4f1ec',
  exterior: '#e6e1d8',
  /** Top faces, where the dollhouse cut exposes the wall section. */
  cap: '#3b3a38',
  ground: '#cfcac1',
} as const
