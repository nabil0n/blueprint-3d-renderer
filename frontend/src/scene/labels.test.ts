import { describe, expect, it } from 'vitest'
import type { Room } from '../plan/schema'
import { roomLabelText } from './labels'

const room = (overrides: Partial<Room> = {}): Room => ({
  id: 'r1',
  kind: 'other',
  name: null,
  polygon: [
    { x: 0, y: 0 },
    { x: 400, y: 0 },
    { x: 400, y: 300 },
  ],
  ...overrides,
})

describe('roomLabelText', () => {
  it('uses the room name and area in square metres', () => {
    expect(roomLabelText(room({ name: 'Kök' }))).toEqual({ title: 'Kök', subtitle: '6.0 m²' })
  })

  it('falls back to the room kind', () => {
    expect(roomLabelText(room({ kind: 'bathroom' })).title).toBe('Bathroom')
    expect(roomLabelText(room()).title).toBe('Room')
  })
})
