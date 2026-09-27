import { describe, expect, it } from 'vitest'
import sample from '../../../tests/fixtures/two_room_apartment.json'
import { parsePlan } from './schema'

describe('parsePlan', () => {
  it('parses the shared sample fixture', () => {
    const plan = parsePlan(sample)
    expect(plan.walls).toHaveLength(5)
    expect(plan.rooms.map((r) => r.kind)).toEqual(['living_room', 'bedroom'])
  })

  it('fills wall defaults', () => {
    const plan = parsePlan({ walls: [{ id: 'w1', start: { x: 0, y: 0 }, end: { x: 100, y: 0 } }] })
    expect(plan.walls[0]).toMatchObject({ thickness: 15, height: 250, exterior: false })
  })

  it('fills opening defaults by kind', () => {
    const plan = parsePlan({
      openings: [
        { id: 'd', wall_id: 'w1', kind: 'door', offset: 50, width: 90 },
        { id: 'w', wall_id: 'w1', kind: 'window', offset: 50, width: 90, height: null },
      ],
    })
    expect(plan.openings[0]).toMatchObject({ height: 210, sill_height: 0 })
    expect(plan.openings[1]).toMatchObject({ height: 120, sill_height: 90 })
  })

  it('defaults to an empty plan', () => {
    expect(parsePlan({})).toMatchObject({ version: 1, units: 'cm', walls: [], openings: [], rooms: [] })
  })

  it('rejects malformed input with a readable message', () => {
    expect(() => parsePlan({ rooms: [{ id: 'r1', polygon: [{ x: 0, y: 0 }] }] })).toThrow(/Invalid plan/)
    expect(() => parsePlan({ walls: [{ id: 'w1', start: { x: 0, y: 0 }, end: { x: 1, y: 0 }, typo: 1 }] })).toThrow(
      /Invalid plan/,
    )
  })
})
