import { useEffect, useMemo } from 'react'
import { Shape, ShapeGeometry, Vector2 } from 'three'
import { polygonCentroid } from '../geometry/polygon'
import { CM_TO_M } from '../geometry/units'
import type { Plan, Room } from '../plan/schema'
import { ROOM_COLORS } from './colors'
import { makeLabelTexture, roomLabelText } from './labels'

/** Lift floors slightly above the ground to avoid z-fighting. */
const FLOOR_LIFT_M = 0.002
const LABEL_HEIGHT_M = 0.6
const LABEL_LIFT_M = 0.6
/** Draw labels on top of walls so they never disappear behind them. */
const LABEL_RENDER_ORDER = 10

function FloorMesh({ room }: { readonly room: Room }) {
  // Shape lives in the XY plane; plan y is negated so that rotating -90° about X maps it to +Z.
  const geometry = useMemo(
    () => new ShapeGeometry(new Shape(room.polygon.map((p) => new Vector2(p.x * CM_TO_M, -p.y * CM_TO_M)))),
    [room],
  )
  useEffect(() => () => geometry.dispose(), [geometry])

  return (
    <mesh geometry={geometry} rotation-x={-Math.PI / 2} position-y={FLOOR_LIFT_M} receiveShadow>
      <meshStandardMaterial color={ROOM_COLORS[room.kind]} roughness={0.95} />
    </mesh>
  )
}

function RoomLabel({ room }: { readonly room: Room }) {
  const label = useMemo(() => makeLabelTexture(roomLabelText(room)), [room])
  useEffect(() => () => label?.texture.dispose(), [label])
  if (!label) return null

  const c = polygonCentroid(room.polygon)
  return (
    <sprite
      position={[c.x * CM_TO_M, LABEL_LIFT_M, c.y * CM_TO_M]}
      scale={[LABEL_HEIGHT_M * label.aspect, LABEL_HEIGHT_M, 1]}
      renderOrder={LABEL_RENDER_ORDER}
    >
      {/* Not tone-mapped: the card should keep its exact colours, like UI. */}
      <spriteMaterial map={label.texture} depthTest={false} toneMapped={false} transparent />
    </sprite>
  )
}

interface FloorsProps {
  readonly plan: Plan
  readonly showLabels: boolean
}

export function Floors({ plan, showLabels }: FloorsProps) {
  return (
    <group>
      {plan.rooms.map((room) => (
        <FloorMesh key={room.id} room={room} />
      ))}
      {showLabels && plan.rooms.map((room) => <RoomLabel key={room.id} room={room} />)}
    </group>
  )
}
