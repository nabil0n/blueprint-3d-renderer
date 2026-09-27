import { Bounds, OrbitControls } from '@react-three/drei'
import { Canvas } from '@react-three/fiber'
import { useMemo } from 'react'
import { planBounds } from '../geometry/polygon'
import { CM_TO_M } from '../geometry/units'
import type { Plan } from '../plan/schema'
import { WALL_COLORS } from './colors'
import { Floors } from './Floors'
import { Walls } from './Walls'

interface DollhouseProps {
  readonly plan: Plan
  readonly cutHeight: number
  readonly showLabels: boolean
}

const MIN_SPAN_M = 6
/** Keep the camera above the floor so the view always reads as a dollhouse. */
const MAX_POLAR_ANGLE = Math.PI / 2.2

export function Dollhouse({ plan, cutHeight, showLabels }: DollhouseProps) {
  const bounds = useMemo(() => planBounds(plan), [plan])
  const width = (bounds.maxX - bounds.minX) * CM_TO_M
  const depth = (bounds.maxY - bounds.minY) * CM_TO_M
  const span = Math.max(width, depth, MIN_SPAN_M)
  const offset: [number, number, number] = [-bounds.center.x * CM_TO_M, 0, -bounds.center.y * CM_TO_M]

  return (
    <Canvas shadows camera={{ position: [span * 0.7, span * 1.0, span * 1.0], fov: 40 }}>
      <hemisphereLight args={['#ffffff', '#b9b4aa', 1.4]} />
      <directionalLight
        position={[span * 0.4, span, span * 0.6]}
        intensity={1.8}
        castShadow
        shadow-mapSize={[2048, 2048]}
        shadow-camera-left={-span}
        shadow-camera-right={span}
        shadow-camera-top={span}
        shadow-camera-bottom={-span}
        shadow-bias={-0.0005}
      />

      {/* Refits the camera when the canvas resizes, e.g. when the original opens beside it. */}
      <Bounds fit observe margin={1.1}>
        <group position={offset}>
          <Floors plan={plan} showLabels={showLabels} />
          <Walls plan={plan} cutHeight={cutHeight} />
        </group>
      </Bounds>

      <mesh rotation-x={-Math.PI / 2} position-y={-0.01} receiveShadow>
        <planeGeometry args={[span * 4, span * 4]} />
        <meshStandardMaterial color={WALL_COLORS.ground} roughness={1} />
      </mesh>

      <OrbitControls makeDefault maxPolarAngle={MAX_POLAR_ANGLE} minDistance={span * 0.3} maxDistance={span * 4} />
    </Canvas>
  )
}
