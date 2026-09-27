import { useEffect, useMemo } from 'react'
import { MeshStandardMaterial } from 'three'
import { pieceTransform, wallPieces } from '../geometry/walls'
import type { Plan } from '../plan/schema'
import { WALL_COLORS } from './colors'

interface WallsProps {
  readonly plan: Plan
  readonly cutHeight: number
}

/** Box faces are ordered +x, -x, +y, -y, +z, -z; +y is the exposed top cap. */
function faceMaterials(side: MeshStandardMaterial, cap: MeshStandardMaterial) {
  return [side, side, cap, side, side, side]
}

function useWallMaterials() {
  const materials = useMemo(() => {
    const cap = new MeshStandardMaterial({ color: WALL_COLORS.cap, roughness: 0.9 })
    const interior = new MeshStandardMaterial({ color: WALL_COLORS.interior, roughness: 0.85 })
    const exterior = new MeshStandardMaterial({ color: WALL_COLORS.exterior, roughness: 0.85 })
    return {
      interior: faceMaterials(interior, cap),
      exterior: faceMaterials(exterior, cap),
      all: [cap, interior, exterior],
    }
  }, [])
  useEffect(() => () => materials.all.forEach((m) => m.dispose()), [materials])
  return materials
}

export function Walls({ plan, cutHeight }: WallsProps) {
  const materials = useWallMaterials()
  const boxes = useMemo(
    () =>
      plan.walls.flatMap((wall) => {
        const openings = plan.openings.filter((o) => o.wall_id === wall.id)
        return wallPieces(wall, openings, cutHeight).map((piece, i) => ({
          key: `${wall.id}-${i}`,
          exterior: wall.exterior,
          ...pieceTransform(wall, piece),
        }))
      }),
    [plan, cutHeight],
  )

  return (
    <group>
      {boxes.map((box) => (
        <mesh
          key={box.key}
          position={box.position}
          rotation-y={box.rotationY}
          material={box.exterior ? materials.exterior : materials.interior}
          castShadow
          receiveShadow
        >
          <boxGeometry args={box.size} />
        </mesh>
      ))}
    </group>
  )
}
