import { useState } from 'react'
import sample from '../../tests/fixtures/two_room_apartment.json'
import { useBackendStatus } from './api/useBackendStatus'
import { usePlanSource, type LoadedPlan } from './app/usePlanSource'
import { DEFAULTS, parsePlan, type Plan } from './plan/schema'
import { Dollhouse } from './scene/Dollhouse'
import { ControlPanel } from './ui/ControlPanel'

const SAMPLE: LoadedPlan = { plan: parsePlan(sample), source: 'Sample: two-room apartment', version: 0 }

const maxWallHeight = (plan: Plan) => Math.max(DEFAULTS.wallHeight, ...plan.walls.map((w) => w.height))

export default function App() {
  const source = usePlanSource(SAMPLE)
  const { loaded } = source
  // The cut belongs to one plan version; a newly loaded plan starts at full height.
  const [cut, setCut] = useState<{ version: number; value: number } | null>(null)
  const [showLabels, setShowLabels] = useState(true)
  const backendStatus = useBackendStatus()

  const maxHeight = maxWallHeight(loaded.plan)
  const cutHeight = cut?.version === loaded.version ? cut.value : maxHeight

  return (
    <main className="app">
      <Dollhouse key={loaded.version} plan={loaded.plan} cutHeight={cutHeight} showLabels={showLabels} />
      <ControlPanel
        source={source}
        backendStatus={backendStatus}
        view={{
          cutHeight,
          maxHeight,
          showLabels,
          onCutHeightChange: (value) => setCut({ version: loaded.version, value }),
          onShowLabelsChange: setShowLabels,
        }}
      />
    </main>
  )
}
