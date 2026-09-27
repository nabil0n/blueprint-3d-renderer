import { useCallback, useState } from 'react'
import sample from '../../tests/fixtures/two_room_apartment.json'
import { useBackendStatus } from './api/useBackendStatus'
import { useParsers } from './api/useParsers'
import { usePlanSource, type LoadedPlan } from './app/usePlanSource'
import { DEFAULTS, parsePlan, type Plan } from './plan/schema'
import { Dollhouse } from './scene/Dollhouse'
import { ControlPanel } from './ui/ControlPanel'
import { OriginalViewer } from './ui/OriginalViewer'

const SAMPLE: LoadedPlan = { plan: parsePlan(sample), source: 'Sample: two-room apartment', version: 0 }

const maxWallHeight = (plan: Plan) => Math.max(DEFAULTS.wallHeight, ...plan.walls.map((w) => w.height))

export default function App() {
  const source = usePlanSource(SAMPLE)
  const { loaded } = source
  // The cut belongs to one plan version; a newly loaded plan starts at full height.
  const [cut, setCut] = useState<{ version: number; value: number } | null>(null)
  const [showLabels, setShowLabels] = useState(true)
  const [comparing, setComparing] = useState(false)
  const closeOriginal = useCallback(() => setComparing(false), [])
  const backendStatus = useBackendStatus()
  const parsers = useParsers()

  const maxHeight = maxWallHeight(loaded.plan)
  const cutHeight = cut?.version === loaded.version ? cut.value : maxHeight
  // A plan loaded from JSON has no image to compare with.
  const original = comparing ? source.image : null

  return (
    <main className={`app${original ? ' is-comparing' : ''}`}>
      <ControlPanel
        source={source}
        backendStatus={backendStatus}
        parsers={parsers}
        comparing={original !== null}
        onToggleOriginal={() => setComparing(original === null)}
        view={{
          cutHeight,
          maxHeight,
          showLabels,
          onCutHeightChange: (value) => setCut({ version: loaded.version, value }),
          onShowLabelsChange: setShowLabels,
        }}
      />
      <div className="stage">
        <Dollhouse key={loaded.version} plan={loaded.plan} cutHeight={cutHeight} showLabels={showLabels} />
      </div>
      {original && <OriginalViewer image={original} onClose={closeOriginal} />}
    </main>
  )
}
