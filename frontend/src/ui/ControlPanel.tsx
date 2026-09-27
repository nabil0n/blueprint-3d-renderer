import type { ComponentProps } from 'react'
import type { BackendStatus } from '../api/useBackendStatus'
import type { ParserOption } from '../api/useParsers'
import type { PlanSource } from '../app/usePlanSource'
import { ImageSummary } from './ImageSummary'
import { ParserPicker } from './ParserPicker'
import { SourcePicker } from './SourcePicker'
import { ViewControls } from './ViewControls'

interface ControlPanelProps {
  readonly source: PlanSource
  readonly view: ComponentProps<typeof ViewControls>
  readonly backendStatus: BackendStatus
  readonly parsers: readonly ParserOption[]
  readonly comparing: boolean
  readonly onToggleOriginal: () => void
}

const STATUS_TEXT: Record<BackendStatus, string> = {
  checking: 'Checking backend…',
  online: 'Backend online',
  offline: 'Backend offline',
}

export function ControlPanel({ source, view, backendStatus, parsers, comparing, onToggleOriginal }: ControlPanelProps) {
  return (
    <aside className="panel">
      <header>
        <h1>Blueprint 3D</h1>
        <p className="source" title={source.loaded.source}>
          {source.loaded.source}
        </p>
      </header>

      <section className="panel-section">
        <h2>Plan</h2>
        <ParserPicker options={parsers} selected={source.parser} busy={source.busy} onChange={source.chooseParser} />
        <SourcePicker
          busy={source.busy}
          error={source.error}
          onImageSelected={source.uploadImage}
          onJsonSelected={source.loadJson}
        />
      </section>

      {source.image && source.meta && (
        <ImageSummary
          image={source.image}
          meta={source.meta}
          busy={source.busy}
          comparing={comparing}
          onRescale={source.reparse}
          onToggleOriginal={onToggleOriginal}
        />
      )}

      <ViewControls {...view} />

      <footer className={`status status-${backendStatus}`}>{STATUS_TEXT[backendStatus]}</footer>
    </aside>
  )
}
