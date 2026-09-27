import type { ComponentProps } from 'react'
import type { BackendStatus } from '../api/useBackendStatus'
import type { PlanSource } from '../app/usePlanSource'
import { ImageSummary } from './ImageSummary'
import { SourcePicker } from './SourcePicker'
import { ViewControls } from './ViewControls'

interface ControlPanelProps {
  readonly source: PlanSource
  readonly view: ComponentProps<typeof ViewControls>
  readonly backendStatus: BackendStatus
}

const STATUS_TEXT: Record<BackendStatus, string> = {
  checking: 'Checking backend…',
  online: 'Backend online',
  offline: 'Backend offline',
}

export function ControlPanel({ source, view, backendStatus }: ControlPanelProps) {
  return (
    <aside className="panel">
      <header>
        <h1>Blueprint 3D</h1>
        <p className="source" title={source.loaded.source}>
          {source.loaded.source}
        </p>
      </header>

      <SourcePicker
        busy={source.busy}
        error={source.error}
        onImageSelected={source.uploadImage}
        onJsonSelected={source.loadJson}
      />

      {source.image && source.meta && (
        <ImageSummary image={source.image} meta={source.meta} busy={source.busy} onRescale={source.reparse} />
      )}

      <ViewControls {...view} />

      <footer className={`status status-${backendStatus}`}>{STATUS_TEXT[backendStatus]}</footer>
    </aside>
  )
}
