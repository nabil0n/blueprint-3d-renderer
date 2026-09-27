import type { ChangeEvent } from 'react'
import { ACCEPTED_IMAGE_TYPES } from '../api/parseFloorPlan'

interface SourcePickerProps {
  readonly busy: boolean
  readonly error: string | null
  readonly onImageSelected: (file: File) => void
  readonly onJsonSelected: (file: File) => void
}

/** Calls `handler` with the chosen file, then clears the input so the same file can be picked again. */
const pickFile = (handler: (file: File) => void) => (event: ChangeEvent<HTMLInputElement>) => {
  const file = event.target.files?.[0]
  if (file) handler(file)
  event.target.value = ''
}

export function SourcePicker({ busy, error, onImageSelected, onJsonSelected }: SourcePickerProps) {
  return (
    <section className="source-picker" aria-busy={busy}>
      <label className={`file-button${busy ? ' is-disabled' : ''}`}>
        {busy ? 'Reading floor plan…' : 'Upload floor plan image'}
        <input
          type="file"
          accept={ACCEPTED_IMAGE_TYPES.join(',')}
          disabled={busy}
          onChange={pickFile(onImageSelected)}
        />
      </label>
      <label className={`link-button${busy ? ' is-disabled' : ''}`}>
        or load a plan JSON
        <input type="file" accept="application/json,.json" disabled={busy} onChange={pickFile(onJsonSelected)} />
      </label>
      {error && (
        <pre className="error" role="alert">
          {error}
        </pre>
      )}
    </section>
  )
}
