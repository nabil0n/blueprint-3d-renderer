import { useState, type FormEvent } from 'react'
import type { ParseMeta } from '../api/parseFloorPlan'
import type { ImageSource } from '../app/usePlanSource'

interface ImageSummaryProps {
  readonly image: ImageSource
  readonly meta: ParseMeta
  readonly busy: boolean
  readonly onRescale: (cmPerPx: number) => void
}

const SCALE_SOURCE_TEXT: Record<ParseMeta['scale_source'], string> = {
  user: 'set by you',
  doors: 'estimated from doors',
  wall_thickness: 'estimated from wall thickness',
}

export function ImageSummary({ image, meta, busy, onRescale }: ImageSummaryProps) {
  return (
    <section className="image-summary">
      <img className="thumb" src={image.url} alt={`Uploaded floor plan ${image.file.name}`} />
      {/* Keyed so the field resets to the new value after every parse. */}
      <ScaleForm key={meta.cm_per_px} meta={meta} busy={busy} onRescale={onRescale} />
      {meta.warnings.length > 0 && (
        <ul className="warnings">
          {meta.warnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      )}
    </section>
  )
}

function ScaleForm({ meta, busy, onRescale }: Omit<ImageSummaryProps, 'image'>) {
  const [value, setValue] = useState(meta.cm_per_px.toFixed(3))
  const parsed = Number(value)
  const valid = Number.isFinite(parsed) && parsed > 0 && parsed <= 100

  const submit = (event: FormEvent) => {
    event.preventDefault()
    if (valid) onRescale(parsed)
  }

  return (
    <form className="scale-form" onSubmit={submit}>
      <label htmlFor="cm-per-px">
        Scale <span className="muted">({SCALE_SOURCE_TEXT[meta.scale_source]})</span>
      </label>
      <div className="scale-row">
        <input
          id="cm-per-px"
          type="number"
          inputMode="decimal"
          min={0.001}
          max={100}
          step="any"
          value={value}
          aria-invalid={!valid}
          onChange={(e) => setValue(e.target.value)}
        />
        <span className="muted">cm/px</span>
        <button type="submit" disabled={busy || !valid}>
          Apply
        </button>
      </div>
      <p className="hint">Known length in cm ÷ its length in image pixels.</p>
    </form>
  )
}
