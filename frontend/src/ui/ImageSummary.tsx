import { useState, type FormEvent } from 'react'
import type { ParseMeta } from '../api/parseFloorPlan'
import type { ImageSource } from '../app/usePlanSource'

interface ImageSummaryProps {
  readonly image: ImageSource
  readonly meta: ParseMeta
  readonly busy: boolean
  /** Whether the original is open next to the 3D view. */
  readonly comparing: boolean
  readonly onRescale: (cmPerPx: number) => void
  readonly onToggleOriginal: () => void
}

const SCALE_SOURCE_TEXT: Record<ParseMeta['scale_source'], string> = {
  user: 'set by you',
  scale_bar: 'from the scale bar',
  page_format: 'assumed A4 at 1:100',
  doors: 'estimated from doors',
  wall_thickness: 'estimated from walls',
}

export function ImageSummary({ image, meta, busy, comparing, onRescale, onToggleOriginal }: ImageSummaryProps) {
  return (
    <section className="panel-section">
      <h2>Original</h2>
      <button type="button" className="thumb" aria-pressed={comparing} onClick={onToggleOriginal}>
        <img src={image.url} alt="" />
        <span className="thumb-label">{comparing ? 'Hide original' : 'Compare with original'}</span>
      </button>
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

type ScaleFormProps = Pick<ImageSummaryProps, 'meta' | 'busy' | 'onRescale'>

function ScaleForm({ meta, busy, onRescale }: ScaleFormProps) {
  const [value, setValue] = useState(meta.cm_per_px.toFixed(3))
  const parsed = Number(value)
  const valid = Number.isFinite(parsed) && parsed > 0 && parsed <= 100

  const submit = (event: FormEvent) => {
    event.preventDefault()
    if (valid) onRescale(parsed)
  }

  return (
    <form className="field" onSubmit={submit}>
      <label htmlFor="cm-per-px">
        Scale <span className="muted">{SCALE_SOURCE_TEXT[meta.scale_source]}</span>
      </label>
      <div className="scale-row">
        <input
          id="cm-per-px"
          className="input"
          type="number"
          inputMode="decimal"
          min={0.001}
          max={100}
          step="any"
          value={value}
          aria-invalid={!valid}
          title="Known length in cm ÷ its length in image pixels"
          onChange={(e) => setValue(e.target.value)}
        />
        <span className="muted">cm/px</span>
        <button type="submit" className="outline-button" disabled={busy || !valid}>
          Apply
        </button>
      </div>
    </form>
  )
}
