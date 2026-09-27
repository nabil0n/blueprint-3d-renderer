import type { ParserOption } from '../api/useParsers'

interface ParserPickerProps {
  readonly options: readonly ParserOption[]
  /** The chosen parser; null means the backend's default. */
  readonly selected: string | null
  readonly busy: boolean
  readonly onChange: (name: string) => void
}

const LABELS: Record<string, string> = {
  opencv: 'Classical (OpenCV)',
  cubicasa: 'Learned (CubiCasa5K)',
}

/** Chooses how uploaded plans are read. Hidden while the backend offers only one parser. */
export function ParserPicker({ options, selected, busy, onChange }: ParserPickerProps) {
  if (options.length < 2) return null
  const current = options.find((o) => o.name === selected) ?? options.find((o) => o.default) ?? options[0]
  const unavailable = options.filter((o) => !o.available)

  return (
    <section className="field">
      <label className="field">
        <span>Parser</span>
        <select className="input" value={current.name} disabled={busy} onChange={(e) => onChange(e.target.value)}>
          {options.map((o) => (
            <option key={o.name} value={o.name} disabled={!o.available}>
              {LABELS[o.name] ?? o.name}
              {o.available ? '' : ' (not set up)'}
            </option>
          ))}
        </select>
      </label>
      <p className="hint">{current.description}</p>
      {unavailable.map((o) => (
        <p key={o.name} className="hint">
          {LABELS[o.name] ?? o.name}: {o.description}
        </p>
      ))}
    </section>
  )
}
