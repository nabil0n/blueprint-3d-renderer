interface ViewControlsProps {
  readonly cutHeight: number
  readonly maxHeight: number
  readonly showLabels: boolean
  readonly onCutHeightChange: (value: number) => void
  readonly onShowLabelsChange: (value: boolean) => void
}

export function ViewControls({ cutHeight, maxHeight, showLabels, onCutHeightChange, onShowLabelsChange }: ViewControlsProps) {
  return (
    <section className="panel-section">
      <h2>View</h2>
      <label className="field">
        <span>
          Wall cut <output>{cutHeight >= maxHeight ? 'full height' : `${cutHeight} cm`}</output>
        </span>
        <input
          type="range"
          min={0}
          max={maxHeight}
          step={5}
          value={cutHeight}
          onChange={(e) => onCutHeightChange(Number(e.target.value))}
        />
      </label>

      <label className="toggle">
        <input type="checkbox" checked={showLabels} onChange={(e) => onShowLabelsChange(e.target.checked)} />
        Room labels
      </label>
    </section>
  )
}
