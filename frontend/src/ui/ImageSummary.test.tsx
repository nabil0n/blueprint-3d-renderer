// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { ParseMeta } from '../api/parseFloorPlan'
import type { ImageSource } from '../app/usePlanSource'
import { ImageSummary } from './ImageSummary'

const IMAGE: ImageSource = { file: new File(['x'], 'plan.png', { type: 'image/png' }), url: 'blob:plan' }
const META: ParseMeta = {
  parser: 'opencv',
  cm_per_px: 1.5,
  scale_source: 'scale_bar',
  scale_detail: '',
  image_width: 800,
  image_height: 600,
  origin_px: [0, 0],
  warnings: [],
}

afterEach(cleanup)

const renderSummary = (props: Partial<Parameters<typeof ImageSummary>[0]> = {}) =>
  render(
    <ImageSummary
      image={IMAGE}
      meta={META}
      busy={false}
      comparing={false}
      onRescale={() => {}}
      onToggleOriginal={() => {}}
      {...props}
    />,
  )

describe('ImageSummary', () => {
  it('opens the original from the thumbnail', () => {
    const onToggleOriginal = vi.fn()
    renderSummary({ onToggleOriginal })
    const toggle = screen.getByRole('button', { name: /Compare with original/ })
    expect(toggle.getAttribute('aria-pressed')).toBe('false')
    fireEvent.click(toggle)
    expect(onToggleOriginal).toHaveBeenCalledOnce()
  })

  it('offers to hide the original while comparing', () => {
    renderSummary({ comparing: true })
    expect(screen.getByRole('button', { name: /Hide original/ }).getAttribute('aria-pressed')).toBe('true')
  })

  it('applies a valid scale', () => {
    const onRescale = vi.fn()
    renderSummary({ onRescale })
    fireEvent.change(screen.getByLabelText(/Scale/), { target: { value: '2.25' } })
    fireEvent.click(screen.getByRole('button', { name: 'Apply' }))
    expect(onRescale).toHaveBeenCalledWith(2.25)
  })
})
