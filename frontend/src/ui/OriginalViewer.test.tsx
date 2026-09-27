// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { ImageSource } from '../app/usePlanSource'
import { OriginalViewer } from './OriginalViewer'

const IMAGE: ImageSource = { file: new File(['x'], 'plan.png', { type: 'image/png' }), url: 'blob:plan' }

afterEach(cleanup)

const zoomButton = (name: string) => screen.getByRole('button', { name })

describe('OriginalViewer', () => {
  it('shows the uploaded image fitted to the pane', () => {
    render(<OriginalViewer image={IMAGE} onClose={() => {}} />)
    const img = screen.getByRole('img', { name: /plan\.png/ }) as HTMLImageElement
    expect(img.src).toBe('blob:plan')
    expect(img.className).toContain('is-fit')
    expect(zoomButton('Fit').getAttribute('aria-pressed')).toBe('true')
  })

  it('switches to full size and back', () => {
    render(<OriginalViewer image={IMAGE} onClose={() => {}} />)
    fireEvent.click(zoomButton('100%'))
    expect(screen.getByRole('img').className).toContain('is-actual')
    expect(zoomButton('100%').getAttribute('aria-pressed')).toBe('true')
    fireEvent.click(zoomButton('Fit'))
    expect(screen.getByRole('img').className).toContain('is-fit')
  })

  it('zooms to full size where the fitted image is clicked', () => {
    render(<OriginalViewer image={IMAGE} onClose={() => {}} />)
    fireEvent.click(screen.getByRole('img'))
    expect(screen.getByRole('img').className).toContain('is-actual')
  })

  it('reports the image size once it has loaded', () => {
    render(<OriginalViewer image={IMAGE} onClose={() => {}} />)
    const img = screen.getByRole('img') as HTMLImageElement
    Object.defineProperty(img, 'naturalWidth', { value: 1200 })
    Object.defineProperty(img, 'naturalHeight', { value: 850 })
    fireEvent.load(img)
    expect(screen.getByText('1200 × 850 px')).toBeTruthy()
  })

  it('closes from its button and from Escape', () => {
    const onClose = vi.fn()
    render(<OriginalViewer image={IMAGE} onClose={onClose} />)
    fireEvent.click(screen.getByRole('button', { name: 'Close original' }))
    fireEvent.keyDown(window, { key: 'Escape' })
    expect(onClose).toHaveBeenCalledTimes(2)
  })
})
