// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { ParserOption } from '../api/useParsers'
import { ParserPicker } from './ParserPicker'

const OPTIONS: ParserOption[] = [
  { name: 'opencv', available: true, default: true, description: 'Classical image processing.' },
  { name: 'cubicasa', available: false, default: false, description: 'Needs a one-time setup.' },
]

afterEach(cleanup)

describe('ParserPicker', () => {
  it('selects the default until a parser is chosen', () => {
    render(<ParserPicker options={OPTIONS} selected={null} busy={false} onChange={() => {}} />)
    expect((screen.getByRole('combobox') as HTMLSelectElement).value).toBe('opencv')
    expect(screen.getByText('Classical image processing.')).toBeTruthy()
  })

  it('disables parsers the backend cannot run, and says why', () => {
    render(<ParserPicker options={OPTIONS} selected={null} busy={false} onChange={() => {}} />)
    const learned = screen.getByRole('option', { name: /CubiCasa/ }) as HTMLOptionElement
    expect(learned.disabled).toBe(true)
    expect(screen.getByText(/Needs a one-time setup/)).toBeTruthy()
  })

  it('reports a new choice', () => {
    const onChange = vi.fn()
    const available = OPTIONS.map((o) => ({ ...o, available: true }))
    render(<ParserPicker options={available} selected="opencv" busy={false} onChange={onChange} />)
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'cubicasa' } })
    expect(onChange).toHaveBeenCalledWith('cubicasa')
  })

  it('shows nothing when there is no choice to make', () => {
    const { container } = render(<ParserPicker options={OPTIONS.slice(0, 1)} selected={null} busy={false} onChange={() => {}} />)
    expect(container.innerHTML).toBe('')
  })
})
