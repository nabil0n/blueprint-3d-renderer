import { useEffect, useLayoutEffect, useRef, useState, type MouseEvent, type PointerEvent } from 'react'
import type { ImageSource } from '../app/usePlanSource'

interface OriginalViewerProps {
  readonly image: ImageSource
  readonly onClose: () => void
}

type Zoom = 'fit' | 'actual'

interface Size {
  readonly width: number
  readonly height: number
}

/** A point on the image as fractions of its width and height. */
interface Focus {
  readonly x: number
  readonly y: number
}

const CENTRE: Focus = { x: 0.5, y: 0.5 }

const ZOOM_LABELS: Record<Zoom, string> = { fit: 'Fit', actual: '100%' }

/** The uploaded plan next to the 3D view, fitted or at full size, for judging the parse by eye. */
export function OriginalViewer({ image, onClose }: OriginalViewerProps) {
  const [zoom, setZoom] = useState<Zoom>('fit')
  const [focus, setFocus] = useState<Focus>(CENTRE)
  const [size, setSize] = useState<Size | null>(null)
  const view = useRef<HTMLDivElement>(null)
  const pan = usePan()

  // Full size opens centred on the focus point rather than at the page's top-left corner.
  useLayoutEffect(() => {
    const el = view.current
    if (zoom !== 'actual' || !el || !size) return
    el.scrollLeft = focus.x * size.width - el.clientWidth / 2
    el.scrollTop = focus.y * size.height - el.clientHeight / 2
  }, [zoom, focus, size])

  const choose = (z: Zoom) => {
    setFocus(CENTRE)
    setZoom(z)
  }

  /** Clicking the fitted image zooms to full size at that spot. */
  const zoomTo = (event: MouseEvent<HTMLImageElement>) => {
    if (zoom !== 'fit') return
    const rect = event.currentTarget.getBoundingClientRect()
    setFocus({ x: (event.clientX - rect.left) / rect.width, y: (event.clientY - rect.top) / rect.height })
    setZoom('actual')
  }

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  return (
    <section className="original" aria-label="Original floor plan">
      <header className="original-bar">
        <div className="original-title">
          <strong title={image.file.name}>{image.file.name}</strong>
          {size && <span className="muted">{`${size.width} × ${size.height} px`}</span>}
        </div>
        <div className="segmented" role="group" aria-label="Zoom">
          {(Object.keys(ZOOM_LABELS) as Zoom[]).map((z) => (
            <button key={z} type="button" aria-pressed={zoom === z} onClick={() => choose(z)}>
              {ZOOM_LABELS[z]}
            </button>
          ))}
        </div>
        <button type="button" className="icon-button" aria-label="Close original" title="Close (Esc)" onClick={onClose}>
          ×
        </button>
      </header>
      <div className={`original-view is-${zoom}`} ref={view} {...(zoom === 'actual' ? pan : {})}>
        <img
          className={`is-${zoom}`}
          src={image.url}
          alt={`Original floor plan ${image.file.name}`}
          draggable={false}
          onClick={zoomTo}
          onLoad={(e) => setSize({ width: e.currentTarget.naturalWidth, height: e.currentTarget.naturalHeight })}
        />
      </div>
    </section>
  )
}

/** Drag-to-scroll for the full-size view. */
function usePan() {
  const drag = useRef<{ x: number; y: number; left: number; top: number } | null>(null)

  const onPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    const el = event.currentTarget
    if (event.button !== 0) return
    el.setPointerCapture(event.pointerId)
    drag.current = { x: event.clientX, y: event.clientY, left: el.scrollLeft, top: el.scrollTop }
  }
  const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {
    const el = event.currentTarget
    const start = drag.current
    if (!start) return
    el.scrollLeft = start.left - (event.clientX - start.x)
    el.scrollTop = start.top - (event.clientY - start.y)
  }
  const onPointerUp = () => {
    drag.current = null
  }

  return { onPointerDown, onPointerMove, onPointerUp, onPointerCancel: onPointerUp }
}
