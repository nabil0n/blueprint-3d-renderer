import { useEffect, useRef, useState } from 'react'
import { parseFloorPlan, type ParseMeta } from '../api/parseFloorPlan'
import { parsePlan, type Plan } from '../plan/schema'

export interface LoadedPlan {
  readonly plan: Plan
  readonly source: string
  /** Bumped on every load so the scene remounts and reframes the camera. */
  readonly version: number
}

export interface ImageSource {
  readonly file: File
  /** Object URL for previewing the uploaded image. */
  readonly url: string
}

export interface PlanSource {
  readonly loaded: LoadedPlan
  readonly image: ImageSource | null
  readonly meta: ParseMeta | null
  readonly busy: boolean
  readonly error: string | null
  readonly uploadImage: (file: File) => void
  readonly reparse: (cmPerPx: number) => void
  readonly loadJson: (file: File) => void
}

export function usePlanSource(initial: LoadedPlan): PlanSource {
  const [loaded, setLoaded] = useState(initial)
  const [image, setImage] = useState<ImageSource | null>(null)
  const [meta, setMeta] = useState<ParseMeta | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const inFlight = useRef<AbortController | null>(null)

  useEffect(() => {
    if (!image) return
    return () => URL.revokeObjectURL(image.url)
  }, [image])

  useEffect(() => () => inFlight.current?.abort(), [])

  const show = (plan: Plan, source: string) => setLoaded((prev) => ({ plan, source, version: prev.version + 1 }))

  /** Runs one load at a time; starting another cancels the previous request. */
  const run = async (task: (signal: AbortSignal) => Promise<void>) => {
    inFlight.current?.abort()
    const controller = new AbortController()
    inFlight.current = controller
    setBusy(true)
    setError(null)
    try {
      await task(controller.signal)
    } catch (err) {
      if (!controller.signal.aborted) setError(err instanceof Error ? err.message : 'Something went wrong.')
    } finally {
      if (inFlight.current === controller) {
        inFlight.current = null
        setBusy(false)
      }
    }
  }

  const uploadImage = (file: File) =>
    void run(async (signal) => {
      const result = await parseFloorPlan(file, { signal })
      setImage({ file, url: URL.createObjectURL(file) })
      setMeta(result.meta)
      show(result.plan, file.name)
    })

  const reparse = (cmPerPx: number) => {
    if (!image) return
    void run(async (signal) => {
      const result = await parseFloorPlan(image.file, { cmPerPx, signal })
      setMeta(result.meta)
      show(result.plan, image.file.name)
    })
  }

  const loadJson = (file: File) =>
    void run(async () => {
      const plan = parsePlan(JSON.parse(await file.text()))
      setImage(null)
      setMeta(null)
      show(plan, file.name)
    })

  return { loaded, image, meta, busy, error, uploadImage, reparse, loadJson }
}
