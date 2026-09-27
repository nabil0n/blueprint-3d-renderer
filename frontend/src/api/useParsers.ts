import { useEffect, useState } from 'react'
import { z } from 'zod'

const parserSchema = z.object({
  name: z.string(),
  available: z.boolean(),
  default: z.boolean(),
  description: z.string(),
})

export type ParserOption = z.infer<typeof parserSchema>

/** The parsers the backend knows, fetched once on mount; empty until known or if the backend can't say. */
export function useParsers(): readonly ParserOption[] {
  const [parsers, setParsers] = useState<readonly ParserOption[]>([])

  useEffect(() => {
    const controller = new AbortController()
    fetch('/api/parsers', { signal: controller.signal })
      .then(async (res) => {
        const parsed = z.array(parserSchema).safeParse(res.ok ? await res.json() : null)
        if (!parsed.success) throw new Error('Unexpected parser list from the backend.')
        setParsers(parsed.data)
      })
      .catch((err: unknown) => {
        if (!controller.signal.aborted) console.warn('Could not list parsers:', err)
      })
    return () => controller.abort()
  }, [])

  return parsers
}
