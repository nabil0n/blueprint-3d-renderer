import { CanvasTexture, SRGBColorSpace } from 'three'
import { polygonArea } from '../geometry/polygon'
import { CM_TO_M } from '../geometry/units'
import type { Room } from '../plan/schema'
import { LABEL_COLORS, ROOM_LABELS } from './colors'

export interface LabelText {
  readonly title: string
  readonly subtitle: string
}

export function roomLabelText(room: Room): LabelText {
  const areaM2 = polygonArea(room.polygon) * CM_TO_M * CM_TO_M
  return { title: room.name ?? ROOM_LABELS[room.kind], subtitle: `${areaM2.toFixed(1)} m²` }
}

export interface LabelTexture {
  readonly texture: CanvasTexture
  /** Width / height, for sizing the sprite. */
  readonly aspect: number
}

const PIXEL_RATIO = 2
const PAD_X = 16
/** Left edge mark, like a leader tick on a drawing. */
const MARK_WIDTH = 5
const TITLE_FONT = '600 24px "IBM Plex Mono", ui-monospace, monospace'
const SUBTITLE_FONT = '20px "IBM Plex Mono", ui-monospace, monospace'
const TITLE_BASELINE = 33
const SUBTITLE_BASELINE = 60
const HEIGHT = 76

/** Draws a label card to a canvas texture. Returns null where 2D canvas is unavailable. */
export function makeLabelTexture({ title, subtitle }: LabelText): LabelTexture | null {
  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')
  if (!ctx) return null

  ctx.font = TITLE_FONT
  const titleWidth = ctx.measureText(title).width
  ctx.font = SUBTITLE_FONT
  const width = Math.ceil(Math.max(titleWidth, ctx.measureText(subtitle).width) + 2 * PAD_X + MARK_WIDTH)

  canvas.width = width * PIXEL_RATIO
  canvas.height = HEIGHT * PIXEL_RATIO
  ctx.scale(PIXEL_RATIO, PIXEL_RATIO)

  ctx.fillStyle = LABEL_COLORS.card
  ctx.fillRect(0, 0, width, HEIGHT)
  ctx.fillStyle = LABEL_COLORS.mark
  ctx.fillRect(0, 0, MARK_WIDTH, HEIGHT)

  const textX = MARK_WIDTH + PAD_X
  ctx.fillStyle = LABEL_COLORS.title
  ctx.font = TITLE_FONT
  ctx.fillText(title, textX, TITLE_BASELINE)
  ctx.fillStyle = LABEL_COLORS.subtitle
  ctx.font = SUBTITLE_FONT
  ctx.fillText(subtitle, textX, SUBTITLE_BASELINE)

  const texture = new CanvasTexture(canvas)
  texture.colorSpace = SRGBColorSpace
  return { texture, aspect: width / HEIGHT }
}
