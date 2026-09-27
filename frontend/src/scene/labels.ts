import { CanvasTexture, SRGBColorSpace } from 'three'
import { polygonArea } from '../geometry/polygon'
import { CM_TO_M } from '../geometry/units'
import type { Room } from '../plan/schema'
import { ROOM_LABELS } from './colors'

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
const PAD_X = 14
const TITLE_FONT = '600 26px system-ui, "Segoe UI", sans-serif'
const SUBTITLE_FONT = '22px system-ui, "Segoe UI", sans-serif'
const TITLE_BASELINE = 34
const SUBTITLE_BASELINE = 62
const HEIGHT = 76
const RADIUS = 10

/** Draws a label card to a canvas texture. Returns null where 2D canvas is unavailable. */
export function makeLabelTexture({ title, subtitle }: LabelText): LabelTexture | null {
  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')
  if (!ctx) return null

  ctx.font = TITLE_FONT
  const titleWidth = ctx.measureText(title).width
  ctx.font = SUBTITLE_FONT
  const width = Math.ceil(Math.max(titleWidth, ctx.measureText(subtitle).width) + 2 * PAD_X)

  canvas.width = width * PIXEL_RATIO
  canvas.height = HEIGHT * PIXEL_RATIO
  ctx.scale(PIXEL_RATIO, PIXEL_RATIO)

  ctx.fillStyle = 'rgba(255, 255, 255, 0.9)'
  ctx.beginPath()
  ctx.roundRect(0, 0, width, HEIGHT, RADIUS)
  ctx.fill()

  ctx.textAlign = 'center'
  ctx.fillStyle = '#2c2b29'
  ctx.font = TITLE_FONT
  ctx.fillText(title, width / 2, TITLE_BASELINE)
  ctx.fillStyle = '#77736c'
  ctx.font = SUBTITLE_FONT
  ctx.fillText(subtitle, width / 2, SUBTITLE_BASELINE)

  const texture = new CanvasTexture(canvas)
  texture.colorSpace = SRGBColorSpace
  return { texture, aspect: width / HEIGHT }
}
