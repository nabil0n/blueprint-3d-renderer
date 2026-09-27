"""Drawing a parse result on top of the source image, for eyeballing what the parser saw."""

import math

import cv2
import numpy as np

from blueprint3d.parsing.result import ParseResult
from blueprint3d.schema import Plan, Point

EXTERIOR = (40, 40, 210)
INTERIOR = (200, 90, 20)
LOW_WALL = (180, 60, 180)
DOOR = (40, 170, 40)
WINDOW = (0, 200, 230)
ROOM_ALPHA = 0.4
MARGIN_PX = 40
HEADER_PX = 34


def _to_image(result: ParseResult):
    ox, oy = result.meta.origin_px
    scale = 1 / result.meta.cm_per_px

    def convert(p: Point | tuple[float, float]) -> tuple[int, int]:
        x, y = (p.x, p.y) if isinstance(p, Point) else p
        return round(ox + x * scale), round(oy + y * scale)

    return convert


def _room_color(index: int) -> tuple[int, int, int]:
    hue = (index * 47) % 180
    hsv = np.uint8([[[hue, 90, 235]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def _draw_rooms(canvas: np.ndarray, plan: Plan, convert) -> np.ndarray:
    fill = canvas.copy()
    for i, room in enumerate(plan.rooms):
        cv2.fillPoly(fill, [np.array([convert(p) for p in room.polygon], np.int32)], _room_color(i))
    return cv2.addWeighted(fill, ROOM_ALPHA, canvas, 1 - ROOM_ALPHA, 0)


def _draw_walls(canvas: np.ndarray, plan: Plan, convert) -> None:
    walls = {w.id: w for w in plan.walls}
    for w in plan.walls:
        color = LOW_WALL if w.height < 200 else EXTERIOR if w.exterior else INTERIOR
        cv2.line(canvas, convert(w.start), convert(w.end), color, 2)
    for o in plan.openings:
        w = walls[o.wall_id]
        dx, dy = (w.end.x - w.start.x) / w.length, (w.end.y - w.start.y) / w.length
        a, b = o.offset - o.width / 2, o.offset + o.width / 2
        p0 = (w.start.x + dx * a, w.start.y + dy * a)
        p1 = (w.start.x + dx * b, w.start.y + dy * b)
        cv2.line(canvas, convert(p0), convert(p1), DOOR if o.kind == "door" else WINDOW, 5)


def draw_overlay(image: np.ndarray, result: ParseResult) -> np.ndarray:
    """The plan drawn over a faded copy of `image`, cropped to the plan with a margin."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    canvas = cv2.cvtColor((gray * 0.45 + 140).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    convert = _to_image(result)
    canvas = _draw_rooms(canvas, result.plan, convert)
    _draw_walls(canvas, result.plan, convert)

    points = [convert(p) for w in result.plan.walls for p in (w.start, w.end)]
    if not points:
        return canvas
    xs, ys = zip(*points, strict=True)
    height, width = canvas.shape[:2]
    x0, x1 = max(0, min(xs) - MARGIN_PX), min(width, max(xs) + MARGIN_PX)
    y0, y1 = max(0, min(ys) - MARGIN_PX), min(height, max(ys) + MARGIN_PX)
    return canvas[y0:y1, x0:x1]


def with_caption(image: np.ndarray, lines: list[str]) -> np.ndarray:
    header = np.full((HEADER_PX * len(lines), image.shape[1], 3), 255, np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(header, text, (8, HEADER_PX * i + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
    return np.vstack([header, image])


def contact_sheet(tiles: list[tuple[np.ndarray, list[str]]], columns: int = 3, tile_height: int = 700) -> np.ndarray:
    """Images scaled to the same height, captioned at that size, and laid out in a grid."""
    scaled = [
        with_caption(cv2.resize(img, (max(1, round(img.shape[1] * tile_height / img.shape[0])), tile_height)), caption)
        for img, caption in tiles
    ]
    cell_height = max(t.shape[0] for t in scaled)
    cell_width = max(t.shape[1] for t in scaled)
    rows = math.ceil(len(scaled) / columns)
    sheet = np.full((rows * cell_height, columns * cell_width, 3), 255, np.uint8)
    for i, tile in enumerate(scaled):
        r, c = divmod(i, columns)
        y, x = r * cell_height, c * cell_width
        sheet[y : y + tile.shape[0], x : x + tile.shape[1]] = tile
    return sheet
