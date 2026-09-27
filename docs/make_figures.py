"""Render the README's pipeline figures from a synthetic showcase plan.

    uv run python docs/make_figures.py

Uses the synthetic drawing helpers from the tests, so no (copyrighted) real plan is needed.
Writes PNGs to docs/images/. The 3D screenshot (docs/images/dollhouse.png) is taken separately
from the running app.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from blueprint3d.evaluation.overlay import draw_overlay  # noqa: E402
from blueprint3d.parsing import opencv_parser as op  # noqa: E402
from blueprint3d.parsing.image_io import binarize, decode_image  # noqa: E402
from blueprint3d.parsing.openings import classify_gap, merge_collinear  # noqa: E402
from blueprint3d.parsing.pipeline import locate_plan  # noqa: E402
from blueprint3d.parsing.rooms import find_rooms, gap_rects  # noqa: E402
from blueprint3d.parsing.segments import extract_axis_segments  # noqa: E402
from blueprint3d.parsing.walls import thickness_stats  # noqa: E402
from tests.parsing.synthetic import (  # noqa: E402
    TRUE_CM_PER_PX,
    blank,
    draw_bay_and_balcony_plan,
    draw_tick_scale_bar,
    encode_png,
)

OUT = ROOT / "docs" / "images"
WIDTH = 760
DOOR = (40, 170, 40)
WINDOW = (0, 190, 240)
WALL = (40, 40, 40)
FAINT = 225


def showcase() -> np.ndarray:
    """The bay-and-balcony test plan, plus a coloured logo and a scale bar below it."""
    plan = draw_bay_and_balcony_plan()
    page = np.vstack([blank(1000, 110), plan, blank(1000, 170)])
    cv2.putText(page, "ACME Homes", (700, 95), cv2.FONT_HERSHEY_DUPLEX, 1.1, (60, 110, 20), 4, cv2.LINE_AA)
    draw_tick_scale_bar(page, 100, 1120, 100 / TRUE_CM_PER_PX)
    cv2.putText(page, "Two-room flat, bay window, balcony", (100, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return page


def save(name: str, image: np.ndarray) -> None:
    height = round(image.shape[0] * WIDTH / image.shape[1])
    resized = cv2.resize(image, (WIDTH, height), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", resized)
    assert ok
    (OUT / name).write_bytes(buf.tobytes())
    print("wrote", OUT / name)


def faint(mask: np.ndarray) -> np.ndarray:
    canvas = np.full((*mask.shape, 3), 255, np.uint8)
    canvas[mask > 0] = FAINT
    return canvas


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    page = showcase()
    save("1-input.png", page)

    decoded = decode_image(encode_png(page))
    area = locate_plan(decoded.gray)
    x0, y0, x1, y1 = area.box

    # 2. Ink: colour removed, dark ink black, light-gray lines gray, plan area outlined.
    dark = binarize(decoded.gray)
    ink_view = np.full((*dark.shape, 3), 255, np.uint8)
    ink_view[area.page_inks[1] > 0] = (150, 150, 150)
    ink_view[dark > 0] = (0, 0, 0)
    cv2.rectangle(ink_view, (x0, y0), (x1, y1), (0, 0, 220), 3)
    save("2-ink.png", ink_view)

    # 3. Walls: thick strokes over faint ink.
    walls_view = faint(area.ink)
    walls_view[area.wall_mask > 0] = WALL
    save("3-walls.png", walls_view)

    # 4. Wall runs and the gaps between their pieces, classified as doors or windows.
    stats = thickness_stats(area.wall_mask)
    segments = extract_axis_segments(area.wall_mask, op.MIN_RUN_FACTOR * stats.maximum + op.MIN_RUN_EXTRA_PX)
    runs = merge_collinear(segments, min_gap=op.MIN_GAP_FACTOR * stats.typical, max_gap=op.MAX_GAP_FACTOR * stats.typical)
    runs_view = faint(area.ink)
    for s in segments:
        lo, hi = round(s.center - s.thickness / 2), round(s.center + s.thickness / 2)
        a, b = round(s.start), round(s.end)
        rect = ((a, lo), (b, hi)) if s.axis == "h" else ((lo, a), (hi, b))
        cv2.rectangle(runs_view, *rect, WALL, -1)
    for run in runs:
        for gap, (gx0, gy0, gx1, gy1) in zip(run.gaps, gap_rects([run], 0), strict=True):
            colour = WINDOW if classify_gap(area.ink, run, gap) == "window" else DOOR
            cv2.rectangle(runs_view, (gx0, gy0), (gx1, gy1), colour, -1)
    save("4-openings.png", runs_view)

    # 5. The two room passes side by side.
    limits = {"min_area_px": 10_000 / TRUE_CM_PER_PX**2, "min_inradius_px": 35 / TRUE_CM_PER_PX}
    panels = []
    for base in (area.wall_mask, area.ink):
        barrier = base.copy()
        for gx0, gy0, gx1, gy1 in gap_rects(runs, 2):
            cv2.rectangle(barrier, (gx0, gy0), (gx1, gy1), 255, -1)
        regions = find_rooms(barrier, **limits)
        panel = np.full((*barrier.shape, 3), 255, np.uint8)
        panel[regions.outside] = (235, 225, 210)
        for i, polygon in enumerate(regions.polygons):
            colour = [(150, 210, 150), (150, 170, 240), (230, 170, 150)][i % 3]
            cv2.fillPoly(panel, [np.array(polygon, np.int32)], colour)
        panel[barrier > 0] = WALL
        panels.append(panel)
    gutter = np.full((panels[0].shape[0], 30, 3), 255, np.uint8)
    save("5-rooms.png", np.hstack([panels[0], gutter, panels[1]]))

    # 6. The finished plan drawn back onto the image.
    result = op.OpenCvParser().parse(encode_png(page))
    save("6-result.png", draw_overlay(page, result))


if __name__ == "__main__":
    main()
