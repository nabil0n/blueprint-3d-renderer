"""Reading the printed scale bar: a long horizontal line or band whose marks repeat regularly.

Both styles seen on rental plans reduce to the same signal. Tick bars have short vertical marks on
a line; checker bars have a band whose fill alternates per step. Either way there are vertical
edges at every step, so we collect edge positions along the bar and look for the smallest spacing
that repeats several times in a row. Only the tallest marks count, so finer subdivisions are ignored.
"""

from dataclasses import dataclass

import cv2
import numpy as np

Box = tuple[int, int, int, int]

MIN_BAR_LENGTH = 0.15
"""Of the page width."""
MAX_BAR_HEIGHT = 0.012
"""Of the page height; taller things are drawings, not bars."""
BAND_PX = 12
"""Rows above and below the bar searched for marks (digit labels usually sit further away)."""
MIN_EDGE_ROWS = 4
MAJOR_MARK_SHARE = 0.6
EVENT_MERGE_PX = 3
MIN_STEP = 0.015
"""Smallest step, as a fraction of the page's long side (1 m at 1:100 on A4 is ~3.4 %)."""
MIN_STEPS = 3
STEP_TOLERANCE = 0.02


@dataclass(frozen=True)
class ScaleBar:
    px_per_step: float
    steps: int
    y: int
    x0: int
    x1: int


def detect_scale_bar(binary: np.ndarray, exclude: Box) -> ScaleBar | None:
    """`binary` has ink as 255. `exclude` (x0, y0, x1, y1) is the plan area, which is ignored."""
    height, width = binary.shape
    page = binary.copy()
    ex0, ey0, ex1, ey1 = exclude
    page[ey0:ey1, ex0:ex1] = 0

    run = max(3, round(width * MIN_BAR_LENGTH)) | 1
    lines = cv2.morphologyEx(page, cv2.MORPH_OPEN, np.ones((1, run), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)

    bars = []
    for i in range(1, count):
        x, y, w, h = (int(v) for v in stats[i, :4])
        if h > max(3, MAX_BAR_HEIGHT * height):
            continue
        events = _mark_positions(page, x, y, w, h)
        found = _regular_step(events, min_step=MIN_STEP * max(height, width), max_step=w / MIN_STEPS)
        if found is not None:
            step, steps = found
            bars.append(ScaleBar(px_per_step=step, steps=steps, y=y, x0=x, x1=x + w))
    return max(bars, key=lambda b: b.steps * b.px_per_step, default=None)


def _mark_positions(page: np.ndarray, x: int, y: int, w: int, h: int) -> list[float]:
    """Centres of vertical edges crossing the band around the bar."""
    r0, r1 = max(0, y - BAND_PX), min(page.shape[0], y + h + BAND_PX)
    c0, c1 = max(0, x - EVENT_MERGE_PX), min(page.shape[1], x + w + EVENT_MERGE_PX)
    band = page[r0:r1, c0:c1] > 0
    edges = (band[:, 1:] != band[:, :-1]).sum(axis=0)
    # Major marks are the tallest; subdivision ticks are shorter and would blur the spacing.
    # The bar's own ends span the whole band, so they are left out when judging mark height.
    interior = edges[2 * EVENT_MERGE_PX : -2 * EVENT_MERGE_PX]
    strong = max(MIN_EDGE_ROWS, MAJOR_MARK_SHARE * int(interior.max(initial=0)))
    columns = np.nonzero(edges >= strong)[0] + c0 + 0.5

    events: list[list[float]] = []
    for col in columns:
        if events and col - events[-1][-1] <= EVENT_MERGE_PX:
            events[-1].append(float(col))
        else:
            events.append([float(col)])
    return [sum(group) / len(group) for group in events]


def _regular_step(events: list[float], min_step: float, max_step: float) -> tuple[float, int] | None:
    """Smallest spacing d >= min_step with a chain of MIN_STEPS+ consecutive marks d apart.
    Returns the refined spacing and the chain's number of steps."""
    marks = np.array(events)
    candidates = sorted(
        {marks[j] - marks[i] for i in range(len(marks)) for j in range(i + 1, len(marks))}
        & {d for d in np.diff(marks) if min_step <= d <= max_step}
    ) if len(marks) > 1 else []
    for d in candidates:
        chain = _longest_chain(marks, d)
        if len(chain) - 1 >= MIN_STEPS:
            return (chain[-1] - chain[0]) / (len(chain) - 1), len(chain) - 1
    return None


def _longest_chain(marks: np.ndarray, step: float) -> list[float]:
    tolerance = max(2.0, STEP_TOLERANCE * step)
    best: list[float] = []
    for start in marks:
        chain = [float(start)]
        while True:
            target = chain[-1] + step
            nearest = marks[np.argmin(np.abs(marks - target))]
            if abs(nearest - target) > tolerance:
                break
            chain.append(float(nearest))
        best = max(best, chain, key=len)
    return best
