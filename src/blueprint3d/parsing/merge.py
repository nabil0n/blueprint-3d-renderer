"""Merging walls that are really two faces of one wall.

Plans that draw walls as outlines (two parallel lines) produce two thin walls side by side, and
boundary completion can add a wall next to one that already exists. Parallel walls of the same
height that overlap along their length and lie within `max_gap` of each other become one wall
spanning both faces; their openings are carried over, with overlapping ones combined.
"""

import math
from itertools import combinations

import numpy as np

from blueprint3d.parsing.geometry import PxOpening, PxWall

PARALLEL_TOLERANCE = math.sin(math.radians(3))
MIN_OVERLAP = 0.5
"""Of the shorter wall's length."""


def merge_parallel_walls(walls: list[PxWall], max_gap: float) -> list[PxWall]:
    current = list(walls)
    while True:
        merge = next(
            (
                (i, j, merged)
                for i, j in combinations(range(len(current)), 2)
                if (merged := _merge(current[i], current[j], max_gap)) is not None
            ),
            None,
        )
        if merge is None:
            return current
        i, j, merged = merge
        current = [w for k, w in enumerate(current) if k not in (i, j)] + [merged]


def _direction(wall: PxWall) -> tuple[np.ndarray, float]:
    start, end = np.array(wall.start), np.array(wall.end)
    length = float(np.linalg.norm(end - start))
    return ((end - start) / length if length else np.zeros(2)), length


def _merge(a: PxWall, b: PxWall, max_gap: float) -> PxWall | None:
    (da, la), (db, lb) = _direction(a), _direction(b)
    if not la or not lb or a.height_cm != b.height_cm or abs(da[0] * db[1] - da[1] * db[0]) > PARALLEL_TOLERANCE:
        return None
    base, u = (a, da) if la >= lb else (b, db)
    origin, n = np.array(base.start), np.array([-u[1], u[0]])

    def along(p) -> float:
        return float(np.dot(np.array(p) - origin, u))

    def across(p) -> float:
        return float(np.dot(np.array(p) - origin, n))

    ta = sorted((along(a.start), along(a.end)))
    tb = sorted((along(b.start), along(b.end)))
    if min(ta[1], tb[1]) - max(ta[0], tb[0]) < MIN_OVERLAP * min(la, lb):
        return None
    ca, cb = (across(a.start) + across(a.end)) / 2, (across(b.start) + across(b.end)) / 2
    if abs(ca - cb) > max_gap:
        return None

    lo = min(ca - a.thickness / 2, cb - b.thickness / 2)
    hi = max(ca + a.thickness / 2, cb + b.thickness / 2)
    t0, t1 = min(ta[0], tb[0]), max(ta[1], tb[1])
    centre = (lo + hi) / 2
    return PxWall(
        start=tuple(origin + u * t0 + n * centre),
        end=tuple(origin + u * t1 + n * centre),
        thickness=hi - lo,
        openings=_combined_openings([a, b], lambda p: along(p) - t0, t1 - t0),
        exterior=a.exterior or b.exterior,
        height_cm=a.height_cm,
    )


def _combined_openings(walls: list[PxWall], position, length: float) -> tuple[PxOpening, ...]:
    """Openings re-expressed along the merged wall; overlapping ones become one."""
    spans = []
    for wall in walls:
        direction, _ = _direction(wall)
        for o in wall.openings:
            centre = position(np.array(wall.start) + direction * o.offset)
            spans.append([max(0.0, centre - o.width / 2), min(length, centre + o.width / 2), o.kind == "window"])
    spans.sort()
    groups: list[list] = []
    for start, end, is_window in spans:
        if groups and start <= groups[-1][1]:
            groups[-1][1] = max(groups[-1][1], end)
            groups[-1][2] = groups[-1][2] and is_window
        else:
            groups.append([start, end, is_window])
    return tuple(
        PxOpening(offset=(s + e) / 2, width=e - s, kind="window" if w else "door") for s, e, w in groups if e > s
    )
