"""Finding rooms as the enclosed free space between walls."""

from dataclasses import dataclass

import cv2
import numpy as np

from blueprint3d.parsing.geometry import PxPoint, WallRun

Rect = tuple[int, int, int, int]
Polygon = tuple[PxPoint, ...]

ABSORB_REACH_PX = 3
"""Pockets look this far for a room: across lines up to ~2 px, never across walls (5+ px)."""
FILL_LINES_PX = 5


@dataclass(frozen=True)
class RoomRegions:
    polygons: tuple[Polygon, ...]
    """Simplified outline per room, for the plan."""
    contours: tuple[np.ndarray, ...]
    """Full-detail outline per room (N x 2 int, every boundary pixel), aligned with `polygons`."""
    outside: np.ndarray
    """Boolean mask of free space connected to the image border (outside the apartment)."""
    outside_drawing: np.ndarray
    """Like `outside`, but beyond all drawn lines (so a balcony is not part of it)."""


def gap_rects(runs: list[WallRun], margin: int) -> list[Rect]:
    """Rectangles (x0, y0, x1, y1) that plug each opening, so rooms don't leak through doors."""
    rects = []
    for run in runs:
        lo = round(run.center - run.thickness / 2) - margin
        hi = round(run.center + run.thickness / 2) + margin
        for g0, g1 in run.gaps:
            a0, a1 = round(g0), round(g1)
            rects.append((a0, lo, a1, hi) if run.axis == "h" else (lo, a0, hi, a1))
    return rects


def find_rooms(barrier: np.ndarray, *, min_area_px: float, min_inradius_px: float) -> RoomRegions:
    """`barrier` is 255 wherever movement is blocked (walls plus plugged openings).

    Enclosed regions that are big enough, and wide enough somewhere to stand in, become rooms.
    Smaller enclosed pockets (cupboards, counter strips, door-swing wedges) join the room they
    open onto across thin lines, so floors have no holes under the furniture.
    """
    free = (barrier == 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    outside = np.isin(labels, border[border > 0])
    enclosed = [i for i in range(1, count) if i not in border]

    rooms = [i for i in enclosed if _is_room(labels, stats[i], i, min_area_px, min_inradius_px)]
    members: dict[int, list[int]] = {r: [r] for r in rooms}
    for i in enclosed:
        if i not in members:
            owner = _neighbouring_room(labels, stats[i], i, members)
            if owner is not None:
                members[owner].append(i)

    polygons, contours = [], []
    for r in rooms:
        outline = _room_outline(np.isin(labels, members[r]).astype(np.uint8))
        if outline is not None:
            polygons.append(outline[0])
            contours.append(outline[1])
    return RoomRegions(
        polygons=tuple(polygons), contours=tuple(contours), outside=outside, outside_drawing=outside
    )


def _bbox(stat: np.ndarray, pad: int, shape: tuple[int, ...]) -> tuple[slice, slice]:
    x, y, w, h = (int(v) for v in stat[:4])
    return slice(max(0, y - pad), min(shape[0], y + h + pad)), slice(max(0, x - pad), min(shape[1], x + w + pad))


def _is_room(labels: np.ndarray, stat: np.ndarray, index: int, min_area_px: float, min_inradius_px: float) -> bool:
    if stat[cv2.CC_STAT_AREA] < min_area_px:
        return False
    region = (labels[_bbox(stat, 1, labels.shape)] == index).astype(np.uint8)
    return bool(cv2.distanceTransform(region, cv2.DIST_L2, 5).max() >= min_inradius_px)


def _neighbouring_room(labels: np.ndarray, stat: np.ndarray, index: int, rooms: dict[int, list[int]]) -> int | None:
    """The room sharing the most boundary with a pocket, looking across thin lines only."""
    window = labels[_bbox(stat, ABSORB_REACH_PX + 1, labels.shape)]
    pocket = (window == index).astype(np.uint8)
    ring = cv2.dilate(pocket, np.ones((3, 3), np.uint8), iterations=ABSORB_REACH_PX) > 0
    neighbours = window[ring & (window != index)]
    candidates = neighbours[np.isin(neighbours, list(rooms))]
    if candidates.size == 0:
        return None
    values, counts = np.unique(candidates, return_counts=True)
    return int(values[np.argmax(counts)])


def merge_room_passes(primary: RoomRegions, fallback: RoomRegions) -> RoomRegions:
    """Combine a pass over thick walls only (clean rooms, unaffected by door arcs and cupboards)
    with a pass over all strokes (also closes spaces bounded by thin lines, e.g. angled windows
    or balcony outlines). Fallback rooms are used only where the primary pass saw outside space.
    `outside` stays the primary pass's: beyond the thick walls, so balconies count as outside."""
    height, width = primary.outside.shape

    def in_primary_outside(polygon: Polygon) -> bool:
        mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(mask, [np.array(polygon, np.int32)], 1)
        inside = mask > 0
        return bool(inside.any() and primary.outside[inside].mean() > 0.5)

    extra = [i for i, p in enumerate(fallback.polygons) if in_primary_outside(p)]
    return RoomRegions(
        polygons=(*primary.polygons, *(fallback.polygons[i] for i in extra)),
        contours=(*primary.contours, *(fallback.contours[i] for i in extra)),
        outside=primary.outside,
        outside_drawing=fallback.outside,
    )


def _room_outline(region: np.ndarray) -> tuple[Polygon, np.ndarray] | None:
    # Swallow the thin lines between a room and the pockets it absorbed.
    region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, np.ones((FILL_LINES_PX, FILL_LINES_PX), np.uint8))
    found, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(found, key=cv2.contourArea)
    approx = cv2.approxPolyDP(contour, epsilon=max(2.0, 0.005 * cv2.arcLength(contour, True)), closed=True)
    if len(approx) < 3:
        return None
    return tuple((float(x), float(y)) for x, y in approx.reshape(-1, 2)), contour.reshape(-1, 2)
