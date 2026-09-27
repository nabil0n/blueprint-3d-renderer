"""Finding rooms as the enclosed free space between walls."""

from dataclasses import dataclass

import cv2
import numpy as np

from blueprint3d.parsing.geometry import PxPoint, WallRun

Rect = tuple[int, int, int, int]
Polygon = tuple[PxPoint, ...]


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
    """
    free = (barrier == 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    outside = np.isin(labels, border[border > 0])

    polygons, contours = [], []
    for i in range(1, count):
        if i in border or stats[i, cv2.CC_STAT_AREA] < min_area_px:
            continue
        room = _room_outline((labels == i).astype(np.uint8), min_inradius_px)
        if room is not None:
            polygons.append(room[0])
            contours.append(room[1])
    return RoomRegions(
        polygons=tuple(polygons), contours=tuple(contours), outside=outside, outside_drawing=outside
    )


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


def _room_outline(region: np.ndarray, min_inradius_px: float) -> tuple[Polygon, np.ndarray] | None:
    if cv2.distanceTransform(region, cv2.DIST_L2, 5).max() < min_inradius_px:
        return None
    found, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(found, key=cv2.contourArea)
    approx = cv2.approxPolyDP(contour, epsilon=max(2.0, 0.005 * cv2.arcLength(contour, True)), closed=True)
    if len(approx) < 3:
        return None
    return tuple((float(x), float(y)) for x, y in approx.reshape(-1, 2)), contour.reshape(-1, 2)
