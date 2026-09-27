"""Separating walls (thick strokes) from everything else, and locating the plan on the page."""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from blueprint3d.parsing.errors import ParseError

MIN_WALL_HALF_WIDTH_PX = 2.5
"""Lower bound for the thin/thick split, so hairline-only drawings don't turn text into walls."""
MIN_SIGNIFICANT_SHARE = 0.005
MIN_SIGNIFICANT_COUNT = 3
"""Histogram bins below these (share of ridge pixels, absolute) are treated as noise."""
WIDTH_JUMP_RATIO = 1.6
WIDTH_JUMP_MIN_PX = 3


@dataclass(frozen=True)
class WallMask:
    mask: np.ndarray
    """uint8, 255 where a thick stroke (wall) is."""
    min_thickness: int
    """Strokes at least this thick (px) count as walls."""


@dataclass(frozen=True)
class ThicknessStats:
    typical: float
    maximum: float


def _ridge_half_widths(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance-transform values on stroke centre lines: about half of each stroke's width."""
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    ridge = (dist > 0) & (dist >= cv2.dilate(dist, np.ones((3, 3), np.uint8)))
    return dist, ridge


def extract_wall_mask(binary: np.ndarray) -> WallMask:
    dist, ridge = _ridge_half_widths(binary)
    widths = 2 * dist[ridge]
    if widths.size == 0:
        raise ParseError("No dark strokes found in the image.")

    # Odd kernel: an even one has an off-centre anchor and shifts the mask by a pixel. Rounding
    # down keeps walls exactly at the threshold; thin strokes sit well below it anyway.
    threshold = max(3, _wall_width_threshold(widths))
    kernel_size = threshold if threshold % 2 else threshold - 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    if not mask.any():
        raise ParseError("No walls found. The plan may be drawn with thin lines only.")
    return WallMask(mask=mask, min_thickness=kernel_size)


def _wall_width_threshold(widths: np.ndarray) -> int:
    """Smallest stroke width (px) that counts as a wall.

    Plans typically show a cluster of thin strokes (text, furniture, symbols) and then one or
    more wall widths. Cut at the first clear jump in the width histogram; Otsu is only a fallback,
    because with several wall widths it tends to split between walls instead.
    """
    counts = np.bincount(np.round(widths).astype(int))
    floor = max(MIN_SIGNIFICANT_COUNT, MIN_SIGNIFICANT_SHARE * counts.sum())
    present = [w for w in range(1, len(counts)) if counts[w] >= floor]
    for lo, hi in zip(present, present[1:], strict=False):
        if hi >= WIDTH_JUMP_RATIO * lo and hi - lo >= WIDTH_JUMP_MIN_PX:
            return math.ceil(math.sqrt(lo * hi))

    scaled = np.clip(widths * 5, 0, 255).astype(np.uint8).reshape(-1, 1)
    otsu, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return round(max(otsu / 5, 2 * MIN_WALL_HALF_WIDTH_PX))


def thickness_stats(wall_mask: np.ndarray) -> ThicknessStats:
    dist, ridge = _ridge_half_widths(wall_mask)
    widths = 2 * dist[ridge]
    if widths.size == 0:
        raise ParseError("No walls found inside the plan area.")
    return ThicknessStats(typical=float(np.median(widths)), maximum=float(np.percentile(widths, 95)))


Box = tuple[int, int, int, int]

MIN_CLUSTER_SHARE = 0.1
"""A wall cluster joins the plan if it has at least this share of the largest cluster's wall pixels..."""
MAX_CLUSTER_DISTANCE = 0.25
"""...and lies within this fraction of the plan's size from it."""


def crop_box(wall_mask: np.ndarray, min_thickness: int) -> Box:
    """Bounding box (x0, y0, x1, y1) of the plan's walls, with a margin.

    Doors and windows split the walls into several clusters. Starting from the largest, nearby
    clusters of substantial size are added; titles, legend icons and orientation maps elsewhere
    on the page are small or far away and stay out.
    """
    reach = 4 * min_thickness
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(wall_mask, np.ones((reach, reach), np.uint8)))
    if count <= 1:
        raise ParseError("No walls found.")

    wall_pixels = np.bincount(labels[wall_mask > 0], minlength=count)
    wall_pixels[0] = 0
    order = [int(i) for i in np.argsort(wall_pixels)[::-1] if wall_pixels[i] > 0]
    box = _stat_box(stats[order[0]])
    candidates = [i for i in order[1:] if wall_pixels[i] >= MIN_CLUSTER_SHARE * wall_pixels[order[0]]]

    grown = True
    while grown:
        near = [i for i in candidates if _box_distance(box, _stat_box(stats[i])) <= MAX_CLUSTER_DISTANCE * _box_size(box)]
        candidates = [i for i in candidates if i not in near]
        box = _union([box, *(_stat_box(stats[i]) for i in near)])
        grown = bool(near)

    # The dilated boxes overshoot by reach/2; trim back, then add the margin.
    pad = 3 * min_thickness - reach // 2
    height, width = wall_mask.shape
    x0, y0, x1, y1 = box
    return max(0, x0 - pad), max(0, y0 - pad), min(width, x1 + pad), min(height, y1 + pad)


INK_PAD_PX = 3


def expand_to_ink(box: Box, binary: np.ndarray, wall_mask: np.ndarray) -> Box:
    """Grow the box to include thin drawings connected to the plan's walls, e.g. balcony outlines.
    Ink that surrounds the whole plan (a page frame) is ignored."""
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    x0, y0, x1, y1 = box
    in_box = np.zeros(binary.shape, bool)
    in_box[y0:y1, x0:x1] = True
    connected = np.unique(labels[(wall_mask > 0) & in_box])
    attached = [b for b in (_stat_box(stats[i]) for i in connected if i > 0) if not _encloses(b, box)]
    if not attached:
        return box
    ux0, uy0, ux1, uy1 = _union([box, *attached])
    height, width = binary.shape
    pad = [INK_PAD_PX if grew else 0 for grew in (ux0 < x0, uy0 < y0, ux1 > x1, uy1 > y1)]
    return max(0, ux0 - pad[0]), max(0, uy0 - pad[1]), min(width, ux1 + pad[2]), min(height, uy1 + pad[3])


def _encloses(outer: Box, inner: Box) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and outer[2] > inner[2] and outer[3] > inner[3]


def _stat_box(stat: np.ndarray) -> Box:
    x, y, w, h = (int(v) for v in stat[:4])
    return x, y, x + w, y + h


def _union(boxes: list[Box]) -> Box:
    return min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)


def _box_distance(a: Box, b: Box) -> int:
    dx = max(0, b[0] - a[2], a[0] - b[2])
    dy = max(0, b[1] - a[3], a[1] - b[3])
    return max(dx, dy)


def _box_size(box: Box) -> int:
    return max(box[2] - box[0], box[3] - box[1])
