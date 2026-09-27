"""Vectorising the wall mask into straight wall pieces."""

import math

import cv2
import numpy as np

from blueprint3d.parsing.geometry import Axis, AxisSegment, PxWall

MIN_ASPECT = 1.5
"""A wall piece must be at least this many times longer than it is thick."""
DIAGONAL_MIN_ASPECT = 3.0
MAX_SPREAD = 3.0
"""An axis piece spans at most this many times its thickness across its axis. A diagonal wall as
thick as the others survives the long-kernel opening as a sheared band that spans far more."""
COVER_PAD_PX = 2


def extract_axis_segments(mask: np.ndarray, min_run: float) -> list[AxisSegment]:
    """Horizontal and vertical wall pieces.

    Opening with a long 1-px line kernel keeps only runs at least `min_run` long in that direction,
    which removes walls running the other way (they are thinner than `min_run`).
    """
    run = max(3, round(min_run)) | 1  # odd, so the kernel anchor is centred and edges don't shift
    horizontal = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((1, run), np.uint8))
    vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((run, 1), np.uint8))
    return [*_segments(horizontal, "h"), *_segments(np.ascontiguousarray(vertical.T), "v")]


def _segments(mask: np.ndarray, axis: Axis) -> list[AxisSegment]:
    """Segments from a mask whose walls run horizontally (vertical walls arrive transposed)."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    segments = []
    for i in range(1, count):
        x, y, w, h = (int(v) for v in stats[i, :4])
        component = labels[y : y + h, x : x + w] == i
        column_heights = component.sum(axis=0)
        thickness = float(np.median(column_heights[column_heights > 0]))
        center = y + float(np.nonzero(component)[0].mean()) + 0.5
        segment = AxisSegment(axis=axis, center=center, start=float(x), end=float(x + w), thickness=thickness)
        if segment.length >= MIN_ASPECT * thickness and h <= MAX_SPREAD * thickness:
            segments.append(segment)
    return segments


def _cover(shape: tuple[int, ...], segments: list[AxisSegment]) -> np.ndarray:
    covered = np.zeros(shape, np.uint8)
    for s in segments:
        lo = round(s.center - s.thickness / 2) - COVER_PAD_PX
        hi = round(s.center + s.thickness / 2) + COVER_PAD_PX
        a0, a1 = round(s.start) - COVER_PAD_PX, round(s.end) + COVER_PAD_PX
        if s.axis == "h":
            cv2.rectangle(covered, (a0, lo), (a1, hi), 255, -1)
        else:
            cv2.rectangle(covered, (lo, a0), (hi, a1), 255, -1)
    return covered


def extract_diagonal_walls(mask: np.ndarray, segments: list[AxisSegment], min_thickness: float) -> list[PxWall]:
    """Walls the axis-aligned pass missed (angled walls), each fitted with a rotated rectangle."""
    residual = cv2.bitwise_and(mask, cv2.bitwise_not(_cover(mask.shape, segments)))
    residual = cv2.morphologyEx(residual, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(residual, connectivity=8)

    walls = []
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] < 3 * min_thickness**2:
            continue
        ys, xs = np.nonzero(labels == i)
        wall = _fit_wall(np.column_stack([xs, ys]).astype(np.float32), min_thickness)
        if wall is not None:
            walls.append(wall)
    return walls


def _fit_wall(points: np.ndarray, min_thickness: float) -> PxWall | None:
    (cx, cy), (w, h), angle = cv2.minAreaRect(points)
    thickness, length = min(w, h), max(w, h)
    if thickness < min_thickness / 2 or length < DIAGONAL_MIN_ASPECT * thickness:
        return None
    theta = math.radians(angle if w >= h else angle + 90)
    half = (length - thickness) / 2
    dx, dy = math.cos(theta) * half, math.sin(theta) * half
    return PxWall(start=(cx - dx, cy - dy), end=(cx + dx, cy + dy), thickness=thickness)
