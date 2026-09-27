"""Completing room outlines that no thick wall explains.

Thick-stroke detection misses walls drawn with thin lines: windows and doors in angled walls, and
balcony railings. Every stretch of a room outline that is not next to a known wall is explained by
what lies beyond it:

- another room, across thin lines           -> wall with a door along the chord
- outside, in a room mostly bounded that way -> balcony: every such stretch is a railing
                                                (thin, half height), even if drawn double
- outside, elsewhere, behind 2+ parallel lines -> exterior wall with a window (glazing)
- outside, elsewhere, behind a single line     -> left alone (e.g. a door leaf drawn outside)

Probes stop at known walls, so furniture drawn against a wall is never mistaken for one.
"""

import math
from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np

from blueprint3d.parsing.geometry import PxOpening, PxWall
from blueprint3d.parsing.openings import count_line_groups

OUTSIDE = -1
COVER_PAD_PX = 3
STRAIGHTNESS_PX = 3.0
"""Uncovered outline stretches are split into straight pieces with this tolerance."""
PROBE_SPACING_PX = 8
GLAZING_CHUNK_PX = 16
"""Lines are counted per short chunk, so glazing that is not exactly parallel still registers."""
GLAZING_LINE_COVERAGE = 0.75
MIN_GLAZING_LINES = 2
MIN_GLAZED_SHARE = 0.3
OPENING_MARGIN_PX = 1.0
BALCONY_OUTSIDE_SHARE = 0.25
"""A room is a balcony if at least this share of its outline faces outside across thin lines.
Loggias (recessed, railing on one side) sit near 30 %; bay windows ~10 %, door leaves ~5 %."""


@dataclass(frozen=True)
class BoundaryConfig:
    """All lengths in working-image pixels."""

    wall_thickness: float
    door_wall_thickness: float
    railing_thickness: float
    railing_height_cm: float
    min_piece: float
    door_range: tuple[float, float]
    probe_reach: float


@dataclass(frozen=True)
class Completion:
    walls: tuple[PxWall, ...]
    balconies: frozenset[int]
    """Indices of rooms (in the contours passed in) bounded by a railing."""


@dataclass(frozen=True)
class _Piece:
    start: np.ndarray
    end: np.ndarray
    outward: np.ndarray
    neighbor: int | None
    """Room index, OUTSIDE, or None when unknown."""

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start))


def covered_mask(wall_mask: np.ndarray, walls: list[PxWall], pad: int = COVER_PAD_PX) -> np.ndarray:
    """Boolean mask of everything already explained by walls: thick strokes plus the known walls
    drawn at full length (so their openings count as covered too)."""
    covered = wall_mask.copy()
    for w in walls:
        start, end = np.array(w.start), np.array(w.end)
        length = float(np.linalg.norm(end - start))
        direction = (end - start) / length if length else np.zeros(2)
        extend = direction * w.thickness / 2
        p0, p1 = (tuple(int(round(v)) for v in p) for p in (start - extend, end + extend))
        cv2.line(covered, p0, p1, 255, max(1, round(w.thickness)))
    size = 2 * pad + 1
    return cv2.dilate(covered, np.ones((size, size), np.uint8)) > 0


def complete_boundaries(
    ink: np.ndarray,
    covered: np.ndarray,
    contours: tuple[np.ndarray, ...],
    outside: np.ndarray,
    config: BoundaryConfig,
) -> Completion:
    """`outside` must be free space beyond all drawn lines (not beyond thick walls only)."""
    labels = _room_labels(ink.shape, contours)
    thick_ink = cv2.dilate(ink, np.ones((3, 3), np.uint8))
    walls: list[PxWall] = []
    balconies: set[int] = set()
    doors: dict[frozenset[int], PxWall] = {}

    for index, contour in enumerate(contours):
        groups = _neighbor_groups(index, contour, covered, labels, outside, config)
        facing_outside = [p for g in groups if g[0].neighbor == OUTSIDE for p in g]
        # Hull perimeter: door leaves and scan staircases would inflate the raw outline's length.
        perimeter = cv2.arcLength(cv2.convexHull(contour.reshape(-1, 1, 2)), True)
        is_balcony = sum(p.length for p in facing_outside) >= BALCONY_OUTSIDE_SHARE * perimeter
        make_wall = _railing if is_balcony else _window_wall
        for wall in (make_wall(p, thick_ink, config) for p in facing_outside):
            if wall is not None:
                walls.append(wall)
                if is_balcony:
                    balconies.add(index)
        for group in groups:
            neighbor = group[0].neighbor
            if neighbor is not None and neighbor != OUTSIDE:
                door = _door_wall(group, config)
                key = frozenset({index, neighbor})
                if door is not None and (key not in doors or _length(door) > _length(doors[key])):
                    doors[key] = door
    return Completion(walls=(*walls, *doors.values()), balconies=frozenset(balconies))


def _length(wall: PxWall) -> float:
    return math.dist(wall.start, wall.end)


def _room_labels(shape: tuple[int, ...], contours: tuple[np.ndarray, ...]) -> np.ndarray:
    """Room index + 1 per pixel, 0 elsewhere."""
    labels = np.zeros(shape[:2], np.uint16)
    for i, contour in enumerate(contours):
        cv2.drawContours(labels, [contour.reshape(-1, 1, 2)], -1, i + 1, thickness=cv2.FILLED)
    return labels


def _uncovered_runs(flags: np.ndarray) -> list[np.ndarray]:
    """Index runs where `flags` is True along a closed outline."""
    if flags.all():
        return [np.arange(len(flags))]
    start = int(np.argmin(flags))  # a covered point, so no run wraps around the start
    order = (np.arange(len(flags)) + start) % len(flags)
    runs, current = [], []
    for i in order:
        if flags[i]:
            current.append(i)
        elif current:
            runs.append(np.array(current))
            current = []
    if current:
        runs.append(np.array(current))
    return runs


def _neighbor_groups(
    index: int,
    contour: np.ndarray,
    covered: np.ndarray,
    labels: np.ndarray,
    outside: np.ndarray,
    config: BoundaryConfig,
) -> list[list[_Piece]]:
    """Straight pieces of uncovered outline, grouped while they face the same neighbour."""
    flags = ~covered[contour[:, 1], contour[:, 0]]
    groups: list[list[_Piece]] = []
    for run in _uncovered_runs(flags):
        points = contour[run].reshape(-1, 1, 2).astype(np.int32)
        vertices = cv2.approxPolyDP(points, STRAIGHTNESS_PX, closed=False).reshape(-1, 2).astype(float)
        run_groups: list[list[_Piece]] = []
        for a, b in zip(vertices, vertices[1:], strict=False):
            piece = _probe_piece(index, a, b, covered, labels, outside, config)
            if piece is None:
                continue
            if run_groups and run_groups[-1][-1].neighbor == piece.neighbor:
                run_groups[-1].append(piece)
            else:
                run_groups.append([piece])
        groups.extend(run_groups)
    return groups


def _probe_piece(
    index: int,
    a: np.ndarray,
    b: np.ndarray,
    covered: np.ndarray,
    labels: np.ndarray,
    outside: np.ndarray,
    config: BoundaryConfig,
) -> _Piece | None:
    length = float(np.linalg.norm(b - a))
    if length < 2:
        return None
    direction = (b - a) / length
    outward = np.array([-direction[1], direction[0]])
    if _label_at(labels, (a + b) / 2 + 3 * outward) == index + 1:
        outward = -outward

    samples = max(1, int(length // PROBE_SPACING_PX))
    found = [
        _probe(labels, covered, outside, a + (b - a) * (k + 0.5) / samples, outward, index + 1, config.probe_reach)
        for k in range(samples)
    ]
    votes = Counter(n for n in found if n is not None)
    neighbor, count = votes.most_common(1)[0] if votes else (None, 0)
    return _Piece(start=a, end=b, outward=outward, neighbor=neighbor if count * 2 >= samples else None)


def _label_at(labels: np.ndarray, point: np.ndarray) -> int:
    x, y = round(point[0]), round(point[1])
    height, width = labels.shape
    return int(labels[y, x]) if 0 <= x < width and 0 <= y < height else 0


def _probe(
    labels: np.ndarray,
    covered: np.ndarray,
    outside: np.ndarray,
    origin: np.ndarray,
    outward: np.ndarray,
    self_label: int,
    reach: float,
) -> int | None:
    """Walk outward until reaching another room or the outside; known walls block the view."""
    height, width = labels.shape
    for step in range(1, int(reach) + 1):
        x, y = round(origin[0] + outward[0] * step), round(origin[1] + outward[1] * step)
        if not (0 <= x < width and 0 <= y < height):
            return OUTSIDE
        if covered[y, x]:
            return None
        label = int(labels[y, x])
        if label == self_label:
            continue
        if label > 0:
            return label - 1
        if outside[y, x]:
            return OUTSIDE
    return None


def _window_wall(piece: _Piece, thick_ink: np.ndarray, config: BoundaryConfig) -> PxWall | None:
    """An exterior wall with a window where the stretch is glazed; None if it is not."""
    length = piece.length
    if length < config.min_piece:
        return None
    glazed = [c >= MIN_GLAZING_LINES for c in _glazing_line_counts(thick_ink, piece, config.wall_thickness)]
    if not glazed or sum(glazed) / len(glazed) < MIN_GLAZED_SHARE:
        return None
    first = glazed.index(True)
    last = len(glazed) - 1 - glazed[::-1].index(True)
    u0 = max(OPENING_MARGIN_PX, first * GLAZING_CHUNK_PX)
    u1 = min(length - OPENING_MARGIN_PX, (last + 1) * GLAZING_CHUNK_PX)
    shift = piece.outward * config.wall_thickness / 2
    window = PxOpening(offset=(u0 + u1) / 2, width=u1 - u0, kind="window")
    return PxWall(
        start=tuple(piece.start + shift),
        end=tuple(piece.end + shift),
        thickness=config.wall_thickness,
        openings=(window,) if u1 > u0 else (),
        exterior=True,
    )


def _railing(piece: _Piece, _thick_ink: np.ndarray, config: BoundaryConfig) -> PxWall | None:
    if piece.length < config.min_piece:
        return None
    shift = piece.outward * (config.railing_thickness / 2 + 1)
    return PxWall(
        start=tuple(piece.start + shift),
        end=tuple(piece.end + shift),
        thickness=config.railing_thickness,
        exterior=True,
        height_cm=config.railing_height_cm,
    )


def _glazing_line_counts(thick_ink: np.ndarray, piece: _Piece, depth: float) -> list[int]:
    """Lines running along the piece, just beyond it, counted per chunk."""
    length = int(piece.length)
    direction = (piece.end - piece.start) / piece.length
    u = np.arange(length, dtype=np.float32)
    v = np.arange(-2, int(depth) + 4, dtype=np.float32)
    xs = piece.start[0] + u[None, :] * direction[0] + v[:, None] * piece.outward[0]
    ys = piece.start[1] + u[None, :] * direction[1] + v[:, None] * piece.outward[1]
    band = cv2.remap(thick_ink, xs.astype(np.float32), ys.astype(np.float32), cv2.INTER_NEAREST, borderValue=0)

    counts = []
    for c0 in range(0, length - GLAZING_CHUNK_PX // 2, GLAZING_CHUNK_PX):
        chunk = band[:, c0 : c0 + GLAZING_CHUNK_PX] > 0
        counts.append(count_line_groups(chunk.mean(axis=1) >= GLAZING_LINE_COVERAGE))
    return counts


def _door_wall(group: list[_Piece], config: BoundaryConfig) -> PxWall | None:
    """The doorway is the chord across the stretch, from one wall end to the other."""
    start, end = group[0].start, group[-1].end
    length = float(np.linalg.norm(end - start))
    low, high = config.door_range
    if not low <= length <= high:
        return None
    door = PxOpening(offset=length / 2, width=length - 2 * OPENING_MARGIN_PX, kind="door")
    return PxWall(start=tuple(start), end=tuple(end), thickness=config.door_wall_thickness, openings=(door,))
