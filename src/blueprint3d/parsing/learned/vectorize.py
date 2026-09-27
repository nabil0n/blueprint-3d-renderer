"""Raster to vector: the model's class maps into walls with openings, rooms and railings (in pixels).

The model marks doors and windows as icons *in* the wall, where the wall class is interrupted. So a
wall is traced through its openings (wall + icon pixels as one solid stroke) and the icon stretches
along it become the openings, their kind taken from the model. A gap with no icon is an open
passage. Rooms are the enclosed spaces between walls, openings and railings.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from blueprint3d.parsing.errors import ParseError
from blueprint3d.parsing.geometry import Gap, PxWall, WallRun
from blueprint3d.parsing.learned.segmentation import Segmentation
from blueprint3d.parsing.openings import OpeningKind, count_line_groups, merge_collinear, run_to_wall
from blueprint3d.parsing.rooms import RoomRegions, find_rooms, gap_rects
from blueprint3d.parsing.segments import extract_axis_segments, extract_diagonal_walls
from blueprint3d.parsing.walls import ThicknessStats, thickness_stats
from blueprint3d.schema import DEFAULT_WALL_HEIGHT_CM, RoomKind

# As in the OpenCV parser, so both are judged on the same geometry rules.
MIN_RUN_FACTOR = 1.1
MIN_RUN_EXTRA_PX = 2
MIN_GAP_FACTOR = 1.2
MAX_GAP_FACTOR = 8.0
MIN_ROOM_M2 = 1.0
MIN_ROOM_HALF_WIDTH_CM = 35.0
GAP_PLUG_MARGIN_PX = 2
RAILING_THICKNESS_CM = 5.0
RAILING_HEIGHT_CM = DEFAULT_WALL_HEIGHT_CM / 2

ICON_COVERAGE = 0.3
"""A position along a wall is an opening if an icon covers this share of the wall's thickness."""
DIAGONAL_MIN_THICKNESS_SHARE = 0.75
RAILING_MIN_RUN_FACTOR = 2.0
"""Railing pieces must be at least this many typical wall thicknesses long."""
MIN_KIND_SHARE = 0.4
"""A room takes a class's kind only if that class covers this share of its floor."""

ROOM_KINDS: dict[str, RoomKind | None] = {
    "Kitchen": "kitchen",
    "Living Room": "living_room",
    "Bed Room": "bedroom",
    "Bath": "bathroom",
    "Entry": "hallway",
    "Storage": "closet",
    "Outdoor": "balcony",
    "Garage": "other",
    "Undefined": None,
}
"""Model room classes -> our kinds. None: the model saw a room but not what kind."""


@dataclass(frozen=True)
class WallLayout:
    walls: tuple[PxWall, ...]
    runs: tuple[WallRun, ...]
    stats: ThicknessStats
    barrier: np.ndarray
    """uint8, 255 where movement is blocked: walls, their openings and railings."""


@dataclass(frozen=True)
class RoomLayout:
    regions: RoomRegions
    kinds: dict[int, RoomKind]
    """The model's guess per room index, where it is clear."""
    balconies: frozenset[int]
    railings: tuple[PxWall, ...]


def vectorize_walls(seg: Segmentation) -> WallLayout:
    wall = _mask(seg.room_mask("Wall"))
    door, window = seg.icon_mask("Door"), seg.icon_mask("Window")
    if not wall.any():
        raise ParseError("The model found no walls in the plan.")
    stats = thickness_stats(wall)
    solid = cv2.bitwise_or(wall, _mask(door | window))

    segments = extract_axis_segments(solid, min_run=MIN_RUN_FACTOR * stats.maximum + MIN_RUN_EXTRA_PX)
    runs = merge_collinear(
        segments, min_gap=MIN_GAP_FACTOR * stats.typical, max_gap=MAX_GAP_FACTOR * stats.typical
    )
    min_opening = MIN_GAP_FACTOR * stats.typical
    axis_walls = [run_to_wall(run, _openings(run, door, window, min_opening)) for run in runs]
    diagonal = extract_diagonal_walls(solid, segments, DIAGONAL_MIN_THICKNESS_SHARE * stats.typical)
    barrier = cv2.bitwise_or(solid, _mask(seg.room_mask("Railing")))
    walls = (*(w for w in axis_walls if w is not None), *diagonal)
    return WallLayout(walls=walls, runs=tuple(runs), stats=stats, barrier=barrier)


def vectorize_rooms(seg: Segmentation, walls: WallLayout, cm_per_px: float) -> RoomLayout:
    barrier = walls.barrier.copy()
    for x0, y0, x1, y1 in gap_rects(list(walls.runs), margin=GAP_PLUG_MARGIN_PX):
        cv2.rectangle(barrier, (x0, y0), (x1, y1), 255, -1)
    regions = find_rooms(
        barrier,
        min_area_px=MIN_ROOM_M2 * 10_000 / cm_per_px**2,
        min_inradius_px=MIN_ROOM_HALF_WIDTH_CM / cm_per_px,
    )
    guesses = {i: _guess_kind(seg, polygon) for i, polygon in enumerate(regions.polygons)}
    kinds = {i: kind for i, kind in guesses.items() if kind is not None}
    return RoomLayout(
        regions=regions,
        kinds=kinds,
        balconies=frozenset(i for i, kind in kinds.items() if kind == "balcony"),
        railings=_railings(seg, walls.stats, cm_per_px),
    )


def _mask(flags: np.ndarray) -> np.ndarray:
    return np.where(flags, 255, 0).astype(np.uint8)


def _openings(
    run: WallRun, door: np.ndarray, window: np.ndarray, min_length: float
) -> list[tuple[Gap, OpeningKind]]:
    """Icon stretches along the run, plus its bare gaps as open passages (doors)."""
    doors, windows = (_coverage(run, icon) for icon in (door, window))
    found: list[tuple[Gap, OpeningKind]] = []
    for lo, hi in _true_runs(doors | windows):
        if hi - lo >= min_length:
            kind: OpeningKind = "window" if windows[lo:hi].sum() > doors[lo:hi].sum() else "door"
            found.append(((run.start + lo, run.start + hi), kind))
    passages = [(gap, "door") for gap in run.gaps if not any(_overlaps(gap, g) for g, _ in found)]
    return sorted([*found, *passages])


def _coverage(run: WallRun, icon: np.ndarray) -> np.ndarray:
    """Per position along the run: is it covered by the icon across the wall's thickness?"""
    horizontal = icon if run.axis == "h" else icon.T
    lo, hi = round(run.center - run.thickness / 2), round(run.center + run.thickness / 2)
    band = horizontal[max(0, lo) : hi + 1, round(run.start) : round(run.end)]
    if band.size == 0:
        return np.zeros(max(0, round(run.end) - round(run.start)), bool)
    return band.mean(axis=0) >= ICON_COVERAGE


def _true_runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) of each run of consecutive True values."""
    if count_line_groups(flags) == 0:
        return []
    edges = np.diff(np.concatenate(([0], flags.astype(np.int8), [0])))
    return list(zip(np.nonzero(edges == 1)[0].tolist(), np.nonzero(edges == -1)[0].tolist(), strict=True))


def _overlaps(a: Gap, b: Gap) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _guess_kind(seg: Segmentation, polygon: tuple[tuple[float, float], ...]) -> RoomKind | None:
    inside = np.zeros(seg.rooms.shape, np.uint8)
    cv2.fillPoly(inside, [np.array(polygon, np.int32)], 1)
    classes = seg.rooms[inside > 0]
    if classes.size == 0:
        return None
    counts = np.bincount(classes, minlength=len(seg.room_classes))
    candidates = [(counts[i], name) for i, name in enumerate(seg.room_classes) if name in ROOM_KINDS]
    count, name = max(candidates)
    return ROOM_KINDS[name] if count >= MIN_KIND_SHARE * classes.size else None


def _railings(seg: Segmentation, stats: ThicknessStats, cm_per_px: float) -> tuple[PxWall, ...]:
    """Straight railing pieces as low, thin walls (drawn along the railing's centre line)."""
    railing = _mask(seg.room_mask("Railing"))
    segments = extract_axis_segments(railing, min_run=RAILING_MIN_RUN_FACTOR * stats.typical)
    walls = []
    for s in segments:
        if s.axis == "h":
            start, end = (s.start, s.center), (s.end, s.center)
        else:
            start, end = (s.center, s.start), (s.center, s.end)
        walls.append(
            PxWall(
                start=start,
                end=end,
                thickness=RAILING_THICKNESS_CM / cm_per_px,
                exterior=True,
                height_cm=RAILING_HEIGHT_CM,
            )
        )
    return tuple(walls)
