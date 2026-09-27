"""Baseline parser using classical image processing. Works best on clean, black-walled plans.

Pipeline: binarise -> keep thick strokes (walls) -> crop to the plan -> axis-aligned wall pieces
(+ angled leftovers) -> merge collinear pieces, gaps become doors/windows -> estimate scale ->
rooms from enclosed free space -> mark walls bordering the outside as exterior -> complete room
outlines drawn with thin lines (angled windows/doors, balcony railings) -> Plan in cm.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from blueprint3d.parsing.boundaries import BoundaryConfig, complete_boundaries, covered_mask
from blueprint3d.parsing.errors import ParseError
from blueprint3d.parsing.geometry import PxOpening, PxWall, WallRun
from blueprint3d.parsing.image_io import MAX_SIDE_PX, binarize, decode_image
from blueprint3d.parsing.openings import classify_gap, merge_collinear
from blueprint3d.parsing.plan_builder import build_plan, mark_exterior, windows_facing_outside
from blueprint3d.parsing.result import ParseMeta, ParseResult
from blueprint3d.parsing.rooms import RoomRegions, find_rooms, gap_rects, merge_room_passes
from blueprint3d.parsing.scale import ScaleEstimate, estimate_scale
from blueprint3d.parsing.segments import extract_axis_segments, extract_diagonal_walls
from blueprint3d.parsing.walls import ThicknessStats, crop_box, expand_to_ink, extract_wall_mask, thickness_stats
from blueprint3d.schema import DEFAULT_WALL_HEIGHT_CM, RoomKind

MIN_RUN_FACTOR = 1.1
MIN_RUN_EXTRA_PX = 2
"""Axis pieces must be a bit longer than the thickest wall, so walls crossing them are filtered out."""
MIN_GAP_FACTOR = 1.2
MAX_GAP_FACTOR = 8.0
"""Openings are between these multiples of the typical wall thickness."""
MIN_ROOM_M2 = 1.0
MIN_ROOM_HALF_WIDTH_CM = 35.0
GAP_PLUG_MARGIN_PX = 2
LIGHT_INK_LEVEL = 230
"""Gray levels below this count as ink when looking for thin details (windows are often light gray)."""
RAILING_THICKNESS_CM = 5.0
MIN_BOUNDARY_PIECE_CM = 40.0
MIN_DOOR_CM = 55.0
MAX_DOOR_CM = 200.0
PROBE_REACH_FACTOR = 3.0
"""How far (in max wall thicknesses) to look beyond a room outline for what bounds it."""


@dataclass(frozen=True)
class _PlanArea:
    ink: np.ndarray
    """Any ink including light gray lines, 255 = ink."""
    wall_mask: np.ndarray
    min_thickness: int


@dataclass(frozen=True)
class OpenCvParser:
    max_side: int = MAX_SIDE_PX
    name: str = "opencv"

    def parse(self, data: bytes, cm_per_px: float | None = None) -> ParseResult:
        decoded = decode_image(data, self.max_side)
        area = _locate_plan(decoded.gray)
        stats = thickness_stats(area.wall_mask)

        segments = extract_axis_segments(area.wall_mask, min_run=MIN_RUN_FACTOR * stats.maximum + MIN_RUN_EXTRA_PX)
        runs = merge_collinear(
            segments, min_gap=MIN_GAP_FACTOR * stats.typical, max_gap=MAX_GAP_FACTOR * stats.typical
        )
        axis_walls = [w for w in (_run_to_wall(run, area.ink) for run in runs) if w is not None]
        diagonal_walls = extract_diagonal_walls(area.wall_mask, segments, area.min_thickness)
        if not axis_walls and not diagonal_walls:
            raise ParseError("No straight walls found in the plan.")

        user_scale = cm_per_px / decoded.resize_factor if cm_per_px is not None else None
        doors = [o.width for w in axis_walls for o in w.openings if o.kind == "door"]
        scale = estimate_scale(user_cm_per_px=user_scale, door_widths_px=doors, max_wall_thickness_px=stats.maximum)

        regions = _find_rooms(area, runs, scale.cm_per_px)
        walls = [
            windows_facing_outside(mark_exterior(w, regions.outside), regions.outside)
            for w in (*axis_walls, *diagonal_walls)
        ]
        completion = complete_boundaries(
            area.ink,
            covered_mask(area.wall_mask, walls),
            regions.contours,
            regions.outside_drawing,
            _boundary_config(stats, scale.cm_per_px),
        )
        kinds: dict[int, RoomKind] = dict.fromkeys(completion.balconies, "balcony")
        plan = build_plan([*walls, *completion.walls], regions.polygons, scale.cm_per_px, kinds)

        width, height = decoded.original_size
        meta = ParseMeta(
            parser=self.name,
            cm_per_px=cm_per_px if cm_per_px is not None else scale.cm_per_px * decoded.resize_factor,
            scale_source=scale.source,
            scale_detail=scale.detail,
            image_width=width,
            image_height=height,
            warnings=_warnings(scale, len(plan.rooms)),
        )
        return ParseResult(plan=plan, meta=meta)


def _locate_plan(gray: np.ndarray) -> _PlanArea:
    """Find the plan with a rough wall mask, then re-derive walls from strokes inside it only:
    page text (bold headings, legends) would otherwise skew the thin/thick split."""
    binary = binarize(gray)
    rough = extract_wall_mask(binary)
    x0, y0, x1, y1 = expand_to_ink(crop_box(rough.mask, rough.min_thickness), binary, rough.mask)
    plan_binary = binary[y0:y1, x0:x1]
    walls = extract_wall_mask(plan_binary)
    ink = np.where(gray[y0:y1, x0:x1] < LIGHT_INK_LEVEL, 255, 0).astype(np.uint8)
    return _PlanArea(ink=ink, wall_mask=walls.mask, min_thickness=walls.min_thickness)


def _run_to_wall(run: WallRun, binary: np.ndarray) -> PxWall | None:
    """Wall centre line between the run's ends, inset by half the thickness (the renderer extends
    walls by the same amount to close corners)."""
    half = run.thickness / 2
    a0, a1 = run.start + half, run.end - half
    if a1 - a0 < 1:
        return None
    if run.axis == "h":
        start, end = (a0, run.center), (a1, run.center)
    else:
        start, end = (run.center, a0), (run.center, a1)
    openings = tuple(
        PxOpening(offset=(g0 + g1) / 2 - a0, width=g1 - g0, kind=classify_gap(binary, run, (g0, g1)))
        for g0, g1 in run.gaps
        if g0 >= a0 and g1 <= a1
    )
    return PxWall(start=start, end=end, thickness=run.thickness, openings=openings)


def _find_rooms(area: _PlanArea, runs: list[WallRun], cm_per_px: float) -> RoomRegions:
    def plugged(barrier: np.ndarray) -> np.ndarray:
        result = barrier.copy()
        for x0, y0, x1, y1 in gap_rects(runs, margin=GAP_PLUG_MARGIN_PX):
            cv2.rectangle(result, (x0, y0), (x1, y1), 255, -1)
        return result

    limits = {
        "min_area_px": MIN_ROOM_M2 * 10_000 / cm_per_px**2,
        "min_inradius_px": MIN_ROOM_HALF_WIDTH_CM / cm_per_px,
    }
    return merge_room_passes(
        find_rooms(plugged(area.wall_mask), **limits),
        find_rooms(plugged(area.ink), **limits),
    )


def _boundary_config(stats: ThicknessStats, cm_per_px: float) -> BoundaryConfig:
    return BoundaryConfig(
        wall_thickness=stats.maximum,
        door_wall_thickness=stats.typical,
        railing_thickness=RAILING_THICKNESS_CM / cm_per_px,
        railing_height_cm=DEFAULT_WALL_HEIGHT_CM / 2,
        min_piece=MIN_BOUNDARY_PIECE_CM / cm_per_px,
        door_range=(MIN_DOOR_CM / cm_per_px, MAX_DOOR_CM / cm_per_px),
        probe_reach=PROBE_REACH_FACTOR * stats.maximum,
    )


def _warnings(scale: ScaleEstimate, room_count: int) -> list[str]:
    warnings = []
    if scale.source != "user":
        warnings.append(f"Scale is estimated. {scale.detail} Enter the real scale if sizes look off.")
    if room_count == 0:
        warnings.append("No enclosed rooms found; floors are missing.")
    return warnings
