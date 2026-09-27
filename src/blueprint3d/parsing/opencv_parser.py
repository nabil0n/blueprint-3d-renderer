"""Baseline parser using classical image processing. Works best on clean, black-walled plans.

Pipeline: binarise -> keep thick strokes (walls) -> crop to the plan -> axis-aligned wall pieces
(+ angled leftovers) -> merge collinear pieces, gaps become doors/windows -> estimate scale ->
rooms from enclosed free space -> mark walls bordering the outside as exterior -> complete room
outlines drawn with thin lines (angled windows/doors, balcony railings) -> Plan in cm.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from blueprint3d.parsing.boundaries import complete_boundaries, covered_mask
from blueprint3d.parsing.errors import ParseError
from blueprint3d.parsing.geometry import PxWall, WallRun
from blueprint3d.parsing.image_io import MAX_SIDE_PX, decode_image
from blueprint3d.parsing.merge import merge_parallel_walls
from blueprint3d.parsing.ocr import TextReader, read_text
from blueprint3d.parsing.openings import classify_gap, merge_collinear, run_to_wall
from blueprint3d.parsing.pipeline import (
    PlanArea,
    best_scale_bar,
    boundary_config,
    label_rooms_safely,
    locate_plan,
    parse_meta,
    room_kinds_and_names,
    scale_warnings,
)
from blueprint3d.parsing.plan_builder import build_plan, mark_exterior, windows_facing_outside
from blueprint3d.parsing.result import ParseResult
from blueprint3d.parsing.rooms import RoomRegions, find_rooms, gap_rects, merge_room_passes, outside_space
from blueprint3d.parsing.scale import estimate_scale, page_format_scale
from blueprint3d.parsing.segments import extract_axis_segments, extract_diagonal_walls
from blueprint3d.parsing.walls import thickness_stats

MIN_RUN_FACTOR = 1.1
MIN_RUN_EXTRA_PX = 2
"""Axis pieces must be a bit longer than the thickest wall, so walls crossing them are filtered out."""
MIN_GAP_FACTOR = 1.2
MAX_GAP_FACTOR = 8.0
"""Openings are between these multiples of the typical wall thickness."""
MIN_ROOM_M2 = 1.0
MIN_ROOM_HALF_WIDTH_CM = 35.0
GAP_PLUG_MARGIN_PX = 2
MAX_WALL_FACE_GAP_CM = 45.0
"""Parallel walls closer than this are taken as the two faces of one (outlined) wall."""


@dataclass(frozen=True)
class OpenCvParser:
    max_side: int = MAX_SIDE_PX
    name: str = "opencv"
    read_text: TextReader | None = read_text
    """OCR for room names; None skips naming."""

    def parse(self, data: bytes, cm_per_px: float | None = None) -> ParseResult:
        decoded = decode_image(data, self.max_side)
        area = locate_plan(decoded.gray)
        stats = thickness_stats(area.wall_mask)

        segments = extract_axis_segments(area.wall_mask, min_run=MIN_RUN_FACTOR * stats.maximum + MIN_RUN_EXTRA_PX)
        runs = merge_collinear(
            segments, min_gap=MIN_GAP_FACTOR * stats.typical, max_gap=MAX_GAP_FACTOR * stats.typical
        )
        axis_walls = [w for w in (_classified_wall(run, area.ink) for run in runs) if w is not None]
        diagonal_walls = extract_diagonal_walls(area.wall_mask, segments, area.min_thickness)
        if not axis_walls and not diagonal_walls:
            raise ParseError("No straight walls found in the plan.")

        user_scale = cm_per_px / decoded.resize_factor if cm_per_px is not None else None
        doors = [o.width for w in axis_walls for o in w.openings if o.kind == "door"]
        bar = best_scale_bar(area)
        page_height, page_width = decoded.gray.shape
        scale = estimate_scale(
            user_cm_per_px=user_scale,
            door_widths_px=doors,
            max_wall_thickness_px=stats.maximum,
            scale_bar_px_per_m=bar.px_per_step if bar else None,
            page_cm_per_px=page_format_scale(page_width, page_height),
        )

        regions = _find_rooms(area, runs, scale.cm_per_px)
        found_walls = [*axis_walls, *diagonal_walls]
        completion = complete_boundaries(
            area.ink,
            covered_mask(area.wall_mask, found_walls),
            regions.contours,
            regions.outside_drawing,
            boundary_config(stats, scale.cm_per_px),
        )
        outside = outside_space(regions, completion.balconies)
        walls = [windows_facing_outside(mark_exterior(w, outside), outside) for w in found_walls]
        labels, naming_warning = label_rooms_safely(self.read_text, area.gray, regions.polygons)
        kinds, names = room_kinds_and_names(labels, balconies=completion.balconies)
        merged = merge_parallel_walls([*walls, *completion.walls], max_gap=MAX_WALL_FACE_GAP_CM / scale.cm_per_px)
        plan = build_plan(merged, regions.polygons, scale.cm_per_px, kinds, names)

        warnings = [*scale_warnings(scale, len(plan.rooms)), *naming_warning]
        meta = parse_meta(self.name, decoded, area, scale, cm_per_px, warnings)
        return ParseResult(plan=plan, meta=meta)


def _classified_wall(run: WallRun, ink: np.ndarray) -> PxWall | None:
    """The run's gaps are windows when drawn lines run along them, doors otherwise."""
    return run_to_wall(run, [(gap, classify_gap(ink, run, gap)) for gap in run.gaps])


def _find_rooms(area: PlanArea, runs: list[WallRun], cm_per_px: float) -> RoomRegions:
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
