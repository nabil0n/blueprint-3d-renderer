"""Parser backed by CubiCasa5K's pretrained segmentation model, with our own raster-to-vector step.

Pipeline: find the plan on the page (shared with the OpenCV parser) -> run the model on the plan
at the wall thickness it was trained on -> walls, openings (the model's doors and windows) ->
scale -> rooms, balconies and railings -> exterior walls -> printed room names -> Plan in cm.
"""

from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

from blueprint3d.parsing.boundaries import complete_boundaries, covered_mask
from blueprint3d.parsing.errors import ParseError
from blueprint3d.parsing.image_io import MAX_SIDE_PX, decode_image
from blueprint3d.parsing.learned.segmentation import DEFAULT_MODEL_DIR, CubiCasaModel, SegmentationModel
from blueprint3d.parsing.learned.vectorize import vectorize_rooms, vectorize_walls
from blueprint3d.parsing.merge import merge_parallel_walls
from blueprint3d.parsing.ocr import TextReader, read_text
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
from blueprint3d.parsing.rooms import outside_space
from blueprint3d.parsing.scale import estimate_scale, page_format_scale
from blueprint3d.parsing.walls import thickness_stats

TARGET_WALL_PX = 10.0
"""Walls are scaled to about this thickness before the model sees them; tuned on the evaluation."""
MODEL_SCALE_RANGE = (0.3, 2.0)
MAX_WALL_FACE_GAP_CM = 45.0
"""As in the OpenCV parser: parallel walls this close are the two faces of one wall."""


@cache
def _load_model(model_dir: Path) -> CubiCasaModel:
    """Loaded once per folder: reading the 70 MB model takes about a second."""
    return CubiCasaModel(model_dir)


@dataclass(frozen=True)
class LearnedParser:
    model_dir: Path = DEFAULT_MODEL_DIR
    model: SegmentationModel | None = field(default=None, compare=False)
    """Overrides the model in `model_dir` (for tests)."""
    read_text: TextReader | None = read_text
    """OCR for room names; None skips naming."""
    max_side: int = MAX_SIDE_PX
    name: str = "cubicasa"

    def parse(self, data: bytes, cm_per_px: float | None = None) -> ParseResult:
        decoded = decode_image(data, self.max_side)
        area = locate_plan(decoded.gray)
        model = self.model or _load_model(self.model_dir)
        seg = model.segment(area.gray, scale=_model_scale(area))
        layout = vectorize_walls(seg)
        if not layout.walls:
            raise ParseError("The model found no straight walls in the plan.")

        bar = best_scale_bar(area)
        page_height, page_width = decoded.gray.shape
        scale = estimate_scale(
            user_cm_per_px=cm_per_px / decoded.resize_factor if cm_per_px is not None else None,
            door_widths_px=[o.width for w in layout.walls for o in w.openings if o.kind == "door"],
            max_wall_thickness_px=layout.stats.maximum,
            scale_bar_px_per_m=bar.px_per_step if bar else None,
            page_cm_per_px=page_format_scale(page_width, page_height),
        )

        rooms = vectorize_rooms(seg, layout, scale.cm_per_px, ink=area.ink)
        # Outlines no wall explains (thin-line balconies, angled glazing): completed as in the OpenCV parser.
        completion = complete_boundaries(
            area.ink,
            covered_mask(layout.barrier, list(layout.walls)),
            rooms.regions.contours,
            rooms.regions.outside_drawing,
            boundary_config(layout.stats, scale.cm_per_px),
        )
        balconies = rooms.balconies | completion.balconies
        outside = outside_space(rooms.regions, balconies)
        walls = [windows_facing_outside(mark_exterior(w, outside), outside) for w in layout.walls]
        merged = merge_parallel_walls(
            [*walls, *rooms.railings, *completion.walls], max_gap=MAX_WALL_FACE_GAP_CM / scale.cm_per_px
        )
        labels, naming_warning = label_rooms_safely(self.read_text, area.gray, rooms.regions.polygons)
        kinds, names = room_kinds_and_names(labels, guessed=rooms.kinds, balconies=balconies)
        plan = build_plan(merged, rooms.regions.polygons, scale.cm_per_px, kinds, names)

        warnings = [*scale_warnings(scale, len(plan.rooms)), *naming_warning]
        return ParseResult(plan=plan, meta=parse_meta(self.name, decoded, area, scale, cm_per_px, warnings))


def _model_scale(area: PlanArea) -> float:
    """Resize factor that brings the plan's walls to the thickness the model knows best."""
    try:
        typical = thickness_stats(area.wall_mask).typical
    except ParseError:
        return 1.0
    low, high = MODEL_SCALE_RANGE
    return min(high, max(low, TARGET_WALL_PX / typical))
