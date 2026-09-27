"""Estimating centimetres per pixel when the user has not calibrated the plan.

Priority: user > printed scale bar > page format (A4 at 1:100) > door widths > wall thickness.
The scale bar and page format rest on assumptions (1 m per bar step, 1:100 printing), so they are
only used when the door estimate, if any, roughly agrees.
"""

import math
import statistics
from dataclasses import dataclass
from typing import Literal

DOOR_WIDTH_CM = 80.0
"""Typical door opening as drawn in residential plans (wall gap for an 80 cm door)."""
EXTERIOR_WALL_CM = 30.0
"""Typical thickness of an apartment's exterior wall as drawn."""
SCALE_BAR_STEP_CM = 100.0
"""Scale bars on rental plans are marked in whole metres."""
A4_LONG_SIDE_CM = 29.7
PAGE_SCALE = 100
"""Rental plans are overwhelmingly printed at 1:100."""
A4_ASPECT_TOLERANCE = 0.02
MAX_DOOR_DISAGREEMENT = 1.6
"""An assumption is dropped if the door estimate differs from it by more than this factor."""

ScaleSource = Literal["user", "scale_bar", "page_format", "doors", "wall_thickness"]


@dataclass(frozen=True)
class ScaleEstimate:
    cm_per_px: float
    source: ScaleSource
    detail: str


def page_format_scale(width_px: int, height_px: int) -> float | None:
    """cm per px if the image has A4 proportions, assuming it was printed at 1:100."""
    long_side, short_side = max(width_px, height_px), min(width_px, height_px)
    if abs(long_side / short_side / math.sqrt(2) - 1) > A4_ASPECT_TOLERANCE:
        return None
    return A4_LONG_SIDE_CM * PAGE_SCALE / long_side


def _agrees(candidate: float, doors: float | None) -> bool:
    return doors is None or 1 / MAX_DOOR_DISAGREEMENT <= candidate / doors <= MAX_DOOR_DISAGREEMENT


def estimate_scale(
    *,
    user_cm_per_px: float | None,
    door_widths_px: list[float],
    max_wall_thickness_px: float,
    scale_bar_px_per_m: float | None,
    page_cm_per_px: float | None,
) -> ScaleEstimate:
    if user_cm_per_px is not None:
        return ScaleEstimate(user_cm_per_px, "user", "Set by user.")

    median = statistics.median(door_widths_px) if door_widths_px else None
    from_doors = DOOR_WIDTH_CM / median if median else None

    if scale_bar_px_per_m is not None:
        from_bar = SCALE_BAR_STEP_CM / scale_bar_px_per_m
        if _agrees(from_bar, from_doors):
            return ScaleEstimate(from_bar, "scale_bar", f"Read from the printed scale bar ({scale_bar_px_per_m:.1f} px per metre).")
    if page_cm_per_px is not None and _agrees(page_cm_per_px, from_doors):
        return ScaleEstimate(page_cm_per_px, "page_format", "Assumed an A4 page printed at 1:100.")
    if from_doors is not None:
        return ScaleEstimate(
            from_doors,
            "doors",
            f"Assumed {DOOR_WIDTH_CM:.0f} cm doors (median of {len(door_widths_px)} door(s): {median:.0f} px).",
        )
    return ScaleEstimate(
        EXTERIOR_WALL_CM / max_wall_thickness_px,
        "wall_thickness",
        f"No doors found; assumed {EXTERIOR_WALL_CM:.0f} cm exterior walls ({max_wall_thickness_px:.0f} px).",
    )
