"""Estimating centimetres per pixel when the user has not calibrated the plan."""

import statistics
from dataclasses import dataclass
from typing import Literal

DOOR_WIDTH_CM = 80.0
"""Typical door opening as drawn in residential plans (wall gap for an 80 cm door)."""
EXTERIOR_WALL_CM = 30.0
"""Typical thickness of an apartment's exterior wall as drawn."""

ScaleSource = Literal["user", "doors", "wall_thickness"]


@dataclass(frozen=True)
class ScaleEstimate:
    cm_per_px: float
    source: ScaleSource
    detail: str


def estimate_scale(
    *, user_cm_per_px: float | None, door_widths_px: list[float], max_wall_thickness_px: float
) -> ScaleEstimate:
    if user_cm_per_px is not None:
        return ScaleEstimate(user_cm_per_px, "user", "Set by user.")
    if door_widths_px:
        median = statistics.median(door_widths_px)
        return ScaleEstimate(
            DOOR_WIDTH_CM / median,
            "doors",
            f"Assumed {DOOR_WIDTH_CM:.0f} cm doors (median of {len(door_widths_px)} door(s): {median:.0f} px).",
        )
    return ScaleEstimate(
        EXTERIOR_WALL_CM / max_wall_thickness_px,
        "wall_thickness",
        f"No doors found; assumed {EXTERIOR_WALL_CM:.0f} cm exterior walls ({max_wall_thickness_px:.0f} px).",
    )
