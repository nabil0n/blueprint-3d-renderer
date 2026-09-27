"""Comparing one parse result with its ground truth."""

from dataclasses import dataclass

import cv2
import numpy as np

from blueprint3d.evaluation.truth import SampleTruth
from blueprint3d.parsing.result import ParseResult
from blueprint3d.schema import Plan

SCALE_TOLERANCE = 0.05
AREA_TOLERANCE = 0.10
AREA_RASTER_CM = 2.0


@dataclass(frozen=True)
class SampleScore:
    name: str
    scale_error: float | None
    """Relative: estimated / true - 1."""
    area_error: float | None
    """Relative: living area (rooms except balconies) / listed area - 1."""
    rooms_found: int
    rooms_expected: int | None
    room_tolerance: int
    balconies_found: int
    balconies_expected: int | None
    scale_source: str | None = None
    error: str | None = None

    @property
    def passed(self) -> bool | None:
        """None without ground truth; otherwise every known metric must be within tolerance."""
        if self.error is not None:
            return False
        if self.rooms_expected is None:
            return None
        checks = [
            self.scale_error is None or abs(self.scale_error) <= SCALE_TOLERANCE,
            self.area_error is None or abs(self.area_error) <= AREA_TOLERANCE,
            abs(self.rooms_found - self.rooms_expected) <= self.room_tolerance,
            self.balconies_found == self.balconies_expected,
        ]
        return all(checks)


def living_area_m2(plan: Plan) -> float:
    """Living area as printed on Swedish plans (BOA, SS 21054): the floor inside the exterior walls,
    interior walls included, balconies excluded. Rooms are rasterised and the gaps between them
    closed up to the thickest interior wall."""
    rooms = [room for room in plan.rooms if room.kind != "balcony"]
    if not rooms:
        return 0.0
    interior = [w.thickness for w in plan.walls if not w.exterior]
    close_px = int(np.ceil(max(interior) / AREA_RASTER_CM)) + 2 if interior else 0
    # Pad beyond the closing kernel: erosion treats pixels outside the image as filled.
    pad_cm = (close_px + 2) * AREA_RASTER_CM
    points = np.array([(p.x, p.y) for room in rooms for p in room.polygon])
    origin = points.min(axis=0) - pad_cm
    width, height = np.ceil((points.max(axis=0) + pad_cm - origin) / AREA_RASTER_CM).astype(int)
    mask = np.zeros((height, width), np.uint8)
    for room in rooms:
        polygon = (np.array([(p.x, p.y) for p in room.polygon]) - origin) / AREA_RASTER_CM
        cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 1)
    if close_px:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_px, close_px), np.uint8))
    return float(mask.sum()) * AREA_RASTER_CM**2 / 10_000


def score(name: str, result: ParseResult, truth: SampleTruth | None) -> SampleScore:
    plan = result.plan
    balconies = sum(room.kind == "balcony" for room in plan.rooms)
    if truth is None:
        return SampleScore(
            name=name,
            scale_error=None,
            area_error=None,
            rooms_found=len(plan.rooms),
            rooms_expected=None,
            room_tolerance=0,
            balconies_found=balconies,
            balconies_expected=None,
            scale_source=result.meta.scale_source,
        )
    return SampleScore(
        name=name,
        scale_error=result.meta.cm_per_px / truth.cm_per_px - 1,
        area_error=living_area_m2(plan) / truth.listed_area_m2 - 1 if truth.listed_area_m2 else None,
        rooms_found=len(plan.rooms),
        rooms_expected=truth.rooms,
        room_tolerance=truth.room_tolerance,
        balconies_found=balconies,
        balconies_expected=truth.balconies,
        scale_source=result.meta.scale_source,
    )


def failed(name: str, message: str, truth: SampleTruth | None) -> SampleScore:
    return SampleScore(
        name=name,
        scale_error=None,
        area_error=None,
        rooms_found=0,
        rooms_expected=truth.rooms if truth else None,
        room_tolerance=truth.room_tolerance if truth else 0,
        balconies_found=0,
        balconies_expected=truth.balconies if truth else None,
        error=message,
    )
