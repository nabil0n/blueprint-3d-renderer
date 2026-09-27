"""Comparing one parse result with its ground truth."""

from dataclasses import dataclass

from blueprint3d.evaluation.truth import SampleTruth
from blueprint3d.parsing.result import ParseResult
from blueprint3d.schema import Plan

SCALE_TOLERANCE = 0.05
AREA_TOLERANCE = 0.10


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
    return sum(room.area for room in plan.rooms if room.kind != "balcony") / 10_000


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
