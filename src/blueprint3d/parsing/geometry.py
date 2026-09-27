"""Pixel-space intermediate shapes used while parsing. Converted to cm plan models at the end."""

from dataclasses import dataclass
from typing import Literal

Axis = Literal["h", "v"]
Gap = tuple[float, float]
PxPoint = tuple[float, float]


@dataclass(frozen=True)
class AxisSegment:
    """An axis-aligned wall piece. For "h", `center` is y and start/end are x edges; vice versa for "v"."""

    axis: Axis
    center: float
    start: float
    end: float
    thickness: float

    @property
    def length(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class WallRun:
    """Collinear axis segments merged into one wall; `gaps` are the openings between them."""

    axis: Axis
    center: float
    thickness: float
    start: float
    end: float
    gaps: tuple[Gap, ...] = ()


@dataclass(frozen=True)
class PxOpening:
    offset: float
    """Distance from the wall's start point to the opening centre."""
    width: float
    kind: Literal["door", "window"]


@dataclass(frozen=True)
class PxWall:
    start: PxPoint
    end: PxPoint
    thickness: float
    openings: tuple[PxOpening, ...] = ()
    exterior: bool = False
