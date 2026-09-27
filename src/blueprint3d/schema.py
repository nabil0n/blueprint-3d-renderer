"""The plan schema: the contract between parsers (image -> plan) and renderers (plan -> 3D).

Coordinates are in plan space: centimetres, origin at the top-left, x to the right, y down.
All models are frozen; build new instances instead of mutating.
"""

import math
from collections import Counter
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_WALL_THICKNESS_CM = 15
DEFAULT_WALL_HEIGHT_CM = 250
DEFAULT_DOOR_HEIGHT_CM = 210
DEFAULT_WINDOW_HEIGHT_CM = 120
DEFAULT_WINDOW_SILL_CM = 90

OpeningKind = Literal["door", "window"]
RoomKind = Literal[
    "living_room", "bedroom", "kitchen", "bathroom", "hallway", "closet", "balcony", "other"
]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Point(_Frozen):
    x: float
    y: float


class Wall(_Frozen):
    """A straight wall segment, centred on the line from start to end."""

    id: str
    start: Point
    end: Point
    thickness: float = Field(default=DEFAULT_WALL_THICKNESS_CM, gt=0)
    height: float = Field(default=DEFAULT_WALL_HEIGHT_CM, gt=0)
    exterior: bool = False

    @property
    def length(self) -> float:
        return math.dist((self.start.x, self.start.y), (self.end.x, self.end.y))

    @model_validator(mode="after")
    def _non_degenerate(self) -> Self:
        if self.length == 0:
            raise ValueError(f"Wall {self.id!r} has zero length")
        return self


class Opening(_Frozen):
    """A door or window cut into a wall.

    `offset` is the distance along the wall from its start point to the opening's centre.
    """

    id: str
    wall_id: str
    kind: OpeningKind
    offset: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float | None = Field(default=None, gt=0)
    sill_height: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _apply_kind_defaults(self) -> Self:
        is_door = self.kind == "door"
        defaults = {
            "height": DEFAULT_DOOR_HEIGHT_CM if is_door else DEFAULT_WINDOW_HEIGHT_CM,
            "sill_height": 0 if is_door else DEFAULT_WINDOW_SILL_CM,
        }
        # Frozen model: fill unset fields during validation, before the instance is shared.
        for field, default in defaults.items():
            if getattr(self, field) is None:
                object.__setattr__(self, field, default)
        return self


class Room(_Frozen):
    id: str
    polygon: list[Point] = Field(min_length=3)
    kind: RoomKind = "other"
    name: str | None = None

    @property
    def area(self) -> float:
        """Area in cm², via the shoelace formula."""
        pts = self.polygon
        twice_area = sum(
            a.x * b.y - b.x * a.y for a, b in zip(pts, pts[1:] + pts[:1], strict=True)
        )
        return abs(twice_area) / 2


class Plan(_Frozen):
    version: Literal[1] = 1
    units: Literal["cm"] = "cm"
    walls: list[Wall] = Field(default_factory=list)
    openings: list[Opening] = Field(default_factory=list)
    rooms: list[Room] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_references(self) -> Self:
        all_ids = [item.id for item in (*self.walls, *self.openings, *self.rooms)]
        duplicates = sorted(i for i, n in Counter(all_ids).items() if n > 1)
        if duplicates:
            raise ValueError(f"Duplicate id(s): {duplicates}")

        walls_by_id = {wall.id: wall for wall in self.walls}
        for opening in self.openings:
            wall = walls_by_id.get(opening.wall_id)
            if wall is None:
                raise ValueError(f"Opening {opening.id!r} references unknown wall {opening.wall_id!r}")
            _check_opening_fits(opening, wall)
        return self


def _check_opening_fits(opening: Opening, wall: Wall) -> None:
    half = opening.width / 2
    if opening.offset - half < 0 or opening.offset + half > wall.length:
        raise ValueError(
            f"Opening {opening.id!r} (offset {opening.offset}, width {opening.width}) "
            f"does not fit on wall {wall.id!r} of length {wall.length:.1f}"
        )
    top = (opening.sill_height or 0) + (opening.height or 0)
    if top > wall.height:
        raise ValueError(
            f"Opening {opening.id!r} top at {top} is taller than wall {wall.id!r} ({wall.height})"
        )
