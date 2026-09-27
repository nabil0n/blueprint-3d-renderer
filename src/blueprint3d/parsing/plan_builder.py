"""Converting pixel-space parse output into a validated Plan in centimetres."""

import math
from dataclasses import replace

import numpy as np

from blueprint3d.parsing.geometry import PxPoint, PxWall
from blueprint3d.schema import Opening, Plan, Point, Room, Wall

EXTERIOR_PROBE_PX = 4
EXTERIOR_SAMPLES = (0.1, 0.3, 0.5, 0.7, 0.9)
DECIMALS = 1


def _faces_outside(wall: PxWall, distance: float, outside: np.ndarray) -> bool:
    """Is there outside space just beyond either face of the wall, `distance` px from its start?"""
    (x0, y0), (x1, y1) = wall.start, wall.end
    length = math.hypot(x1 - x0, y1 - y0)
    if length == 0:
        return False
    dx, dy = (x1 - x0) / length, (y1 - y0) / length
    reach = wall.thickness / 2 + EXTERIOR_PROBE_PX
    px, py = x0 + dx * distance, y0 + dy * distance
    height, width = outside.shape
    for side in (-1, 1):
        x, y = round(px - side * dy * reach), round(py + side * dx * reach)
        if 0 <= x < width and 0 <= y < height and outside[y, x]:
            return True
    return False


def mark_exterior(wall: PxWall, outside: np.ndarray) -> PxWall:
    """A wall is exterior if, just beyond either face, there is free space outside the apartment."""
    length = math.hypot(wall.end[0] - wall.start[0], wall.end[1] - wall.start[1])
    exterior = any(_faces_outside(wall, t * length, outside) for t in EXTERIOR_SAMPLES)
    return replace(wall, exterior=exterior)


def windows_facing_outside(wall: PxWall, outside: np.ndarray) -> PxWall:
    """Windows need outside space beyond them. A line-filled gap between two rooms is usually
    a drawn threshold or an open passage marked with a dashed line, so it becomes a door."""
    openings = tuple(
        replace(o, kind="door") if o.kind == "window" and not _faces_outside(wall, o.offset, outside) else o
        for o in wall.openings
    )
    return replace(wall, openings=openings)


def build_plan(walls: list[PxWall], rooms: tuple[tuple[PxPoint, ...], ...], cm_per_px: float) -> Plan:
    def cm(value: float) -> float:
        return round(value * cm_per_px, DECIMALS)

    def point(p: PxPoint) -> Point:
        return Point(x=cm(p[0]), y=cm(p[1]))

    plan_walls, plan_openings = [], []
    for w in walls:
        wall = Wall(
            id=f"w{len(plan_walls) + 1}",
            start=point(w.start),
            end=point(w.end),
            thickness=max(cm(w.thickness), 1.0),
            exterior=w.exterior,
        )
        plan_walls.append(wall)
        for o in w.openings:
            opening = Opening(
                id=f"o{len(plan_openings) + 1}", wall_id=wall.id, kind=o.kind, offset=cm(o.offset), width=cm(o.width)
            )
            if opening.offset - opening.width / 2 >= 0 and opening.offset + opening.width / 2 <= wall.length:
                plan_openings.append(opening)

    plan_rooms = [Room(id=f"r{i}", polygon=[point(p) for p in poly]) for i, poly in enumerate(rooms, start=1)]
    return Plan(walls=plan_walls, openings=plan_openings, rooms=plan_rooms)
