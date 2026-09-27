import numpy as np
import pytest

from blueprint3d.parsing.geometry import PxOpening, PxWall
from blueprint3d.parsing.plan_builder import build_plan, mark_exterior, windows_facing_outside


def wall(**kwargs):
    defaults = {"start": (100.0, 100.0), "end": (300.0, 100.0), "thickness": 10.0}
    return PxWall(**{**defaults, **kwargs})


def test_wall_bordering_outside_space_is_exterior():
    outside = np.zeros((400, 400), bool)
    outside[:95, :] = True  # everything above the wall is outside
    assert mark_exterior(wall(), outside).exterior


def test_wall_between_rooms_is_interior():
    assert not mark_exterior(wall(), np.zeros((400, 400), bool)).exterior


def test_windows_must_face_outside_space():
    outside = np.zeros((400, 400), bool)
    outside[:95, :200] = True  # outside only above the first half of the wall
    facing = PxOpening(offset=50, width=40, kind="window")
    between_rooms = PxOpening(offset=150, width=40, kind="window")

    result = windows_facing_outside(wall(openings=(facing, between_rooms)), outside)

    assert [o.kind for o in result.openings] == ["window", "door"]


def test_build_plan_scales_to_cm():
    plan = build_plan(
        [wall(openings=(PxOpening(offset=100, width=40, kind="door"),))],
        rooms=(((0.0, 0.0), (10.0, 0.0), (10.0, 10.0)),),
        cm_per_px=2.0,
    )
    assert plan.walls[0].length == pytest.approx(400)
    assert plan.walls[0].thickness == 20
    assert (plan.openings[0].offset, plan.openings[0].width) == (200, 80)
    assert plan.rooms[0].area == pytest.approx(200)


def test_build_plan_drops_openings_that_do_not_fit():
    plan = build_plan([wall(openings=(PxOpening(offset=195, width=40, kind="door"),))], rooms=(), cm_per_px=1.0)
    assert plan.openings == []
