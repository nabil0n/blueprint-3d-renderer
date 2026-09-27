import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from blueprint3d.schema import Opening, Plan, Point, Room, Wall

FIXTURES = Path(__file__).parent / "fixtures"


def make_wall(wall_id="w1", start=(0, 0), end=(400, 0), **kwargs):
    return Wall(id=wall_id, start=Point(x=start[0], y=start[1]), end=Point(x=end[0], y=end[1]), **kwargs)


class TestWall:
    def test_length_is_euclidean_distance(self):
        wall = make_wall(start=(0, 0), end=(300, 400))
        assert wall.length == pytest.approx(500)

    def test_defaults(self):
        wall = make_wall()
        assert wall.thickness == 15
        assert wall.height == 250

    def test_rejects_zero_length(self):
        with pytest.raises(ValidationError, match="zero length"):
            make_wall(start=(10, 10), end=(10, 10))

    def test_rejects_non_positive_thickness(self):
        with pytest.raises(ValidationError):
            make_wall(thickness=0)


class TestOpening:
    def test_window_defaults_to_raised_sill(self):
        window = Opening(id="o1", wall_id="w1", kind="window", offset=100, width=120)
        assert window.sill_height == 90

    def test_door_sits_on_floor(self):
        door = Opening(id="o1", wall_id="w1", kind="door", offset=100, width=90)
        assert door.sill_height == 0

    def test_rejects_unknown_kind(self):
        with pytest.raises(ValidationError):
            Opening(id="o1", wall_id="w1", kind="portal", offset=100, width=90)


class TestRoom:
    def test_rejects_polygon_with_fewer_than_three_points(self):
        with pytest.raises(ValidationError):
            Room(id="r1", polygon=[Point(x=0, y=0), Point(x=1, y=1)])

    def test_area_uses_shoelace_formula(self):
        room = Room(
            id="r1",
            polygon=[Point(x=0, y=0), Point(x=400, y=0), Point(x=400, y=300), Point(x=0, y=300)],
        )
        assert room.area == pytest.approx(120_000)


class TestPlan:
    def test_rejects_duplicate_ids(self):
        with pytest.raises(ValidationError, match="Duplicate id"):
            Plan(walls=[make_wall("w1"), make_wall("w1", start=(0, 0), end=(0, 300))])

    def test_rejects_opening_on_unknown_wall(self):
        with pytest.raises(ValidationError, match="unknown wall"):
            Plan(
                walls=[make_wall("w1")],
                openings=[Opening(id="o1", wall_id="nope", kind="door", offset=100, width=90)],
            )

    def test_rejects_opening_that_overhangs_wall(self):
        with pytest.raises(ValidationError, match="does not fit"):
            Plan(
                walls=[make_wall("w1", end=(100, 0))],
                openings=[Opening(id="o1", wall_id="w1", kind="door", offset=80, width=90)],
            )

    def test_rejects_window_taller_than_wall(self):
        with pytest.raises(ValidationError, match="taller than"):
            Plan(
                walls=[make_wall("w1", height=200)],
                openings=[Opening(id="o1", wall_id="w1", kind="window", offset=200, width=100, height=150)],
            )

    def test_sample_fixture_round_trips(self):
        raw = json.loads((FIXTURES / "two_room_apartment.json").read_text())
        plan = Plan.model_validate(raw)
        assert Plan.model_validate_json(plan.model_dump_json()) == plan
        assert len(plan.rooms) == 2
