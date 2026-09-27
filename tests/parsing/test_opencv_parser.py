from pathlib import Path

import numpy as np
import pytest

from blueprint3d.parsing.errors import ParseError
from blueprint3d.parsing.opencv_parser import OpenCvParser
from blueprint3d.parsing.scale import DOOR_WIDTH_CM
from tests.parsing.synthetic import (
    TRUE_CM_PER_PX,
    blank,
    draw_bay_and_balcony_plan,
    draw_two_room_plan,
    encode_png,
)

REPO = Path(__file__).resolve().parents[2]
REAL_SAMPLES = sorted([*REPO.glob("data/*.jpg"), *REPO.glob("samples/*.jpg"), *REPO.glob("samples/*.png")])


@pytest.fixture(scope="module")
def synthetic_result():
    return OpenCvParser().parse(encode_png(draw_two_room_plan()))


def test_estimates_scale_from_doors(synthetic_result):
    meta = synthetic_result.meta
    assert meta.scale_source == "doors"
    assert meta.cm_per_px == pytest.approx(TRUE_CM_PER_PX, rel=0.05)


def test_finds_the_walls(synthetic_result):
    walls = synthetic_result.plan.walls
    assert len(walls) == 5
    assert sum(w.exterior for w in walls) == 4


def test_finds_doors_and_windows(synthetic_result):
    kinds = sorted(o.kind for o in synthetic_result.plan.openings)
    assert kinds == ["door", "door", "window"]
    door_widths = [o.width for o in synthetic_result.plan.openings if o.kind == "door"]
    assert door_widths == pytest.approx([DOOR_WIDTH_CM, DOOR_WIDTH_CM], abs=6)


def test_finds_both_rooms_at_a_sensible_size(synthetic_result):
    rooms = synthetic_result.plan.rooms
    assert len(rooms) == 2
    # Interior of each room is roughly 370 x 560 px.
    expected_m2 = 370 * 560 * TRUE_CM_PER_PX**2 / 10_000
    for room in rooms:
        assert room.area / 10_000 == pytest.approx(expected_m2, rel=0.15)


def test_light_gray_window_lines_are_still_a_window():
    result = OpenCvParser().parse(encode_png(draw_two_room_plan(window_ink=190)))
    assert sorted(o.kind for o in result.plan.openings) == ["door", "door", "window"]


def test_thin_interior_walls_still_separate_rooms():
    result = OpenCvParser().parse(encode_png(draw_two_room_plan(interior_px=7)))
    assert len(result.plan.rooms) == 2
    assert len(result.plan.walls) == 5


@pytest.fixture(scope="module")
def bay_result():
    return OpenCvParser().parse(encode_png(draw_bay_and_balcony_plan()), cm_per_px=TRUE_CM_PER_PX)


def _is_angled(wall, min_cm=20):
    return abs(wall.end.x - wall.start.x) > min_cm and abs(wall.end.y - wall.start.y) > min_cm


def test_angled_glazed_wall_becomes_a_full_height_wall_with_a_window(bay_result):
    plan = bay_result.plan
    angled = [w for w in plan.walls if _is_angled(w)]
    assert len(angled) == 1
    wall = angled[0]
    assert wall.height == 250
    assert wall.exterior
    # The glazing spans ~95 px each way between the corner blocks' inner faces.
    assert wall.length == pytest.approx(95 * np.sqrt(2) * TRUE_CM_PER_PX, rel=0.2)
    windows = [o for o in plan.openings if o.wall_id == wall.id]
    assert [o.kind for o in windows] == ["window"]
    assert windows[0].width > 0.5 * wall.length


def test_balcony_gets_half_height_railings(bay_result):
    plan = bay_result.plan
    balconies = [r for r in plan.rooms if r.kind == "balcony"]
    assert len(balconies) == 1
    railings = [w for w in plan.walls if w.height == 125]
    assert len(railings) == 3
    assert all(w.thickness <= 10 and w.exterior for w in railings)
    assert not any(o.wall_id in {w.id for w in railings} for o in plan.openings)


def test_rooms_with_ordinary_walls_are_not_balconies(bay_result):
    assert sorted(r.kind for r in bay_result.plan.rooms) == ["balcony", "other", "other"]


def test_user_scale_is_in_original_image_pixels():
    img = draw_two_room_plan()
    result = OpenCvParser(max_side=500).parse(encode_png(img), cm_per_px=2.0)
    assert result.meta.cm_per_px == 2.0
    assert result.meta.image_width == 1000
    north = max(result.plan.walls, key=lambda w: abs(w.end.x - w.start.x))
    assert abs(north.end.x - north.start.x) == pytest.approx((800 - 20) * 2.0, rel=0.05)


def test_rejects_an_image_without_walls():
    with pytest.raises(ParseError):
        OpenCvParser().parse(encode_png(blank()))


@pytest.mark.parametrize("path", REAL_SAMPLES, ids=lambda p: p.name)
def test_real_sample_produces_a_plausible_plan(path):
    result = OpenCvParser().parse(path.read_bytes())
    assert len(result.plan.walls) >= 8
    assert len(result.plan.rooms) >= 4
    assert any(o.kind == "door" for o in result.plan.openings)
    assert any(o.kind == "window" for o in result.plan.openings)
    assert 0.5 < result.meta.cm_per_px < 5
