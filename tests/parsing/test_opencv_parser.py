from pathlib import Path

import pytest

from blueprint3d.parsing.errors import ParseError
from blueprint3d.parsing.opencv_parser import OpenCvParser
from blueprint3d.parsing.scale import DOOR_WIDTH_CM
from tests.parsing.synthetic import TRUE_CM_PER_PX, blank, draw_two_room_plan, encode_png

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
