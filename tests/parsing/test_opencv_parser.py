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
    draw_decorated_plan,
    draw_outward_door_and_double_railing_plan,
    draw_tick_scale_bar,
    draw_two_room_plan,
    encode_png,
)

REPO = Path(__file__).resolve().parents[2]
REAL_SAMPLES = sorted([*REPO.glob("data/*.jpg"), *REPO.glob("samples/*.jpg"), *REPO.glob("samples/*.png")])
UNSUPPORTED_STYLES: dict[str, str] = {
    "länsmansgården.jpg": "windows drawn as thick double frames in the wall read as solid wall: no openings",
}


def _sample_param(path: Path):
    reason = UNSUPPORTED_STYLES.get(path.name)
    marks = [pytest.mark.xfail(reason=reason, strict=True)] if reason else []
    return pytest.param(path, id=path.name, marks=marks)


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
    kinds = [r.kind for r in bay_result.plan.rooms]
    assert len(kinds) == 3 and kinds.count("balcony") == 1


@pytest.fixture(scope="module")
def decorated_result():
    return OpenCvParser().parse(encode_png(draw_decorated_plan()), cm_per_px=TRUE_CM_PER_PX)


def test_coloured_logos_are_not_walls(decorated_result):
    """The logo spans y 100-175 px in the image; the plan's north wall is at y 220-239."""
    oy = decorated_result.meta.origin_px[1]
    tops = [oy + min(w.start.y, w.end.y) / TRUE_CM_PER_PX for w in decorated_result.plan.walls]
    assert min(tops) > 200


def test_flat_room_fills_do_not_hide_rooms(decorated_result):
    kinds = sorted(r.kind for r in decorated_result.plan.rooms)
    assert kinds == ["balcony", "other", "other"]


@pytest.fixture(scope="module")
def outward_door_result():
    img = draw_outward_door_and_double_railing_plan()
    return OpenCvParser().parse(encode_png(img), cm_per_px=TRUE_CM_PER_PX)


def test_a_door_leaf_drawn_outside_is_not_a_railing(outward_door_result):
    kinds = [r.kind for r in outward_door_result.plan.rooms]
    assert len(kinds) == 3 and kinds.count("balcony") == 1


def test_a_double_line_balcony_outline_is_a_railing_not_glazing(outward_door_result):
    plan = outward_door_result.plan
    railings = [w for w in plan.walls if w.height < 200]
    assert len(railings) == 3
    windows_on_low_walls = [o for o in plan.openings if o.wall_id in {w.id for w in railings}]
    assert windows_on_low_walls == []


def test_a_printed_scale_bar_sets_the_scale():
    img = np.vstack([draw_two_room_plan(), blank(1000, 200)])
    draw_tick_scale_bar(img, 100, 900, 60.0)
    meta = OpenCvParser().parse(encode_png(img)).meta
    assert meta.scale_source == "scale_bar"
    assert meta.cm_per_px == pytest.approx(100 / 60, rel=0.02)


def test_a_light_gray_scale_bar_is_found_too():
    img = np.vstack([draw_two_room_plan(), blank(1000, 200)])
    draw_tick_scale_bar(img, 100, 900, 60.0, ink=170)
    assert OpenCvParser().parse(encode_png(img)).meta.scale_source == "scale_bar"


def test_a4_pages_assume_1_to_100():
    page = blank(1414, 2000)
    page[600:1400, 200:1200] = draw_two_room_plan()
    meta = OpenCvParser().parse(encode_png(page)).meta
    assert meta.scale_source == "page_format"
    assert meta.cm_per_px == pytest.approx(2970 / 2000)


def test_origin_maps_plan_coordinates_back_onto_the_image():
    """Plan (x, y) in cm -> image pixels: origin_px + (x, y) / cm_per_px, in original image pixels."""
    img = draw_two_room_plan()
    result = OpenCvParser(max_side=500).parse(encode_png(img), cm_per_px=2.0)
    ox, oy = result.meta.origin_px
    north = min(result.plan.walls, key=lambda w: w.start.y + w.end.y)
    # The north wall's centre line is at y = 109.5 in the original image.
    assert oy + north.start.y / 2.0 == pytest.approx(109.5, abs=3)
    assert 0 <= ox < 100


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


@pytest.mark.parametrize("path", [_sample_param(p) for p in REAL_SAMPLES])
def test_real_sample_produces_a_plausible_plan(path):
    result = OpenCvParser().parse(path.read_bytes())
    assert len(result.plan.walls) >= 8
    assert len(result.plan.rooms) >= 4
    assert any(o.kind == "door" for o in result.plan.openings)
    assert any(o.kind == "window" for o in result.plan.openings)
    assert 0.5 < result.meta.cm_per_px < 5


def test_rooms_are_named_from_their_printed_labels(synthetic_result):
    names = sorted((room.name, room.kind) for room in synthetic_result.plan.rooms)
    assert names == [("Bedroom", "bedroom"), ("Living", "living_room")]


def test_a_named_balcony_stays_a_balcony(bay_result):
    balcony = next(room for room in bay_result.plan.rooms if room.kind == "balcony")
    assert balcony.name == "Balcony"


def test_room_names_are_optional():
    result = OpenCvParser(read_text=None).parse(encode_png(draw_two_room_plan()))
    assert [room.name for room in result.plan.rooms] == [None, None]


def test_unreadable_text_leaves_rooms_unnamed_with_a_warning():
    def broken_reader(_gray):
        raise RuntimeError("model missing")

    result = OpenCvParser(read_text=broken_reader).parse(encode_png(draw_two_room_plan()))
    assert len(result.plan.rooms) == 2
    assert all(room.name is None for room in result.plan.rooms)
    assert any("Room names could not be read" in w for w in result.meta.warnings)
