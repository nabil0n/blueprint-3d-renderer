import pytest

from blueprint3d.evaluation.metrics import failed, living_area_m2, score
from blueprint3d.evaluation.truth import SampleTruth
from blueprint3d.parsing.result import ParseMeta, ParseResult
from blueprint3d.schema import Plan, Point, Room, Wall


def square_room(room_id: str, side_cm: float, kind: str = "other", x0: float = 0) -> Room:
    corners = [(x0, 0), (x0 + side_cm, 0), (x0 + side_cm, side_cm), (x0, side_cm)]
    return Room(id=room_id, kind=kind, polygon=[Point(x=x, y=y) for x, y in corners])


def result(rooms: list[Room], cm_per_px: float = 1.5) -> ParseResult:
    meta = ParseMeta(
        parser="test", cm_per_px=cm_per_px, scale_source="doors", scale_detail="", image_width=10, image_height=10
    )
    return ParseResult(plan=Plan(rooms=rooms), meta=meta)


TRUTH = SampleTruth(cm_per_px=1.5, listed_area_m2=20, rooms=3, room_tolerance=0, balconies=1)
ROOMS = [
    square_room("r1", 300),
    square_room("r2", 332, x0=2000),
    square_room("r3", 200, kind="balcony", x0=4000),
]


def test_living_area_excludes_balconies():
    assert living_area_m2(Plan(rooms=ROOMS)) == pytest.approx(9 + 11.0224, rel=0.02)


def test_living_area_includes_interior_walls_like_the_printed_boa():
    """Two 3 x 3 m rooms either side of a 20 cm interior wall: BOA counts 6.2 x 3 m."""
    a, b = square_room("a", 300), square_room("b", 300, x0=320)
    wall = Wall(id="w", start=Point(x=310, y=0), end=Point(x=310, y=300), thickness=20)
    assert living_area_m2(Plan(rooms=[a, b], walls=[wall])) == pytest.approx(6.2 * 3, rel=0.02)


def test_score_compares_against_truth():
    s = score("plan.jpg", result(ROOMS, cm_per_px=1.65), TRUTH)
    assert s.scale_error == pytest.approx(0.10)
    assert s.area_error == pytest.approx((9 + 11.0224) / 20 - 1, abs=0.02)
    assert (s.rooms_found, s.balconies_found) == (3, 1)
    assert s.error is None


def test_pass_requires_every_known_metric_within_tolerance():
    assert score("a", result(ROOMS), TRUTH).passed
    assert not score("a", result(ROOMS, cm_per_px=1.65), TRUTH).passed, "scale 10 % off"
    assert not score("a", result(ROOMS[:2]), TRUTH).passed, "balcony missing"
    lenient = TRUTH.model_copy(update={"rooms": 4, "room_tolerance": 1})
    assert score("a", result(ROOMS), lenient).passed


def test_unknown_metrics_are_skipped():
    s = score("a", result(ROOMS), SampleTruth(cm_per_px=1.5, rooms=3, balconies=1))
    assert s.area_error is None
    assert s.passed


def test_without_truth_nothing_is_judged():
    s = score("a", result(ROOMS), None)
    assert s.scale_error is None and s.rooms_expected is None
    assert s.passed is None


def test_parse_failures_are_recorded():
    s = failed("a", "No walls found.", TRUTH)
    assert s.error == "No walls found."
    assert s.passed is False


def named(room: Room, name: str) -> Room:
    return room.model_copy(update={"name": name})


def test_counts_printed_room_names_that_were_read():
    truth = SampleTruth(cm_per_px=1.5, rooms=3, balconies=1, room_names=["Sovrum", "Sovrum", "Kök", "Balkong"])
    rooms = [named(ROOMS[0], "SOVRUM"), named(ROOMS[1], "Kök / Sovrum / Entré"), ROOMS[2]]
    s = score("a", result(rooms), truth)
    assert (s.names_found, s.names_expected) == (3, 4)


def test_a_name_with_a_slash_is_one_name():
    truth = SampleTruth(cm_per_px=1.5, rooms=3, balconies=1, room_names=["Wc/dusch"])
    s = score("a", result([named(ROOMS[0], "Wc/dusch"), *ROOMS[1:]]), truth)
    assert s.names_found == 1


def test_room_names_do_not_decide_pass_or_fail():
    truth = SampleTruth(cm_per_px=1.5, rooms=3, balconies=1, room_names=["Sovrum"])
    s = score("a", result(ROOMS), truth)
    assert (s.names_found, s.names_expected, s.passed) == (0, 1, True)


def test_names_are_not_judged_without_listed_names():
    s = score("a", result(ROOMS), SampleTruth(cm_per_px=1.5, rooms=3, balconies=1))
    assert s.names_expected is None
