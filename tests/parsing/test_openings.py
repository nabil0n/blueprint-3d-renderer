import numpy as np

from blueprint3d.parsing.geometry import AxisSegment, WallRun
from blueprint3d.parsing.openings import classify_gap, merge_collinear, run_to_wall


def seg(start, end, center=110.0, thickness=20.0, axis="h"):
    return AxisSegment(axis=axis, center=center, start=start, end=end, thickness=thickness)


def test_merges_collinear_segments_and_records_the_gap():
    [run] = merge_collinear([seg(300, 500), seg(50, 240)], min_gap=20, max_gap=200)
    assert (run.start, run.end) == (50, 500)
    assert run.gaps == ((240, 300),)


def test_chains_several_gaps():
    [run] = merge_collinear([seg(0, 100), seg(160, 260), seg(320, 400)], min_gap=20, max_gap=200)
    assert run.gaps == ((100, 160), (260, 320))


def test_fills_tiny_breaks_without_an_opening():
    [run] = merge_collinear([seg(0, 100), seg(105, 200)], min_gap=20, max_gap=200)
    assert (run.start, run.end, run.gaps) == (0, 200, ())


def test_keeps_far_apart_segments_separate():
    runs = merge_collinear([seg(0, 100), seg(400, 500)], min_gap=20, max_gap=200)
    assert len(runs) == 2


def test_does_not_merge_parallel_offset_or_perpendicular_segments():
    parallel = merge_collinear([seg(0, 100), seg(150, 250, center=200)], min_gap=20, max_gap=200)
    perpendicular = merge_collinear([seg(0, 100), seg(150, 250, axis="v")], min_gap=20, max_gap=200)
    assert len(parallel) == 2
    assert len(perpendicular) == 2


def horizontal_run():
    return WallRun(axis="h", center=110, thickness=20, start=0, end=400, gaps=((100, 250),))


def test_gap_with_lines_along_the_wall_is_a_window():
    binary = np.zeros((300, 400), np.uint8)
    for y in (100, 109, 119):
        binary[y, 100:250] = 255
    assert classify_gap(binary, horizontal_run(), (100, 250)) == "window"


def test_empty_gap_or_crossing_door_leaf_is_a_door():
    binary = np.zeros((300, 400), np.uint8)
    binary[40:120, 100] = 255  # door leaf crossing the wall band
    assert classify_gap(binary, horizontal_run(), (100, 250)) == "door"


def test_dashed_lines_across_a_gap_mark_a_passage_not_a_window():
    binary = np.zeros((300, 400), np.uint8)
    for y in (104, 115):
        for x in range(100, 250, 10):
            binary[y, x : x + 7] = 255  # 70 % dashes
    assert classify_gap(binary, horizontal_run(), (100, 250)) == "door"


def test_classifies_vertical_gaps_too():
    run = WallRun(axis="v", center=110, thickness=20, start=0, end=400, gaps=((100, 250),))
    binary = np.zeros((400, 300), np.uint8)
    for x in (100, 119):
        binary[100:250, x] = 255
    assert classify_gap(binary, run, (100, 250)) == "window"


def test_run_to_wall_insets_the_centre_line_and_places_openings():
    run = WallRun(axis="h", center=50, thickness=10, start=0, end=300, gaps=((100, 180),))
    wall = run_to_wall(run, [((100, 180), "window"), ((200, 260), "door")])
    assert wall is not None
    assert (wall.start, wall.end) == ((5, 50), (295, 50))
    assert [(o.offset, o.width, o.kind) for o in wall.openings] == [(135, 80, "window"), (225, 60, "door")]


def test_run_to_wall_drops_openings_beyond_the_wall_ends():
    run = WallRun(axis="v", center=20, thickness=10, start=0, end=100)
    wall = run_to_wall(run, [((0, 30), "door")])
    assert wall is not None and wall.openings == ()


def test_run_shorter_than_its_thickness_is_no_wall():
    assert run_to_wall(WallRun(axis="h", center=0, thickness=10, start=0, end=10), []) is None
