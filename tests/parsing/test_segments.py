import cv2
import numpy as np
import pytest

from blueprint3d.parsing.segments import extract_axis_segments, extract_diagonal_walls


def mask(width=400, height=300):
    return np.zeros((height, width), np.uint8)


def test_horizontal_bar_becomes_one_segment():
    m = mask()
    m[100:120, 50:350] = 255
    [seg] = extract_axis_segments(m, min_run=30)
    assert seg.axis == "h"
    assert seg.center == pytest.approx(110)
    assert (seg.start, seg.end) == (50, 350)
    assert seg.thickness == pytest.approx(20)


def test_vertical_bar_becomes_one_segment():
    m = mask()
    m[20:280, 200:212] = 255
    [seg] = extract_axis_segments(m, min_run=30)
    assert seg.axis == "v"
    assert seg.center == pytest.approx(206)
    assert (seg.start, seg.end) == (20, 280)
    assert seg.thickness == pytest.approx(12)


def test_l_shape_gives_one_segment_per_axis():
    m = mask()
    m[100:120, 50:350] = 255
    m[100:280, 50:70] = 255
    segs = extract_axis_segments(m, min_run=30)
    assert sorted(s.axis for s in segs) == ["h", "v"]


def test_drops_blobs_that_are_not_wall_shaped():
    m = mask()
    m[100:140, 100:140] = 255  # a filled square, e.g. a column or legend icon
    assert extract_axis_segments(m, min_run=30) == []


def test_diagonal_wall_is_recovered_from_the_residual():
    m = mask()
    cv2.line(m, (50, 250), (300, 50), 255, 14)
    walls = extract_diagonal_walls(m, [], min_thickness=6)
    assert len(walls) == 1
    wall = walls[0]
    assert wall.thickness == pytest.approx(14, abs=2)
    length = np.hypot(wall.end[0] - wall.start[0], wall.end[1] - wall.start[1])
    # The stroke has round caps; the wall line runs between cap centres, i.e. the drawn endpoints.
    assert length == pytest.approx(np.hypot(250, 200), abs=8)


def test_diagonal_extraction_ignores_pixels_covered_by_axis_segments():
    m = mask()
    m[100:120, 50:350] = 255
    segs = extract_axis_segments(m, min_run=30)
    assert extract_diagonal_walls(m, segs, min_thickness=6) == []
