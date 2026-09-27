import math

import pytest

from blueprint3d.parsing.geometry import PxOpening, PxWall
from blueprint3d.parsing.merge import merge_parallel_walls


def wall(start, end, thickness=4.0, openings=(), **kwargs):
    return PxWall(start=start, end=end, thickness=thickness, openings=tuple(openings), **kwargs)


def length(w):
    return math.dist(w.start, w.end)


def test_an_outlined_wall_becomes_one_thick_wall():
    """Two 4 px lines 20 px apart (centre to centre) are the faces of one ~24 px wall."""
    [merged] = merge_parallel_walls([wall((0, 100), (300, 100)), wall((20, 120), (320, 120))], max_gap=40)
    assert merged.thickness == pytest.approx(24)
    assert merged.start[1] == pytest.approx(110) and merged.end[1] == pytest.approx(110)
    assert length(merged) == pytest.approx(320)


def test_openings_from_both_faces_are_kept_once():
    a = wall((0, 100), (300, 100), openings=[PxOpening(offset=150, width=60, kind="window")])
    b = wall((0, 120), (300, 120), openings=[PxOpening(offset=148, width=60, kind="window")])
    c = wall((300, 140), (0, 140), openings=[PxOpening(offset=50, width=40, kind="door")])  # reversed
    [merged] = merge_parallel_walls([a, b], max_gap=40)
    assert [(round(o.offset), round(o.width), o.kind) for o in merged.openings] == [(149, 62, "window")]
    [merged] = merge_parallel_walls([a, c], max_gap=40)
    kinds = sorted((round(o.offset), o.kind) for o in merged.openings)
    assert kinds == [(150, "window"), (250, "door")]


@pytest.mark.parametrize(
    "other",
    [
        wall((0, 160), (300, 160)),  # too far apart
        wall((400, 110), (600, 110)),  # no overlap along the wall
        wall((150, 0), (150, 300)),  # perpendicular
        wall((0, 110), (300, 110), height_cm=125),  # railing next to a wall
    ],
)
def test_walls_that_are_not_two_faces_of_one_wall_stay_apart(other):
    assert len(merge_parallel_walls([wall((0, 100), (300, 100)), other], max_gap=40)) == 2


def test_exterior_wins_and_merging_repeats_until_stable():
    walls = [wall((0, 100), (300, 100)), wall((0, 115), (300, 115), exterior=True), wall((0, 130), (300, 130))]
    [merged] = merge_parallel_walls(walls, max_gap=40)
    assert merged.exterior
    assert merged.thickness == pytest.approx(34)


def test_angled_walls_merge_too():
    s = math.sqrt(0.5)
    a = wall((0, 0), (200, 200))
    b = wall((10 * s * 2, -10 * s * 2), (200 + 10 * s * 2, 200 - 10 * s * 2))
    [merged] = merge_parallel_walls([a, b], max_gap=40)
    assert merged.thickness == pytest.approx(20 + 4, abs=0.5)
