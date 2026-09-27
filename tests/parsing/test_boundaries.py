import math

import cv2
import numpy as np
import pytest

from blueprint3d.parsing.boundaries import BoundaryConfig, complete_boundaries, covered_mask, enclose_rooms
from blueprint3d.parsing.geometry import PxWall
from blueprint3d.parsing.rooms import find_rooms

CONFIG = BoundaryConfig(
    wall_thickness=20,
    door_wall_thickness=12,
    railing_thickness=4,
    railing_height_cm=125,
    min_piece=40,
    door_range=(40, 150),
    probe_reach=60,
)


def box_with_open_east_side():
    """A room with thick walls on three sides. Returns (ink, wall_mask); the east side is left for the test."""
    wall_mask = np.zeros((400, 500), np.uint8)
    cv2.rectangle(wall_mask, (100, 100), (339, 119), 255, -1)
    cv2.rectangle(wall_mask, (100, 280), (339, 299), 255, -1)
    cv2.rectangle(wall_mask, (100, 100), (119, 299), 255, -1)
    return wall_mask.copy(), wall_mask


def complete(ink, wall_mask, walls=()):
    regions = find_rooms(ink, min_area_px=2000, min_inradius_px=20)
    covered = covered_mask(wall_mask, list(walls))
    return complete_boundaries(ink, covered, regions.contours, regions.outside, CONFIG)


def test_a_room_mostly_bounded_by_a_single_line_is_a_balcony():
    """Thick wall only on the north side (the building); a thin railing on the other three."""
    wall_mask = np.zeros((400, 500), np.uint8)
    cv2.rectangle(wall_mask, (100, 100), (399, 119), 255, -1)
    ink = wall_mask.copy()
    cv2.rectangle(ink, (120, 119), (379, 260), 255, 1)

    result = complete(ink, wall_mask)

    assert len(result.walls) == 3
    assert all(w.height_cm == 125 and w.thickness == 4 and w.exterior for w in result.walls)
    assert result.balconies == frozenset({0})


def test_a_double_line_railing_is_still_a_railing():
    wall_mask = np.zeros((400, 500), np.uint8)
    cv2.rectangle(wall_mask, (100, 100), (399, 119), 255, -1)
    ink = wall_mask.copy()
    for inset in (0, 5):
        cv2.rectangle(ink, (120 + inset, 119), (379 - inset, 260 - inset), 255, 1)

    result = complete(ink, wall_mask)

    assert result.balconies == frozenset({0})
    assert all(w.height_cm == 125 and not w.openings for w in result.walls)


def test_a_single_thin_line_on_one_side_of_an_ordinary_room_is_left_alone():
    ink, wall_mask = box_with_open_east_side()
    cv2.line(ink, (339, 100), (339, 299), 255, 1)
    result = complete(ink, wall_mask)
    assert result.walls == ()
    assert result.balconies == frozenset()


def test_parallel_lines_facing_outside_are_a_window_wall():
    ink, wall_mask = box_with_open_east_side()
    for x in (330, 339, 348):
        cv2.line(ink, (x, 100), (x, 299), 255, 1)

    result = complete(ink, wall_mask)

    [wall] = result.walls
    assert wall.height_cm is None
    assert wall.exterior
    assert wall.thickness == 20
    assert [o.kind for o in wall.openings] == ["window"]
    length = math.dist(wall.start, wall.end)
    assert wall.openings[0].width == pytest.approx(length, abs=20)
    assert result.balconies == frozenset()


def test_angled_glazing_is_found_too():
    wall_mask = np.zeros((500, 500), np.uint8)
    cv2.rectangle(wall_mask, (100, 250), (399, 269), 255, -1)  # south
    cv2.rectangle(wall_mask, (380, 100), (399, 269), 255, -1)  # east
    cv2.rectangle(wall_mask, (250, 100), (399, 119), 255, -1)  # north
    cv2.rectangle(wall_mask, (100, 200), (119, 269), 255, -1)  # west stub
    ink = wall_mask.copy()
    for offset in (-6, 0, 6):
        cv2.line(ink, (110 + offset, 205 + offset), (255 + offset, 110 + offset), 255, 1)

    [wall] = complete(ink, wall_mask).walls

    angle = math.degrees(math.atan2(wall.end[1] - wall.start[1], wall.end[0] - wall.start[0])) % 180
    assert 20 < angle < 70 or 110 < angle < 160
    assert [o.kind for o in wall.openings] == ["window"]


def test_thin_lines_between_two_rooms_become_a_door_wall():
    wall_mask = np.zeros((400, 600), np.uint8)
    cv2.rectangle(wall_mask, (100, 100), (499, 299), 255, 20)
    cv2.rectangle(wall_mask, (295, 100), (304, 170), 255, -1)  # wall stub from the north
    cv2.rectangle(wall_mask, (295, 250), (304, 299), 255, -1)  # wall stub from the south
    ink = wall_mask.copy()
    # Door leaf drawn open across the opening: a wedge from one stub end to the other.
    cv2.polylines(ink, [np.array([(300, 171), (340, 210), (300, 249)], np.int32)], False, 255, 1)

    result = complete(ink, wall_mask)

    [door_wall] = result.walls
    assert [o.kind for o in door_wall.openings] == ["door"]
    assert not door_wall.exterior
    assert door_wall.thickness == 12
    assert math.dist(door_wall.start, door_wall.end) == pytest.approx(80, abs=25)


def test_boundaries_already_explained_by_walls_are_left_alone():
    ink, wall_mask = box_with_open_east_side()
    cv2.rectangle(ink, (320, 100), (339, 299), 255, -1)
    east = PxWall(start=(330, 110), end=(330, 290), thickness=20)
    assert complete(ink, wall_mask, [east]).walls == ()


def test_probe_does_not_see_through_walls():
    """A thin line inside a room, in front of a thick wall, must not become a wall."""
    ink, wall_mask = box_with_open_east_side()
    cv2.rectangle(wall_mask, (320, 100), (339, 299), 255, -1)
    ink = wall_mask.copy()
    cv2.rectangle(ink, (280, 150), (318, 200), 255, 1)  # cupboard outline next to the east wall
    assert complete(ink, wall_mask).walls == ()


def enclose(ink, walls, balconies=frozenset()):
    regions = find_rooms(ink, min_area_px=2000, min_inradius_px=20)
    return enclose_rooms(ink, list(walls), regions.contours, balconies, regions.outside, CONFIG)


WEST = PxWall(start=(110, 100), end=(110, 299), thickness=20)
NORTH = PxWall(start=(100, 110), end=(339, 110), thickness=20)
SOUTH = PxWall(start=(100, 290), end=(339, 290), thickness=20)


def test_a_drawn_wall_that_never_became_a_wall_is_closed():
    """Thick strokes all round, but the parser only turned three sides into walls."""
    ink, _ = box_with_open_east_side()
    cv2.rectangle(ink, (320, 100), (339, 299), 255, -1)

    [wall] = enclose(ink, [WEST, NORTH, SOUTH])

    assert wall.exterior and not wall.openings and wall.height_cm is None
    assert wall.thickness == 20
    assert [wall.start[0], wall.end[0]] == pytest.approx([330, 330], abs=2)
    assert sorted(y for _, y in (wall.start, wall.end)) == pytest.approx([120, 280], abs=6)


def test_a_single_line_facing_outside_is_closed_too():
    ink, _ = box_with_open_east_side()
    cv2.line(ink, (339, 100), (339, 299), 255, 1)
    [wall] = enclose(ink, [WEST, NORTH, SOUTH])
    assert wall.exterior and not wall.openings


def test_closing_glazing_keeps_the_window():
    ink, _ = box_with_open_east_side()
    for x in (330, 339, 348):
        cv2.line(ink, (x, 100), (x, 299), 255, 1)
    [wall] = enclose(ink, [WEST, NORTH, SOUTH])
    assert [o.kind for o in wall.openings] == ["window"]


def test_a_balcony_is_closed_with_a_railing():
    ink, _ = box_with_open_east_side()
    cv2.line(ink, (339, 100), (339, 299), 255, 1)
    [wall] = enclose(ink, [WEST, NORTH, SOUTH], balconies=frozenset({0}))
    assert wall.height_cm == 125 and wall.thickness == 4


def test_rooms_already_enclosed_by_walls_get_nothing():
    ink, _ = box_with_open_east_side()
    cv2.rectangle(ink, (320, 100), (339, 299), 255, -1)
    east = PxWall(start=(330, 100), end=(330, 299), thickness=20)
    assert enclose(ink, [WEST, NORTH, SOUTH, east]) == ()


def test_an_opening_between_two_rooms_is_not_closed():
    """Open plans are real: only stretches facing outside must have walls."""
    ink = np.zeros((400, 600), np.uint8)
    cv2.rectangle(ink, (100, 100), (499, 299), 255, 20)
    cv2.line(ink, (300, 110), (300, 290), 255, 1)  # thin divider, e.g. a floor change
    walls = [
        PxWall(start=(100, 110), end=(499, 110), thickness=20),
        PxWall(start=(100, 290), end=(499, 290), thickness=20),
        PxWall(start=(110, 100), end=(110, 299), thickness=20),
        PxWall(start=(490, 100), end=(490, 299), thickness=20),
    ]
    assert enclose(ink, walls) == ()


def test_a_closing_wall_fills_the_drawn_wall_from_the_room_outwards():
    """A drawn wall 16 px deep: the new wall spans it, not the plan's thickest wall (20 px)."""
    ink, _ = box_with_open_east_side()
    cv2.rectangle(ink, (324, 100), (339, 299), 255, -1)
    [wall] = enclose(ink, [WEST, NORTH, SOUTH])
    assert wall.thickness == pytest.approx(16, abs=2)
    assert wall.start[0] - wall.thickness / 2 == pytest.approx(324, abs=2)


def test_a_single_line_gets_at_least_an_ordinary_wall_thickness():
    ink, _ = box_with_open_east_side()
    cv2.line(ink, (339, 100), (339, 299), 255, 1)
    [wall] = enclose(ink, [WEST, NORTH, SOUTH])
    assert wall.thickness == 12
