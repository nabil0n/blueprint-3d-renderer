import cv2
import numpy as np
import pytest

from blueprint3d.parsing.errors import ParseError
from blueprint3d.parsing.learned.segmentation import Segmentation
from blueprint3d.parsing.learned.vectorize import vectorize_rooms, vectorize_walls
from tests.parsing.learned.toy_model import ICON_CLASSES, ROOM_CLASSES

CM_PER_PX = 2.0
ROOM = {name: i for i, name in enumerate(ROOM_CLASSES)}
ICON = {name: i for i, name in enumerate(ICON_CLASSES)}


class ClassMap:
    """Draws class maps like the model's output: rectangles are inclusive (x0, y0, x1, y1)."""

    def __init__(self, width: int = 400, height: int = 360) -> None:
        self.rooms = np.full((height, width), ROOM["Background"], np.uint8)
        self.icons = np.full((height, width), ICON["No Icon"], np.uint8)

    def room(self, name: str, x0: int, y0: int, x1: int, y1: int) -> "ClassMap":
        self.rooms[y0 : y1 + 1, x0 : x1 + 1] = ROOM[name]
        return self

    def icon(self, name: str, x0: int, y0: int, x1: int, y1: int) -> "ClassMap":
        """An opening: the icon replaces the wall there, as in CubiCasa's labels."""
        self.rooms[y0 : y1 + 1, x0 : x1 + 1] = ROOM["Background"]
        self.icons[y0 : y1 + 1, x0 : x1 + 1] = ICON[name]
        return self

    def segmentation(self) -> Segmentation:
        return Segmentation(self.rooms, self.icons, tuple(ROOM_CLASSES), tuple(ICON_CLASSES))


def two_rooms() -> ClassMap:
    """Kitchen (left) and bedroom (right) in a 300x200 box with 10 px walls. A door between them,
    a window in the north wall, and an open passage (a gap with no icon) in the south wall."""
    m = ClassMap()
    m.room("Kitchen", 60, 60, 194, 239).room("Bed Room", 205, 60, 339, 239)
    north, south, west, east = (50, 50, 349, 59), (50, 240, 349, 249), (50, 50, 59, 249), (340, 50, 349, 249)
    for rect in (north, south, west, east, (195, 60, 204, 239)):
        m.room("Wall", *rect)
    m.icon("Door", 195, 120, 204, 169)
    m.icon("Window", 80, 50, 149, 59)
    m.rooms[240:250, 250:290] = ROOM["Background"]  # passage
    return m


def walls_by_axis(walls):
    horizontal = sorted((w for w in walls if w.start[1] == w.end[1]), key=lambda w: w.start[1])
    vertical = sorted((w for w in walls if w.start[0] == w.end[0]), key=lambda w: w.start[0])
    return horizontal, vertical


def test_walls_run_through_their_openings():
    result = vectorize_walls(two_rooms().segmentation())
    horizontal, vertical = walls_by_axis(result.walls)
    assert len(horizontal) == 2 and len(vertical) == 3
    assert result.stats.typical == pytest.approx(10, abs=1.5)


def test_openings_take_their_kind_from_the_icons():
    horizontal, vertical = walls_by_axis(vectorize_walls(two_rooms().segmentation()).walls)
    north, south = horizontal
    interior = vertical[1]
    assert [(o.kind, round(o.width)) for o in north.openings] == [("window", 70)]
    assert [(o.kind, round(o.width)) for o in interior.openings] == [("door", 50)]
    assert [(o.kind, round(o.width)) for o in south.openings] == [("door", 40)], "a bare gap is a passage"


def test_opening_offsets_are_measured_from_the_wall_start():
    north = walls_by_axis(vectorize_walls(two_rooms().segmentation()).walls)[0][0]
    (window,) = north.openings
    assert north.start[0] + window.offset == pytest.approx(115, abs=1.5)


def test_rooms_are_separated_by_walls_and_closed_openings():
    seg = two_rooms().segmentation()
    rooms = vectorize_rooms(seg, vectorize_walls(seg), CM_PER_PX)
    assert len(rooms.regions.polygons) == 2
    assert sorted(rooms.kinds.values()) == ["bedroom", "kitchen"]
    assert rooms.balconies == frozenset()


def test_a_room_without_a_clear_class_gets_no_guess():
    m = two_rooms().room("Undefined", 60, 60, 194, 239)
    seg = m.segmentation()
    rooms = vectorize_rooms(seg, vectorize_walls(seg), CM_PER_PX)
    assert sorted(rooms.kinds.values()) == ["bedroom"]


def test_an_outdoor_space_behind_a_railing_is_a_balcony():
    m = two_rooms().icon("Door", 100, 240, 149, 249)
    m.room("Outdoor", 70, 250, 229, 319)
    m.room("Railing", 70, 318, 229, 320).room("Railing", 68, 250, 70, 320).room("Railing", 229, 250, 231, 320)
    seg = m.segmentation()
    rooms = vectorize_rooms(seg, vectorize_walls(seg), CM_PER_PX)
    assert len(rooms.regions.polygons) == 3
    (balcony,) = rooms.balconies
    assert rooms.kinds[balcony] == "balcony"
    assert len(rooms.railings) == 3
    assert all(r.height_cm == 125 and r.thickness == pytest.approx(5 / CM_PER_PX) for r in rooms.railings)


def test_an_angled_wall_is_kept():
    m = ClassMap()
    cv2.line(m.rooms, (60, 300), (300, 60), ROOM["Wall"], 10)
    walls = vectorize_walls(m.segmentation()).walls
    assert len(walls) == 1
    (x0, y0), (x1, y1) = walls[0].start, walls[0].end
    assert abs(x1 - x0) == pytest.approx(abs(y1 - y0), rel=0.1), "45 degrees"


def test_no_walls_is_a_parse_error():
    with pytest.raises(ParseError, match="no walls"):
        vectorize_walls(ClassMap().segmentation())
