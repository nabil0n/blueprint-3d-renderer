import cv2
import numpy as np

from blueprint3d.parsing.geometry import WallRun
from blueprint3d.parsing.rooms import find_rooms, gap_rects, merge_room_passes


def two_room_barrier():
    """Walls as 255: a 400x300 box split in two, plus a closet too small to be a room."""
    barrier = np.zeros((400, 500), np.uint8)
    cv2.rectangle(barrier, (50, 50), (449, 349), 255, 10)
    cv2.rectangle(barrier, (245, 50), (254, 349), 255, -1)
    cv2.rectangle(barrier, (60, 60), (90, 90), 255, 1)  # tiny closet in the corner
    return barrier


def test_finds_enclosed_rooms_and_skips_small_regions():
    regions = find_rooms(two_room_barrier(), min_area_px=2000, min_inradius_px=10)
    assert len(regions.polygons) == 2
    areas = sorted(cv2.contourArea(np.array(p, np.float32)) for p in regions.polygons)
    assert areas[0] > 150 * 250


def test_skips_regions_too_narrow_to_stand_in():
    barrier = np.zeros((300, 600), np.uint8)
    cv2.rectangle(barrier, (50, 50), (549, 80), 255, 5)  # long, 20 px wide corridor-like strip
    assert find_rooms(barrier, min_area_px=1000, min_inradius_px=15).polygons == ()


def test_merge_adds_fallback_rooms_only_where_primary_saw_outside():
    thick_only = np.zeros((400, 700), np.uint8)
    cv2.rectangle(thick_only, (50, 50), (449, 349), 255, 10)
    cv2.rectangle(thick_only, (450, 150), (649, 349), 255, 10)
    thick_only[160:340, 640:660] = 0  # east wall of the annex is only a thin line (e.g. glazing)
    with_thin = thick_only.copy()
    cv2.line(with_thin, (645, 150), (645, 349), 255, 1)

    primary = find_rooms(thick_only, min_area_px=2000, min_inradius_px=10)
    fallback = find_rooms(with_thin, min_area_px=2000, min_inradius_px=10)
    merged = merge_room_passes(primary, fallback)

    assert len(primary.polygons) == 1
    assert len(merged.polygons) == 2
    assert merged.outside is primary.outside


def test_outside_region_is_marked():
    regions = find_rooms(two_room_barrier(), min_area_px=2000, min_inradius_px=10)
    assert regions.outside[5, 5]
    assert not regions.outside[200, 150]


def test_gap_rects_cover_openings_across_the_wall():
    run = WallRun(axis="h", center=110, thickness=20, start=0, end=400, gaps=((100, 160),))
    [(x0, y0, x1, y1)] = gap_rects([run], margin=2)
    assert (x0, x1) == (100, 160)
    assert y0 <= 100 and y1 >= 120

    vertical = WallRun(axis="v", center=50, thickness=10, start=0, end=300, gaps=((20, 80),))
    [(x0, y0, x1, y1)] = gap_rects([vertical], margin=0)
    assert (x0, y0, x1, y1) == (45, 20, 55, 80)
