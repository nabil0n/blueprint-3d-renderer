import cv2
import numpy as np
import pytest

from blueprint3d.parsing.errors import ParseError
from blueprint3d.parsing.walls import crop_box, expand_to_ink, extract_wall_mask, fill_hatching, ink_box, thickness_stats
from tests.parsing.synthetic import (
    BLACK,
    EXTERIOR_PX,
    INTERIOR_PX,
    blank,
    draw_hatched_plan_with_bold_title,
    draw_two_room_plan,
    to_binary,
)


def test_keeps_thick_walls_and_drops_thin_strokes():
    img = blank(400, 300)
    cv2.rectangle(img, (50, 50), (349, 64), BLACK, -1)  # 15 px wall
    cv2.line(img, (50, 150), (349, 150), BLACK, 1)  # thin furniture line
    cv2.putText(img, "Kitchen", (60, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, BLACK, 2)

    walls = extract_wall_mask(to_binary(img))

    assert walls.mask[57, 200] == 255
    assert walls.mask[150, 200] == 0
    assert walls.mask[200:260, :].max() == 0


def test_thin_walls_count_when_thick_walls_dominate():
    """Otsu alone would split between the 8 px and 24 px walls; the first width jump is after text."""
    img = blank(600, 400)
    for y in (40, 340):
        cv2.rectangle(img, (40, y), (559, y + 23), BLACK, -1)  # 24 px exterior walls
    cv2.rectangle(img, (40, 40), (63, 363), BLACK, -1)
    cv2.rectangle(img, (296, 64), (303, 339), BLACK, -1)  # 8 px partition
    for i in range(12):
        cv2.putText(img, "Sovrum", (80 + (i % 3) * 70, 120 + (i // 3) * 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 1)

    walls = extract_wall_mask(to_binary(img))

    assert walls.mask[200, 300] == 255
    assert walls.mask[100:300, 80:280].max() == 0


def test_six_pixel_partitions_count_when_text_stops_at_four():
    """Stroke widths 2-4 (text), a hole at 5, then 6 px partitions and thick walls (dahlströmsgatan)."""
    img = blank(700, 400)
    for y in (40, 340):
        cv2.rectangle(img, (40, y), (659, y + 31), BLACK, -1)  # 32 px exterior walls
    cv2.rectangle(img, (40, 40), (71, 371), BLACK, -1)
    cv2.rectangle(img, (400, 72), (405, 339), BLACK, -1)  # 6 px partition
    for i in range(12):
        cv2.putText(img, "Sovrum", (90 + (i % 3) * 95, 120 + (i // 3) * 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, BLACK, 2)

    walls = extract_wall_mask(to_binary(img))

    assert walls.mask[200, 402] == 255
    assert walls.mask[90:300, 90:380].max() == 0


def test_wall_mask_is_aligned_with_the_input():
    """An even-sized opening kernel shifts the result by a pixel, which lets rooms leak past door plugs."""
    img = blank(400, 300)
    cv2.rectangle(img, (50, 50), (349, 57), BLACK, -1)  # 8 px wall
    cv2.putText(img, "text", (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, BLACK, 1)
    binary = to_binary(img)
    walls = extract_wall_mask(binary)
    assert walls.min_thickness % 2 == 1
    assert np.array_equal(walls.mask[40:70, 40:360] > 0, binary[40:70, 40:360] > 0)


def test_rejects_blank_image():
    with pytest.raises(ParseError, match="No dark strokes"):
        extract_wall_mask(np.zeros((100, 100), np.uint8))


def test_crop_box_excludes_page_decoration():
    walls = extract_wall_mask(to_binary(draw_two_room_plan()))
    x0, y0, x1, y1 = crop_box(walls.mask, walls.min_thickness)
    assert x0 < 100 and y0 < 100 and x1 > 900 and y1 > 700
    assert y0 > 60, "title should be outside the crop"
    assert x1 <= 960, "legend icon should be outside the crop"


def test_crop_box_ignores_a_bold_title_heavier_than_the_walls():
    plan = cv2.resize(draw_two_room_plan(), (500, 400), interpolation=cv2.INTER_AREA)
    img = blank(1000, 900)
    img[:400, :500] = plan
    cv2.putText(img, "Obj.nr 5403-0474", (20, 800), cv2.FONT_HERSHEY_SIMPLEX, 3.0, BLACK, 14)
    walls = extract_wall_mask(to_binary(img), partitions=False)
    box = crop_box(walls.mask, walls.min_thickness)
    assert box is not None
    x0, y0, x1, y1 = box
    assert x0 < 50 and y0 < 50 and x1 > 450 and y1 > 350
    assert y1 < 500, "the title has no long straight strokes and stays out"


def test_crop_box_ignores_a_scale_bar_and_a_scan_border_heavier_than_the_walls():
    plan = cv2.resize(draw_two_room_plan(), (500, 400), interpolation=cv2.INTER_AREA)
    img = blank(1000, 900)
    img[100:500, 250:750] = plan
    cv2.rectangle(img, (100, 700), (900, 730), BLACK, -1)  # a solid scale bar, long and thick
    cv2.rectangle(img, (0, 870), (999, 899), BLACK, -1)  # a black scan border along the page edge
    walls = extract_wall_mask(to_binary(img), partitions=False)
    box = crop_box(walls.mask, walls.min_thickness)
    assert box is not None
    x0, y0, x1, y1 = box
    assert x0 < 300 and y0 < 150 and x1 > 700 and y1 > 450
    assert y1 < 650, "straight strokes running one way only are not a plan"


def test_crop_box_finds_nothing_when_no_cluster_has_a_long_straight_wall():
    walls = extract_wall_mask(to_binary(draw_hatched_plan_with_bold_title()), partitions=False)
    assert crop_box(walls.mask, walls.min_thickness) is None


def test_ink_box_finds_the_largest_drawing_on_the_page():
    img = draw_hatched_plan_with_bold_title()
    cv2.rectangle(img, (2, 2), (997, 1097), BLACK, 1)  # a page frame is not the plan
    x0, y0, x1, y1 = ink_box(to_binary(img))
    assert x0 <= 100 and y0 <= 100 and x1 >= 900 and y1 >= 700
    assert x0 > 2 and y1 < 850


def test_fill_hatching_turns_hatched_walls_into_solid_ones():
    binary = to_binary(draw_hatched_plan_with_bold_title())
    x0, y0, x1, y1 = ink_box(binary)
    walls = extract_wall_mask(fill_hatching(binary[y0:y1, x0:x1]))
    at = lambda x, y: walls.mask[y - y0, x - x0]  # noqa: E731
    assert at(500, 110) and at(110, 400) and at(500, 250), "exterior and interior walls are solid"
    assert not at(300, 400) and not at(500, 408), "rooms and the door gap stay open"
    assert crop_box(walls.mask, walls.min_thickness) is not None


def test_expand_to_ink_includes_thin_outlines_attached_to_the_plan():
    img = draw_two_room_plan()
    cv2.rectangle(img, (40, 300), (100, 500), BLACK, 1)  # balcony outline against the west wall
    binary = to_binary(img)
    walls = extract_wall_mask(binary)
    box = crop_box(walls.mask, walls.min_thickness)
    x0, y0, x1, y1 = expand_to_ink(box, binary, walls.mask)
    assert x0 <= 40
    assert y0 > 60, "the unattached title stays out"


def test_expand_to_ink_ignores_a_page_frame():
    img = draw_two_room_plan()
    cv2.rectangle(img, (2, 2), (997, 797), BLACK, 1)
    cv2.line(img, (2, 400), (100, 400), BLACK, 1)  # frame touches the plan
    binary = to_binary(img)
    walls = extract_wall_mask(binary)
    box = crop_box(walls.mask, walls.min_thickness)
    assert expand_to_ink(box, binary, walls.mask) == box


def test_thickness_stats_reflect_wall_widths():
    walls = extract_wall_mask(to_binary(draw_two_room_plan()))
    stats = thickness_stats(walls.mask[80:720, 80:920])
    assert INTERIOR_PX - 2 <= stats.typical <= EXTERIOR_PX + 2
    assert stats.maximum == pytest.approx(EXTERIOR_PX, abs=3)
