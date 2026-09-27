import cv2
import pytest

from blueprint3d.parsing.scale_bar import detect_scale_bar
from tests.parsing.synthetic import (
    BLACK,
    blank,
    draw_checker_scale_bar,
    draw_tick_scale_bar,
    draw_two_room_plan,
    to_binary,
)

NOWHERE = (0, 0, 0, 0)


@pytest.mark.parametrize("px_per_m", [67.3, 52.0, 120.4])
def test_reads_a_tick_bar(px_per_m):
    img = blank(1400, 400)
    draw_tick_scale_bar(img, 100, 300, px_per_m)
    bar = detect_scale_bar(to_binary(img), exclude=NOWHERE)
    assert bar is not None
    assert bar.px_per_step == pytest.approx(px_per_m, rel=0.02)
    assert bar.steps == 5


def test_reads_a_checker_bar():
    img = blank(1400, 400)
    draw_checker_scale_bar(img, 100, 300, 67.7)
    bar = detect_scale_bar(to_binary(img), exclude=NOWHERE)
    assert bar is not None
    assert bar.px_per_step == pytest.approx(67.7, rel=0.02)
    assert bar.steps >= 5


def test_no_bar_on_a_plain_plan():
    assert detect_scale_bar(to_binary(draw_two_room_plan()), exclude=NOWHERE) is None


def test_irregular_marks_on_a_line_are_not_a_bar():
    img = blank(1400, 400)
    cv2.line(img, (100, 300), (1100, 300), BLACK, 2)
    for x in (100, 180, 390, 420, 700, 1010):
        cv2.line(img, (x, 300), (x, 286), BLACK, 2)
    assert detect_scale_bar(to_binary(img), exclude=NOWHERE) is None


def test_ignores_the_plan_area():
    img = blank(1400, 400)
    draw_tick_scale_bar(img, 100, 300, 67.3)
    assert detect_scale_bar(to_binary(img), exclude=(0, 200, 1400, 400)) is None
