import cv2
import numpy as np
import pytest

from blueprint3d.parsing.errors import ParseError
from blueprint3d.parsing.image_io import binarize, decode_image, line_ink
from tests.parsing.synthetic import blank, encode_png


def test_decodes_png_to_grayscale():
    decoded = decode_image(encode_png(blank(300, 200)))
    assert decoded.gray.shape == (200, 300)
    assert decoded.gray.dtype == np.uint8
    assert decoded.original_size == (300, 200)
    assert decoded.resize_factor == 1


def test_downscales_large_images_and_reports_factor():
    decoded = decode_image(encode_png(blank(4000, 1000)), max_side=2000)
    assert decoded.gray.shape == (500, 2000)
    assert decoded.resize_factor == pytest.approx(0.5)
    assert decoded.original_size == (4000, 1000)


def test_transparent_background_becomes_white():
    rgba = np.zeros((50, 50, 4), np.uint8)  # fully transparent black
    rgba[20:30, 20:30] = (0, 0, 0, 255)  # opaque black square
    ok, buf = cv2.imencode(".png", rgba)
    assert ok
    gray = decode_image(buf.tobytes()).gray
    assert gray[0, 0] == 255
    assert gray[25, 25] == 0


@pytest.mark.parametrize("data", [b"", b"definitely not an image"])
def test_rejects_undecodable_data(data):
    with pytest.raises(ParseError):
        decode_image(data)


def test_binarize_marks_dark_pixels():
    gray = np.full((10, 10), 250, np.uint8)
    gray[2:5, 2:5] = 20
    binary = binarize(gray)
    assert binary[3, 3] == 255
    assert binary[8, 8] == 0


def test_coloured_ink_is_treated_as_paper():
    img = np.full((40, 40, 3), 255, np.uint8)
    img[5:15, 5:15] = (20, 90, 30)  # dark green logo
    img[25:35, 25:35] = (15, 15, 15)  # black wall
    ok, buf = cv2.imencode(".png", img)
    assert ok
    gray = decode_image(buf.tobytes()).gray
    assert gray[10, 10] == 255
    assert gray[30, 30] < 50


def test_line_ink_keeps_lines_and_drops_flat_fills():
    gray = np.full((100, 100), 255, np.uint8)
    gray[10:90, 10:90] = 215  # light room fill
    gray[50, 20:80] = 170  # light gray line inside it
    gray[0:100, 95:100] = 0  # thick dark wall
    ink = line_ink(gray, binarize(gray))
    assert ink[50, 50] == 255
    assert ink[30, 50] == 0
    assert ink[50, 97] == 255


def test_line_ink_keeps_gray_lines_that_touch_black_walls():
    gray = np.full((60, 100), 255, np.uint8)
    gray[0:20, :] = 0  # wall
    gray[20:22, :] = 160  # railing line drawn right against it
    gray[22:60, 50] = 160  # and running away from it
    ink = line_ink(gray, binarize(gray))
    assert ink[20:22, 10:90].all()
    assert ink[22:60, 50].all()


def test_anti_aliased_edges_of_coloured_ink_are_removed_too():
    img = np.full((60, 200, 3), 255, np.uint8)
    cv2.putText(img, "LOGO", (10, 45), cv2.FONT_HERSHEY_DUPLEX, 1.5, (60, 110, 20), 4, cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    gray = decode_image(buf.tobytes()).gray
    assert gray.min() >= 230, "no halo pixels dark enough to count as ink"
