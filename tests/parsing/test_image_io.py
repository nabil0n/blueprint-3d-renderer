import cv2
import numpy as np
import pytest

from blueprint3d.parsing.errors import ParseError
from blueprint3d.parsing.image_io import binarize, decode_image
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
