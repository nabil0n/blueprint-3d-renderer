"""Decoding uploaded images into a normalised grayscale array."""

from dataclasses import dataclass

import cv2
import numpy as np

from blueprint3d.parsing.errors import ParseError

MAX_SIDE_PX = 2500
COLOUR_CHROMA = 40
"""Pixels whose channels spread more than this are coloured (logos, fills) and treated as paper;
plan lines are black or gray. Near-black JPEG noise stays well below it."""
LINE_CONTRAST = 30
"""How much darker than its immediate surroundings a pixel must be to count as a drawn line."""
LINE_KERNEL_PX = 9
LIGHT_INK_LEVEL = 230
"""Gray levels below this count as (light) ink."""
FILL_KERNEL_PX = 15
"""Light areas at least this wide in every direction are fills, not lines."""


@dataclass(frozen=True)
class DecodedImage:
    gray: np.ndarray
    """Working image: uint8 grayscale, possibly downscaled."""
    resize_factor: float
    """Working pixels per original pixel (<= 1)."""
    original_size: tuple[int, int]
    """(width, height) of the uploaded image."""


def decode_image(data: bytes, max_side: int = MAX_SIDE_PX) -> DecodedImage:
    if not data:
        raise ParseError("The uploaded file is empty.")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ParseError("Could not decode the image. Use a PNG, JPEG or WebP file.")

    gray = _to_gray(image)
    height, width = gray.shape
    factor = min(1.0, max_side / max(width, height))
    if factor < 1.0:
        gray = cv2.resize(gray, (round(width * factor), round(height * factor)), interpolation=cv2.INTER_AREA)
    return DecodedImage(gray=gray, resize_factor=factor, original_size=(width, height))


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        # Composite onto white so transparent backgrounds do not turn black.
        alpha = image[:, :, 3:4].astype(np.float32) / 255
        image = (image[:, :, :3].astype(np.float32) * alpha + 255 * (1 - alpha)).astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    chroma = image.max(axis=2).astype(np.int16) - image.min(axis=2)
    return np.where(chroma > COLOUR_CHROMA, 255, gray).astype(np.uint8)


def binarize(gray: np.ndarray) -> np.ndarray:
    """Otsu threshold; returns dark (ink) pixels as 255 and background as 0."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return binary


def line_ink(gray: np.ndarray, dark: np.ndarray) -> np.ndarray:
    """Dark ink plus light gray lines, without flat room fills, as 255.

    Light pixels count as ink unless they belong to a large uniform light area (a fill such as a
    gray balcony or light blue bathroom). Lines drawn *inside* a fill are recovered with a black-hat
    filter, which finds strokes darker than their immediate surroundings.
    """
    light = (gray < LIGHT_INK_LEVEL) & (dark == 0)
    fill = cv2.morphologyEx(light.astype(np.uint8), cv2.MORPH_OPEN, np.ones((FILL_KERNEL_PX, FILL_KERNEL_PX), np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (LINE_KERNEL_PX, LINE_KERNEL_PX))
    in_fill_lines = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel) > LINE_CONTRAST
    return np.where((dark > 0) | (light & (fill == 0)) | in_fill_lines, 255, 0).astype(np.uint8)
