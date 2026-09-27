"""Decoding uploaded images into a normalised grayscale array."""

from dataclasses import dataclass

import cv2
import numpy as np

from blueprint3d.parsing.errors import ParseError

MAX_SIDE_PX = 2500


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
        rgb = image[:, :, :3].astype(np.float32) * alpha + 255 * (1 - alpha)
        return cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def binarize(gray: np.ndarray) -> np.ndarray:
    """Otsu threshold; returns dark (ink) pixels as 255 and background as 0."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return binary
