"""Reading printed text off a plan image. The engine is loaded once, on first use."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class TextBox:
    text: str
    score: float
    """Recognition confidence, 0..1."""
    center: tuple[float, float]
    """(x, y) in pixels of the image that was read."""


TextReader = Callable[[np.ndarray], list[TextBox]]
"""Takes a grayscale image, returns the text lines found in it."""


@cache
def _engine() -> Any:
    from rapidocr import RapidOCR  # imported lazily: loading the models takes a moment

    return RapidOCR(params={"Global.log_level": "critical"})


def read_text(gray: np.ndarray) -> list[TextBox]:
    """RapidOCR (PP-OCR models on ONNX Runtime, bundled with the package: no downloads)."""
    result = _engine()(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    if result.boxes is None or result.txts is None or result.scores is None:
        return []
    return [
        TextBox(text=text, score=float(score), center=tuple(float(v) for v in np.asarray(box).mean(axis=0)))
        for box, text, score in zip(result.boxes, result.txts, result.scores, strict=True)
    ]
