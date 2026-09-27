"""Reading printed text off a plan image. The engine is loaded once, on first use."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from pathlib import Path
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


MODEL_FILES = {
    "Det": "PP-OCRv6_det_small.onnx",
    "Cls": "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
    "Rec": "PP-OCRv6_rec_small.onnx",
}
"""The models bundled in the rapidocr wheel (text detection, orientation, recognition)."""


def model_paths() -> dict[str, Path]:
    return {stage: Path(str(files("rapidocr") / "models" / name)) for stage, name in MODEL_FILES.items()}


@cache
def _engine() -> Any:
    from rapidocr import RapidOCR  # imported lazily: loading the models takes a moment

    # Explicit model paths bypass RapidOCR's downloader: without them, a missing or corrupt model
    # would be fetched from the internet. With them, it is a local FileNotFoundError instead.
    paths = {f"{stage}.model_path": str(path) for stage, path in model_paths().items()}
    return RapidOCR(params={"Global.log_level": "critical", **paths})


def read_text(gray: np.ndarray) -> list[TextBox]:
    """RapidOCR: PP-OCR models on ONNX Runtime, run on this machine's CPU. The models ship with the
    package; nothing is downloaded or sent anywhere."""
    result = _engine()(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    if result.boxes is None or result.txts is None or result.scores is None:
        return []
    return [
        TextBox(text=text, score=float(score), center=tuple(float(v) for v in np.asarray(box).mean(axis=0)))
        for box, text, score in zip(result.boxes, result.txts, result.scores, strict=True)
    ]
