"""Running the exported CubiCasa5K model: a plan image in, per-pixel room and icon classes out.

The model runs on this machine's CPU with onnxruntime, from a file loaded by explicit path. Nothing
is downloaded: without the file, the parser is unavailable (see scripts/setup_cubicasa.py).
"""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

MODEL_FILE = "cubicasa5k.onnx"
MANIFEST_FILE = "cubicasa5k.json"
DEFAULT_MODEL_DIR = Path(os.environ.get("BLUEPRINT3D_MODEL_DIR", "models"))


class ModelFileError(RuntimeError):
    """The model is missing or not the exported file."""


@dataclass(frozen=True)
class Segmentation:
    rooms: np.ndarray
    """uint8 class index per pixel, into `room_classes`."""
    icons: np.ndarray
    """uint8 class index per pixel, into `icon_classes`."""
    room_classes: tuple[str, ...]
    icon_classes: tuple[str, ...]

    def room_mask(self, name: str) -> np.ndarray:
        return self.rooms == self._index(self.room_classes, name)

    def icon_mask(self, name: str) -> np.ndarray:
        return self.icons == self._index(self.icon_classes, name)

    @staticmethod
    def _index(classes: tuple[str, ...], name: str) -> int:
        if name not in classes:
            raise KeyError(f"No class {name!r}; the model has {', '.join(classes)}.")
        return classes.index(name)


class SegmentationModel(Protocol):
    def segment(self, image: np.ndarray, scale: float = 1.0, rotations: int = 1) -> Segmentation: ...


def model_available(model_dir: Path = DEFAULT_MODEL_DIR) -> bool:
    return (model_dir / MODEL_FILE).is_file() and (model_dir / MANIFEST_FILE).is_file()


class CubiCasaModel:
    """The exported network (scripts/export_cubicasa_onnx.py) plus its manifest."""

    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR) -> None:
        import onnxruntime as ort

        model_path, manifest_path = model_dir / MODEL_FILE, model_dir / MANIFEST_FILE
        if not model_available(model_dir):
            raise ModelFileError(
                f"No model in {model_dir}. Run scripts/setup_cubicasa.py and scripts/export_cubicasa_onnx.py."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        data = model_path.read_bytes()
        if hashlib.sha256(data).hexdigest() != manifest["sha256"]:
            raise ModelFileError(
                f"{model_path} does not match the SHA-256 in {manifest_path}; export it again."
            )
        self._session = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
        self._multiple = int(manifest["input"]["size_multiple"])
        self._pad_value = int(manifest["input"]["pad_value"])
        self._room_classes = tuple(manifest["room_classes"])
        self._icon_classes = tuple(manifest["icon_classes"])

    def segment(self, image: np.ndarray, scale: float = 1.0, rotations: int = 1) -> Segmentation:
        """Classes for every pixel of `image` (grayscale or BGR). `scale` resizes the image for the
        network only, e.g. to bring walls to the thickness it was trained on; the result is on the
        original pixel grid. `rotations` (1-4) averages the network's probabilities over that many
        90-degree turns of the input, as CubiCasa's authors do: steadier, but that many times slower."""
        if not 1 <= rotations <= 4:
            raise ValueError("rotations must be between 1 and 4")
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image
        height, width = bgr.shape[:2]
        if scale != 1.0:
            interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
            bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=interpolation)
        rooms, icons = self._probabilities(bgr)
        for turns in range(1, rotations):
            turned_rooms, turned_icons = self._probabilities(np.ascontiguousarray(np.rot90(bgr, turns)))
            rooms += np.rot90(turned_rooms, -turns, axes=(1, 2))
            icons += np.rot90(turned_icons, -turns, axes=(1, 2))
        return Segmentation(
            rooms=_classes(rooms, width, height),
            icons=_classes(icons, width, height),
            room_classes=self._room_classes,
            icon_classes=self._icon_classes,
        )

    def _probabilities(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Class probabilities (classes x H x W) at the size of `bgr`. The input is padded to the
        network's size multiple, RGB, scaled to [-1, 1] as in CubiCasa's own loader."""
        height, width = bgr.shape[:2]
        pad, multiple = self._pad_value, self._multiple
        padded = cv2.copyMakeBorder(
            bgr, 0, -height % multiple, 0, -width % multiple, cv2.BORDER_CONSTANT, value=(pad, pad, pad)
        )
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32)
        batch = (2 * rgb / 255 - 1).transpose(2, 0, 1)[None]
        rooms, icons = self._session.run(["rooms", "icons"], {"image": batch})
        return rooms[0, :, :height, :width], icons[0, :, :height, :width]


def _classes(probabilities: np.ndarray, width: int, height: int) -> np.ndarray:
    """Most likely class per pixel, resized (nearest) to the original grid."""
    classes = probabilities.argmax(axis=0).astype(np.uint8)
    return cv2.resize(classes, (width, height), interpolation=cv2.INTER_NEAREST)
