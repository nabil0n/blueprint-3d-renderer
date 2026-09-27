"""A tiny stand-in for the CubiCasa ONNX model, with the same inputs, outputs and class layout.

Rooms: dark pixels are "Wall", light ones "Background". Icons: always "No Icon". Enough to test the
wrapper (preprocessing, padding, resizing, loading) without the real 70 MB model.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from blueprint3d.parsing.learned.segmentation import MANIFEST_FILE, MODEL_FILE

ROOM_CLASSES = [
    "Background", "Outdoor", "Wall", "Kitchen", "Living Room", "Bed Room",
    "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined",
]  # fmt: skip
ICON_CLASSES = [
    "No Icon", "Window", "Door", "Closet", "Electrical Appliance", "Toilet",
    "Sink", "Sauna Bench", "Fire Place", "Bathtub", "Chimney",
]  # fmt: skip
SIZE_MULTIPLE = 64


def _conv(name: str, weights: np.ndarray, bias: np.ndarray) -> tuple[onnx.NodeProto, list[onnx.TensorProto]]:
    w = numpy_helper.from_array(weights.astype(np.float32), f"{name}_w")
    b = numpy_helper.from_array(bias.astype(np.float32), f"{name}_b")
    return helper.make_node("Conv", ["image", w.name, b.name], [f"{name}_logits"]), [w, b]


def write_toy_model(directory: Path) -> None:
    rooms_w = np.zeros((len(ROOM_CLASSES), 3, 1, 1))
    rooms_b = np.full(len(ROOM_CLASSES), -20.0)
    rooms_w[0, :, 0, 0], rooms_b[0] = 10.0, 0.0  # light -> Background
    rooms_w[2, :, 0, 0], rooms_b[2] = -10.0, 0.0  # dark -> Wall
    icons_b = np.full(len(ICON_CLASSES), -20.0)
    icons_b[0] = 0.0
    rooms_conv, rooms_init = _conv("rooms", rooms_w, rooms_b)
    icons_conv, icons_init = _conv("icons", np.zeros((len(ICON_CLASSES), 3, 1, 1)), icons_b)
    graph = helper.make_graph(
        [
            rooms_conv,
            icons_conv,
            helper.make_node("Softmax", ["rooms_logits"], ["rooms"], axis=1),
            helper.make_node("Softmax", ["icons_logits"], ["icons"], axis=1),
        ],
        "toy_cubicasa",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, "height", "width"])],
        [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, len(classes), "height", "width"])
            for name, classes in (("rooms", ROOM_CLASSES), ("icons", ICON_CLASSES))
        ],
        [*rooms_init, *icons_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, directory / MODEL_FILE)
    manifest = {
        "sha256": hashlib.sha256((directory / MODEL_FILE).read_bytes()).hexdigest(),
        "input": {"channels": "RGB", "range": [-1, 1], "pad_value": 255, "size_multiple": SIZE_MULTIPLE},
        "room_classes": ROOM_CLASSES,
        "icon_classes": ICON_CLASSES,
    }
    (directory / MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")
