"""Export CubiCasa5K's pretrained network to ONNX, so the backend can run it with onnxruntime alone.

    uv run --group ml python scripts/export_cubicasa_onnx.py [--check data/some_plan.jpg]

Needs scripts/setup_cubicasa.py first. Writes models/cubicasa5k.onnx and models/cubicasa5k.json
(class lists and input conventions for the parser). Runs on the CPU; no network access.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
VENDOR_DIR = ROOT / "vendor" / "cubicasa5k"
WEIGHTS_FILE = ROOT / "models" / "cubicasa5k_weights.pkl"
ONNX_FILE = ROOT / "models" / "cubicasa5k.onnx"
MANIFEST_FILE = ROOT / "models" / "cubicasa5k.json"

HEATMAP_CHANNELS, ROOM_CHANNELS, ICON_CHANNELS = 21, 12, 11
ROOM_CLASSES = [
    "Background", "Outdoor", "Wall", "Kitchen", "Living Room", "Bed Room",
    "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined",
]  # fmt: skip
ICON_CLASSES = [
    "No Icon", "Window", "Door", "Closet", "Electrical Appliance", "Toilet",
    "Sink", "Sauna Bench", "Fire Place", "Bathtub", "Chimney",
]  # fmt: skip
SIZE_MULTIPLE = 64
"""A strided first convolution plus five 2x poolings: sides divisible by this line up exactly."""
MAX_DIFFERENCE = 1e-3


class RoomsAndIcons(nn.Module):
    """The network with its heads turned into class probabilities; the junction heatmaps (used by
    CubiCasa's own vectoriser, not ours) are dropped."""

    def __init__(self, network: nn.Module) -> None:
        super().__init__()
        self.network = network

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.network(image)
        rooms = out[:, HEATMAP_CHANNELS : HEATMAP_CHANNELS + ROOM_CHANNELS]
        icons = out[:, HEATMAP_CHANNELS + ROOM_CHANNELS :]
        return torch.softmax(rooms, dim=1), torch.softmax(icons, dim=1)


def build_network() -> nn.Module:
    """As in CubiCasa's samples.ipynb, minus init_weights(): it loads a second pickle and every
    weight is overwritten by the checkpoint anyway."""
    sys.path.insert(0, str(VENDOR_DIR))
    from floortrans.models.hg_furukawa_original import hg_furukawa_original

    classes = HEATMAP_CHANNELS + ROOM_CHANNELS + ICON_CHANNELS
    network = hg_furukawa_original(n_classes=51)
    network.conv4_ = nn.Conv2d(256, classes, bias=True, kernel_size=1)
    network.upsample = nn.ConvTranspose2d(classes, classes, kernel_size=4, stride=4)
    # weights_only: the checkpoint is a pickle; this refuses to run any code stored in it.
    checkpoint = torch.load(WEIGHTS_FILE, map_location="cpu", weights_only=True)
    network.load_state_dict(checkpoint["model_state"])
    return network.eval()


def to_input(image_bgr: np.ndarray) -> np.ndarray:
    """RGB scaled to [-1, 1] as CubiCasa's loader does, NCHW, padded white to SIZE_MULTIPLE."""
    height, width = image_bgr.shape[:2]
    pad_h, pad_w = -height % SIZE_MULTIPLE, -width % SIZE_MULTIPLE
    padded = cv2.copyMakeBorder(image_bgr, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32)
    return (2 * rgb / 255 - 1).transpose(2, 0, 1)[None]


def export(model: nn.Module) -> None:
    sample = torch.from_numpy(to_input(np.full((256, 320, 3), 255, np.uint8)))
    torch.onnx.export(
        model,
        (sample,),
        str(ONNX_FILE),
        input_names=["image"],
        output_names=["rooms", "icons"],
        dynamic_axes={"image": {2: "height", 3: "width"}, "rooms": {2: "height", 3: "width"},
                      "icons": {2: "height", 3: "width"}},
        opset_version=17,
        dynamo=False,
    )  # fmt: skip


def check(model: nn.Module, image_path: Path) -> float:
    """Largest probability difference between PyTorch and ONNX Runtime on a real plan."""
    import onnxruntime as ort

    image = cv2.imdecode(np.fromfile(image_path, np.uint8), cv2.IMREAD_COLOR)
    scale = 1000 / max(image.shape[:2])
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    batch = to_input(image)
    with torch.no_grad():
        expected = [t.numpy() for t in model(torch.from_numpy(batch))]
    session = ort.InferenceSession(str(ONNX_FILE), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"image": batch})
    return max(float(np.abs(a - e).max()) for a, e in zip(actual, expected, strict=True))


def write_manifest() -> None:
    digest = hashlib.sha256(ONNX_FILE.read_bytes()).hexdigest()
    manifest = {
        "model": "CubiCasa5K hg_furukawa_original, pretrained (CC BY-NC 4.0)",
        "sha256": digest,
        "input": {"channels": "RGB", "range": [-1, 1], "pad_value": 255, "size_multiple": SIZE_MULTIPLE},
        "room_classes": ROOM_CLASSES,
        "icon_classes": ICON_CLASSES,
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=formatter)
    parser.add_argument("--check", type=Path, help="a plan image to compare PyTorch and ONNX outputs on")
    args = parser.parse_args()
    if not WEIGHTS_FILE.is_file() or not VENDOR_DIR.is_dir():
        sys.exit("Run scripts/setup_cubicasa.py first.")

    model = RoomsAndIcons(build_network()).eval()
    export(model)
    write_manifest()
    print(f"Wrote {ONNX_FILE.relative_to(ROOT)} and {MANIFEST_FILE.relative_to(ROOT)}")
    if args.check:
        difference = check(model, args.check)
        print(f"Max probability difference PyTorch vs ONNX on {args.check.name}: {difference:.2e}")
        if difference > MAX_DIFFERENCE:
            sys.exit(f"Outputs differ by more than {MAX_DIFFERENCE}.")


if __name__ == "__main__":
    main()
