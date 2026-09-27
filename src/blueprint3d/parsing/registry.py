"""The parsers this backend can run. The learned one needs its exported model (a one-time setup)."""

from dataclasses import dataclass
from pathlib import Path

from blueprint3d.parsing.learned.learned_parser import LearnedParser
from blueprint3d.parsing.learned.segmentation import DEFAULT_MODEL_DIR, model_available
from blueprint3d.parsing.opencv_parser import OpenCvParser
from blueprint3d.parsing.pipeline import Parser

DEFAULT_PARSER = "opencv"


@dataclass(frozen=True)
class ParserInfo:
    name: str
    available: bool
    description: str


def available_parsers(model_dir: Path = DEFAULT_MODEL_DIR) -> dict[str, Parser]:
    """Checked on every call (a file lookup): exporting the model makes it available without a restart."""
    parsers: dict[str, Parser] = {"opencv": OpenCvParser()}
    if model_available(model_dir):
        parsers["cubicasa"] = LearnedParser(model_dir=model_dir)
    return parsers


def parser_catalogue(model_dir: Path = DEFAULT_MODEL_DIR) -> list[ParserInfo]:
    return [
        ParserInfo("opencv", True, "Classical image processing, tuned on real plans."),
        ParserInfo(
            "cubicasa",
            model_available(model_dir),
            "CubiCasa5K's pretrained segmentation model plus our own vectoriser. Needs a one-time "
            "setup: scripts/setup_cubicasa.py, then scripts/export_cubicasa_onnx.py.",
        ),
    ]
