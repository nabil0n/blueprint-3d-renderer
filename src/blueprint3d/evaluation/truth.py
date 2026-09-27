"""Ground truth per sample image, kept in a JSON file keyed by file name.

Only facts that can be read off the drawing without tracing it:
    {
      "drheymansgata5.jpg": {
        "cm_per_px": 1.488,          # from the printed scale bar
        "listed_area_m2": 75,        # printed living area (excludes balconies), if shown
        "rooms": 7,                  # enclosed spaces, including balconies
        "room_tolerance": 0,         # allowed +/- where the drawing is ambiguous (open plans)
        "balconies": 1,
        "notes": "..."
      }
    }
"""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class SampleTruth(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    cm_per_px: float = Field(gt=0, description="True scale in centimetres per original image pixel.")
    listed_area_m2: float | None = Field(default=None, gt=0)
    rooms: int = Field(ge=0)
    room_tolerance: int = Field(default=0, ge=0)
    balconies: int = Field(default=0, ge=0)
    notes: str = ""


_TRUTH_FILE = TypeAdapter(dict[str, SampleTruth])


def load_truth(path: Path) -> dict[str, SampleTruth]:
    """Truth per file name; empty if the file does not exist."""
    if not path.is_file():
        return {}
    return _TRUTH_FILE.validate_python(json.loads(path.read_text(encoding="utf-8")))
