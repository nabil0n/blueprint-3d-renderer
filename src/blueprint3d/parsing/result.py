from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from blueprint3d.schema import Plan


class ParseMeta(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    parser: str
    cm_per_px: float = Field(gt=0, description="Centimetres per pixel of the uploaded image.")
    scale_source: Literal["user", "doors", "wall_thickness"]
    scale_detail: str
    image_width: int
    image_height: int
    warnings: list[str] = Field(default_factory=list)


class ParseResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    plan: Plan
    meta: ParseMeta
