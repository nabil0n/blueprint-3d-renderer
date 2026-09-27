from pydantic import BaseModel, ConfigDict, Field

from blueprint3d.parsing.scale import ScaleSource
from blueprint3d.schema import Plan


class ParseMeta(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    parser: str
    cm_per_px: float = Field(gt=0, description="Centimetres per pixel of the uploaded image.")
    scale_source: ScaleSource
    scale_detail: str
    image_width: int
    image_height: int
    origin_px: tuple[float, float] = Field(
        default=(0.0, 0.0),
        description="Where plan (0, 0) lies in the uploaded image, in its pixels. "
        "Image position = origin_px + plan position / cm_per_px.",
    )
    warnings: list[str] = Field(default_factory=list)


class ParseResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    plan: Plan
    meta: ParseMeta
