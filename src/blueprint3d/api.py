"""HTTP API."""

from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from blueprint3d.parsing.errors import ParseError
from blueprint3d.parsing.opencv_parser import OpenCvParser
from blueprint3d.parsing.result import ParseResult
from blueprint3d.schema import Plan

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_TYPES = frozenset({"image/png", "image/jpeg", "image/webp"})

app = FastAPI(title="Blueprint 3D", version="0.1.0")
parser = OpenCvParser()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/plans/validate")
def validate_plan(plan: Plan) -> Plan:
    """Validate a plan and return it with defaults filled in. FastAPI returns 422 on failure."""
    return plan


@app.post("/api/plans/parse")
def parse_plan(
    file: Annotated[UploadFile, File(description="Floor plan image (PNG, JPEG or WebP).")],
    cm_per_px: Annotated[float | None, Form(gt=0, le=100, description="Known scale; estimated if omitted.")] = None,
) -> ParseResult:
    """Turn a floor plan image into a plan. Sync on purpose: parsing is CPU-bound and runs in the threadpool."""
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(415, f"Unsupported file type {file.content_type!r}. Upload a PNG, JPEG or WebP image.")
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Image is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    try:
        return parser.parse(data, cm_per_px=cm_per_px)
    except ParseError as error:
        raise HTTPException(422, str(error)) from error
