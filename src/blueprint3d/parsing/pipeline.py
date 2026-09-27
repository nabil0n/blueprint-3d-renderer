"""Steps every parser shares: finding the plan on the page, the scale bar, room names, warnings."""

from collections.abc import Collection
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from blueprint3d.parsing.geometry import PxPoint
from blueprint3d.parsing.image_io import binarize, line_ink
from blueprint3d.parsing.ocr import TextReader
from blueprint3d.parsing.result import ParseResult
from blueprint3d.parsing.room_names import RoomLabel, label_rooms
from blueprint3d.parsing.scale import ScaleEstimate
from blueprint3d.parsing.scale_bar import ScaleBar, detect_scale_bar
from blueprint3d.parsing.walls import crop_box, expand_to_ink, extract_wall_mask, fill_hatching, ink_box
from blueprint3d.schema import RoomKind


class Parser(Protocol):
    name: str

    def parse(self, data: bytes, cm_per_px: float | None = None) -> ParseResult: ...


@dataclass(frozen=True)
class PlanArea:
    ink: np.ndarray
    """Any ink including light gray lines, 255 = ink."""
    gray: np.ndarray
    """The plan area of the working image, for reading its text."""
    wall_mask: np.ndarray
    min_thickness: int
    box: tuple[int, int, int, int]
    """Plan area (x0, y0, x1, y1) in the working image."""
    page_inks: tuple[np.ndarray, ...]
    """The whole page as dark ink and as any ink (light gray included), 255 = ink."""


def locate_plan(gray: np.ndarray) -> PlanArea:
    """Find the plan with a rough wall mask, then re-derive walls from strokes inside it only:
    page text (bold headings, legends) would otherwise skew the thin/thick split. Without solid
    walls the plan is taken to be hatched: the largest drawing on the page, with its hatching
    closed into solid walls."""
    binary = binarize(gray)
    rough = extract_wall_mask(binary, partitions=False)
    page_ink = line_ink(gray, binary)
    box = crop_box(rough.mask, rough.min_thickness)
    if box is not None:
        x0, y0, x1, y1 = expand_to_ink(box, page_ink, rough.mask)
        plan_binary = binary[y0:y1, x0:x1]
    else:
        x0, y0, x1, y1 = ink_box(binary)
        plan_binary = fill_hatching(binary[y0:y1, x0:x1])
    walls = extract_wall_mask(plan_binary)
    return PlanArea(
        ink=page_ink[y0:y1, x0:x1],
        gray=gray[y0:y1, x0:x1],
        wall_mask=walls.mask,
        min_thickness=walls.min_thickness,
        box=(x0, y0, x1, y1),
        page_inks=(binary, page_ink),
    )


def best_scale_bar(area: PlanArea) -> ScaleBar | None:
    """Scale bars may be drawn in light gray (lost by Otsu) or be scanned and noisy (cleaner with
    Otsu), so search both inks and keep the longest regular run of marks."""
    bars = [b for b in (detect_scale_bar(ink, exclude=area.box) for ink in area.page_inks) if b is not None]
    return max(bars, key=lambda b: b.steps * b.px_per_step, default=None)


def label_rooms_safely(
    reader: TextReader | None, gray: np.ndarray, polygons: tuple[tuple[PxPoint, ...], ...]
) -> tuple[tuple[RoomLabel | None, ...], list[str]]:
    """Room labels read off the plan area, plus a warning if reading failed. A plan without
    names is still a plan, so OCR errors never fail the parse."""
    if reader is None:
        return (None,) * len(polygons), []
    try:
        boxes = reader(gray)
    except Exception as error:  # noqa: BLE001 - any engine failure just means no names
        return (None,) * len(polygons), [f"Room names could not be read ({error}); rooms are unnamed."]
    return label_rooms(polygons, boxes), []


def room_kinds_and_names(
    labels: tuple[RoomLabel | None, ...],
    guessed: dict[int, RoomKind] | None = None,
    balconies: Collection[int] = (),
) -> tuple[dict[int, RoomKind], dict[int, str]]:
    """Kinds and names per room index. A printed label beats a guess (e.g. a model's class), and a
    balcony found by its railings stays one, whatever its label says."""
    named = {i: label for i, label in enumerate(labels) if label is not None}
    kinds = {**(guessed or {}), **{i: label.kind for i, label in named.items()}}
    kinds.update(dict.fromkeys(balconies, "balcony"))
    return kinds, {i: label.name for i, label in named.items()}


def scale_warnings(scale: ScaleEstimate, room_count: int) -> list[str]:
    warnings = []
    if scale.source != "user":
        warnings.append(f"Scale is estimated. {scale.detail} Enter the real scale if sizes look off.")
    if room_count == 0:
        warnings.append("No enclosed rooms found; floors are missing.")
    return warnings
