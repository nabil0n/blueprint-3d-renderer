"""Joining collinear wall pieces and classifying the gaps between them as doors or windows."""

from dataclasses import replace
from typing import Literal

import numpy as np

from blueprint3d.parsing.geometry import AxisSegment, Gap, WallRun

MAX_THICKNESS_RATIO = 2.0
BAND_PAD_PX = 2
EDGE_TRIM_PX = 2
LINE_COVERAGE = 0.9
"""A row counts as a drawn line if it is dark along this fraction of the gap.
Solid window lines reach ~1.0; dashed passage markings stay well below."""
MIN_WINDOW_LINES = 2


def merge_collinear(segments: list[AxisSegment], *, min_gap: float, max_gap: float) -> list[WallRun]:
    """Merge pieces lying on the same line. Gaps shorter than `min_gap` are closed (drawing noise);
    gaps between `min_gap` and `max_gap` become openings; longer gaps separate walls."""
    runs: list[WallRun] = []
    for segment in sorted(segments, key=lambda s: (s.axis, s.start)):
        index = _find_run(runs, segment, max_gap)
        if index is None:
            runs.append(_new_run(segment))
        else:
            runs[index] = _extend(runs[index], segment, min_gap)
    return runs


def _new_run(s: AxisSegment) -> WallRun:
    return WallRun(axis=s.axis, center=s.center, thickness=s.thickness, start=s.start, end=s.end)


def _find_run(runs: list[WallRun], s: AxisSegment, max_gap: float) -> int | None:
    candidates = [
        (abs(run.center - s.center), i)
        for i, run in enumerate(runs)
        if run.axis == s.axis
        and abs(run.center - s.center) <= max(run.thickness, s.thickness) / 2
        and max(run.thickness, s.thickness) <= MAX_THICKNESS_RATIO * min(run.thickness, s.thickness)
        and -s.length / 2 <= s.start - run.end <= max_gap
    ]
    return min(candidates)[1] if candidates else None


def _extend(run: WallRun, s: AxisSegment, min_gap: float) -> WallRun:
    run_length, total = run.end - run.start, run.end - run.start + s.length
    gap = s.start - run.end
    return replace(
        run,
        end=max(run.end, s.end),
        center=(run.center * run_length + s.center * s.length) / total,
        thickness=(run.thickness * run_length + s.thickness * s.length) / total,
        gaps=(*run.gaps, (run.end, s.start)) if gap >= min_gap else run.gaps,
    )


def classify_gap(binary: np.ndarray, run: WallRun, gap: Gap) -> Literal["door", "window"]:
    """Windows are drawn as thin lines running along the wall inside the gap; doors leave it empty
    (their leaf and swing arc cross the wall band or sit beside it)."""
    band = _gap_band(binary if run.axis == "h" else binary.T, run, gap)
    if band.size == 0:
        return "door"
    is_line = (band > 0).mean(axis=1) >= LINE_COVERAGE
    # Count separate lines: rising edges in the per-row flags.
    lines = int(is_line[0]) + int(np.count_nonzero(is_line[1:] & ~is_line[:-1]))
    return "window" if lines >= MIN_WINDOW_LINES else "door"


def _gap_band(horizontal: np.ndarray, run: WallRun, gap: Gap) -> np.ndarray:
    half = run.thickness / 2 + BAND_PAD_PX
    r0, r1 = max(0, round(run.center - half)), round(run.center + half) + 1
    c0, c1 = round(gap[0]) + EDGE_TRIM_PX, round(gap[1]) - EDGE_TRIM_PX
    return horizontal[r0:r1, c0:c1]
