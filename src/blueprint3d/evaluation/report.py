"""Evaluating a directory of sample plans and writing the report."""

import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from blueprint3d.evaluation.metrics import SampleScore, failed, score
from blueprint3d.evaluation.overlay import contact_sheet, draw_overlay, with_caption
from blueprint3d.evaluation.truth import load_truth
from blueprint3d.parsing.errors import ParseError
from blueprint3d.parsing.opencv_parser import OpenCvParser

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})


def _read_image(path: Path) -> np.ndarray | None:
    # np.fromfile + imdecode instead of imread: imread cannot open non-ASCII paths on Windows.
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def _write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"Could not encode {path.name}")
    path.write_bytes(encoded.tobytes())


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{value:+.1%}"


def _caption(s: SampleScore) -> list[str]:
    verdict = {True: "PASS", False: "FAIL", None: "?"}[s.passed]
    if s.error:
        return [f"{verdict} {s.name}", s.error[:70]]
    expected = "?" if s.rooms_expected is None else s.rooms_expected
    return [
        f"{verdict} {s.name}",
        f"scale {_percent(s.scale_error)} ({s.scale_source})  area {_percent(s.area_error)}",
        f"rooms {s.rooms_found}/{expected}  balconies {s.balconies_found}/{s.balconies_expected}"
        f"  names {_names(s)}  kinds {_kinds(s)}",
    ]


def _names(s: SampleScore) -> str:
    return "-" if s.names_expected is None else f"{s.names_found}/{s.names_expected}"


def _kinds(s: SampleScore) -> str:
    return "-" if s.kinds_expected is None else f"{s.kinds_found}/{s.kinds_expected}"


def _verdict(s: SampleScore) -> str:
    return {True: "PASS", False: "FAIL", None: "?"}[s.passed]


def evaluate_directory(data_dir: Path, truth_path: Path, out_dir: Path, parser=None) -> list[SampleScore]:
    """Parse every image in `data_dir`, score it, and write overlays, `overview.png` and `report.json`."""
    parser = parser or OpenCvParser()
    truth = load_truth(truth_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    scores, tiles = [], []
    for path in sorted(p for p in data_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
        image = _read_image(path)
        try:
            result = parser.parse(path.read_bytes())
        except ParseError as error:
            sample = failed(path.name, str(error), truth.get(path.name))
            overlay = image
        else:
            sample = score(path.name, result, truth.get(path.name))
            overlay = draw_overlay(image, result)
        scores.append(sample)
        _write_image(out_dir / f"{path.stem}.overlay.png", with_caption(overlay, _caption(sample)))
        tiles.append((overlay, _caption(sample)))

    if tiles:
        _write_image(out_dir / "overview.png", contact_sheet(tiles))
    report = [asdict(s) | {"passed": s.passed} for s in scores]
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return scores


def format_table(scores: list[SampleScore]) -> str:
    header = (
        f"{'sample':<28} {'result':<6} {'scale':>8} {'source':<15} {'area':>8} {'rooms':>7} {'balc':>6} "
        f"{'names':>6} {'kinds':>6}"
    )
    rows = [header, "-" * len(header)]
    for s in scores:
        verdict = {True: "PASS", False: "FAIL", None: "?"}[s.passed]
        if s.error:
            rows.append(f"{s.name:<28} {verdict:<6} error: {s.error}")
            continue
        rooms = f"{s.rooms_found}/{'?' if s.rooms_expected is None else s.rooms_expected}"
        balconies = f"{s.balconies_found}/{'?' if s.balconies_expected is None else s.balconies_expected}"
        rows.append(
            f"{s.name:<28} {verdict:<6} {_percent(s.scale_error):>8} {s.scale_source or '-':<15} "
            f"{_percent(s.area_error):>8} {rooms:>7} {balconies:>6} {_names(s):>6} {_kinds(s):>6}"
        )
    judged = [s for s in scores if s.passed is not None]
    rows.append(f"\n{sum(bool(s.passed) for s in judged)}/{len(judged)} passed")
    return "\n".join(rows)


def format_comparison(results: dict[str, list[SampleScore]]) -> str:
    """One row per sample, one column group per parser: verdict, scale error, area error, rooms."""
    parsers = list(results)
    by_sample = {name: {s.name: s for s in scores} for name, scores in results.items()}
    samples = sorted({s.name for scores in results.values() for s in scores})
    header = f"{'sample':<28}" + "".join(f" | {p:<33}" for p in parsers)
    rows = [header, "-" * len(header)]
    for sample in samples:
        cells = []
        for parser in parsers:
            s = by_sample[parser].get(sample)
            if s is None or s.error:
                cells.append(f" | {'FAIL  error':<33}")
                continue
            rooms = f"{s.rooms_found}/{'?' if s.rooms_expected is None else s.rooms_expected}"
            scale, area = _percent(s.scale_error), _percent(s.area_error)
            cell = f"{_verdict(s):<5} {scale:>7} {area:>7} {rooms:>6} {_kinds(s):>5}"
            cells.append(f" | {cell:<33}")
        rows.append(f"{sample:<28}" + "".join(cells))
    rows.append("\ncolumns: result, scale error, area error, rooms, kinds")
    for parser, scores in results.items():
        judged = [s for s in scores if s.passed is not None]
        rows.append(f"{parser}: {sum(bool(s.passed) for s in judged)}/{len(judged)} passed")
    return "\n".join(rows)
