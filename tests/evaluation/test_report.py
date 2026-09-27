import json

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from blueprint3d.evaluation.report import evaluate_directory, format_table
from blueprint3d.evaluation.truth import load_truth
from tests.parsing.learned.toy_model import write_toy_model
from tests.parsing.synthetic import TRUE_CM_PER_PX, blank, draw_two_room_plan, encode_png


@pytest.fixture
def samples(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "två rum.png").write_bytes(encode_png(draw_two_room_plan()))  # non-ASCII name on purpose
    (data / "blank.png").write_bytes(encode_png(blank()))
    truth = {"två rum.png": {"cm_per_px": TRUE_CM_PER_PX, "rooms": 2}, "blank.png": {"cm_per_px": 1.0, "rooms": 1}}
    (data / "truth.json").write_text(json.dumps(truth), encoding="utf-8")
    return data


def test_evaluates_every_image_and_writes_overlays(samples, tmp_path):
    out = tmp_path / "out"
    scores = evaluate_directory(samples, samples / "truth.json", out)

    by_name = {s.name: s for s in scores}
    assert set(by_name) == {"två rum.png", "blank.png"}
    assert by_name["två rum.png"].passed
    assert by_name["blank.png"].error is not None

    assert (out / "två rum.overlay.png").is_file()
    assert (out / "overview.png").is_file()
    overview = cv2.imdecode(np.fromfile(out / "overview.png", np.uint8), cv2.IMREAD_COLOR)
    assert overview is not None and overview.shape[1] > 0
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert {r["name"] for r in report} == {"två rum.png", "blank.png"}


def test_table_lists_each_sample_and_a_summary(samples, tmp_path):
    table = format_table(evaluate_directory(samples, samples / "truth.json", tmp_path / "out"))
    assert "två rum.png" in table
    assert "FAIL" in table and "PASS" in table
    assert "1/2 passed" in table


def test_missing_truth_file_means_no_judgement(samples, tmp_path):
    scores = evaluate_directory(samples, samples / "missing.json", tmp_path / "out")
    assert all(s.rooms_expected is None for s in scores)


def test_truth_file_is_validated(tmp_path):
    path = tmp_path / "truth.json"
    path.write_text(json.dumps({"a.png": {"cm_per_px": -1, "rooms": 2}}))
    with pytest.raises(ValidationError):
        load_truth(path)


def test_cli_exits_non_zero_when_a_sample_fails(samples, tmp_path, capsys):
    from blueprint3d.evaluation.__main__ import main

    assert main([str(samples), "--out", str(tmp_path / "out")]) == 1
    assert "1/2 passed" in capsys.readouterr().out
    assert (tmp_path / "out" / "opencv" / "overview.png").is_file(), "each parser writes to its own folder"


def test_cli_compares_parsers_side_by_side(samples, tmp_path, capsys):
    from blueprint3d.evaluation.__main__ import main

    models = tmp_path / "models"
    models.mkdir()
    write_toy_model(models)
    main([str(samples), "--out", str(tmp_path / "out"), "--parser", "all", "--model-dir", str(models)])
    output = capsys.readouterr().out
    assert (tmp_path / "out" / "opencv" / "report.json").is_file()
    assert (tmp_path / "out" / "cubicasa" / "report.json").is_file()
    assert "opencv" in output and "cubicasa" in output
    assert "två rum.png" in output


def test_cli_refuses_a_parser_whose_model_is_missing(samples, tmp_path):
    from blueprint3d.evaluation.__main__ import main

    with pytest.raises(SystemExit):
        main([str(samples), "--parser", "cubicasa", "--model-dir", str(tmp_path / "no-model")])
