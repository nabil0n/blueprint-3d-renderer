import json

import cv2
import numpy as np
import pytest

from blueprint3d.parsing.learned.segmentation import (
    MANIFEST_FILE,
    MODEL_FILE,
    CubiCasaModel,
    ModelFileError,
    model_available,
)
from tests.parsing.learned.toy_model import ROOM_CLASSES, write_toy_model
from tests.parsing.test_ocr import no_network  # noqa: F401 - fixture


@pytest.fixture
def model_dir(tmp_path):
    write_toy_model(tmp_path)
    return tmp_path


def test_dark_strokes_become_walls_at_the_input_size(model_dir):
    gray = np.full((70, 90), 255, np.uint8)
    gray[20:30, 10:80] = 0
    seg = CubiCasaModel(model_dir).segment(gray)
    assert seg.rooms.shape == seg.icons.shape == (70, 90)
    assert seg.room_mask("Wall")[25, 40] and not seg.room_mask("Wall")[50, 40]
    assert seg.room_mask("Background")[50, 40]
    assert not seg.icon_mask("Door").any()


def test_scaling_the_input_keeps_the_output_on_the_original_grid(model_dir):
    gray = np.full((130, 150), 255, np.uint8)
    gray[40:60, 20:130] = 0
    seg = CubiCasaModel(model_dir).segment(gray, scale=0.5)
    assert seg.rooms.shape == (130, 150)
    assert seg.room_mask("Wall")[50, 75]
    assert not seg.room_mask("Wall")[100, 75]


def test_class_names_come_from_the_manifest(model_dir):
    seg = CubiCasaModel(model_dir).segment(np.full((64, 64), 255, np.uint8))
    assert seg.room_classes == tuple(ROOM_CLASSES)
    with pytest.raises(KeyError):
        seg.room_mask("Swimming pool")


def test_runs_without_network_access(model_dir, no_network):  # noqa: F811
    CubiCasaModel(model_dir).segment(np.full((64, 64), 255, np.uint8))


def test_missing_model_is_a_local_error(tmp_path):
    assert not model_available(tmp_path)
    with pytest.raises(ModelFileError, match="setup_cubicasa"):
        CubiCasaModel(tmp_path)


def test_a_changed_model_file_is_refused(model_dir):
    manifest = json.loads((model_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
    (model_dir / MANIFEST_FILE).write_text(json.dumps({**manifest, "sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(ModelFileError, match="SHA-256"):
        CubiCasaModel(model_dir)


def test_model_available_needs_both_files(model_dir):
    assert model_available(model_dir)
    (model_dir / MODEL_FILE).unlink()
    assert not model_available(model_dir)


def test_colour_input_is_accepted(model_dir):
    image = np.full((64, 64, 3), 255, np.uint8)
    cv2.rectangle(image, (10, 10), (50, 20), (0, 0, 0), -1)
    assert CubiCasaModel(model_dir).segment(image).room_mask("Wall")[15, 30]
