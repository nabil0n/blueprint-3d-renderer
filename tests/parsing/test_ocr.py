"""OCR runs on this machine only: bundled models, no downloads, no network."""

import socket

import cv2
import numpy as np
import pytest

from blueprint3d.parsing import ocr


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError(f"network access attempted: {args}")

    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


@pytest.fixture
def fresh_engine():
    ocr._engine.cache_clear()
    yield
    ocr._engine.cache_clear()


def test_the_models_ship_with_the_installed_package():
    for path in ocr.model_paths().values():
        assert path.is_file(), path


def test_reads_text_without_network_access(no_network, fresh_engine):
    img = np.full((120, 400), 255, np.uint8)
    cv2.putText(img, "Sovrum", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 3)
    assert [box.text for box in ocr.read_text(img)] == ["Sovrum"]


def test_a_missing_model_fails_locally_instead_of_downloading(no_network, fresh_engine, monkeypatch, tmp_path):
    monkeypatch.setattr(ocr, "model_paths", lambda: {key: tmp_path / "missing.onnx" for key in ocr.MODEL_FILES})
    with pytest.raises(FileNotFoundError):
        ocr.read_text(np.full((50, 50), 255, np.uint8))
