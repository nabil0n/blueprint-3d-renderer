from fastapi.testclient import TestClient

from blueprint3d import api
from blueprint3d.api import app
from tests.parsing.synthetic import blank, draw_two_room_plan, encode_png

client = TestClient(app)


def upload(data: bytes, content_type: str = "image/png", **form):
    return client.post("/api/plans/parse", files={"file": ("plan.png", data, content_type)}, data=form)


def test_parse_returns_plan_and_meta():
    response = upload(encode_png(draw_two_room_plan()))
    assert response.status_code == 200
    body = response.json()
    assert len(body["plan"]["rooms"]) == 2
    assert body["meta"]["parser"] == "opencv"
    assert body["meta"]["scale_source"] == "doors"


def test_parse_accepts_a_scale_override():
    response = upload(encode_png(draw_two_room_plan()), cm_per_px="2.5")
    assert response.json()["meta"]["cm_per_px"] == 2.5


def test_parse_rejects_bad_scale():
    assert upload(encode_png(draw_two_room_plan()), cm_per_px="-1").status_code == 422


def test_parse_rejects_unsupported_type():
    response = upload(b"%PDF-1.7", content_type="application/pdf")
    assert response.status_code == 415


def test_parse_rejects_oversized_upload(monkeypatch):
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 100)
    assert upload(encode_png(draw_two_room_plan())).status_code == 413


def test_parse_reports_unparseable_images():
    response = upload(encode_png(blank()))
    assert response.status_code == 422
    assert "detail" in response.json()


def test_parse_reports_undecodable_images():
    assert upload(b"not really a png").status_code == 422


def test_health_reports_ok():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_validate_plan_returns_normalised_plan():
    response = client.post(
        "/api/plans/validate",
        json={"openings": [], "walls": [{"id": "w1", "start": {"x": 0, "y": 0}, "end": {"x": 100, "y": 0}}]},
    )
    assert response.status_code == 200
    assert response.json()["walls"][0]["thickness"] == 15


def test_validate_plan_rejects_invalid_plan():
    response = client.post(
        "/api/plans/validate",
        json={
            "walls": [{"id": "w1", "start": {"x": 0, "y": 0}, "end": {"x": 100, "y": 0}}],
            "openings": [{"id": "o1", "wall_id": "nope", "kind": "door", "offset": 50, "width": 90}],
        },
    )
    assert response.status_code == 422
    assert "unknown wall" in response.text
