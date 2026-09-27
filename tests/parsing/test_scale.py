import pytest

from blueprint3d.parsing.scale import DOOR_WIDTH_CM, EXTERIOR_WALL_CM, estimate_scale


def test_user_value_wins():
    est = estimate_scale(user_cm_per_px=2.0, door_widths_px=[50, 60], max_wall_thickness_px=20)
    assert (est.cm_per_px, est.source) == (2.0, "user")


def test_uses_median_door_width():
    est = estimate_scale(user_cm_per_px=None, door_widths_px=[50, 57, 200], max_wall_thickness_px=20)
    assert est.source == "doors"
    assert est.cm_per_px == pytest.approx(DOOR_WIDTH_CM / 57)
    assert "3 door" in est.detail


def test_falls_back_to_wall_thickness():
    est = estimate_scale(user_cm_per_px=None, door_widths_px=[], max_wall_thickness_px=20)
    assert est.source == "wall_thickness"
    assert est.cm_per_px == pytest.approx(EXTERIOR_WALL_CM / 20)
