import pytest

from blueprint3d.parsing.scale import DOOR_WIDTH_CM, EXTERIOR_WALL_CM, estimate_scale, page_format_scale

NO_EXTRAS = {"scale_bar_px_per_m": None, "page_cm_per_px": None}


def test_user_value_wins():
    est = estimate_scale(
        user_cm_per_px=2.0, door_widths_px=[50, 60], max_wall_thickness_px=20, scale_bar_px_per_m=67, page_cm_per_px=1.5
    )
    assert (est.cm_per_px, est.source) == (2.0, "user")


def test_uses_median_door_width():
    est = estimate_scale(user_cm_per_px=None, door_widths_px=[50, 57, 200], max_wall_thickness_px=20, **NO_EXTRAS)
    assert est.source == "doors"
    assert est.cm_per_px == pytest.approx(DOOR_WIDTH_CM / 57)
    assert "3 door" in est.detail


def test_falls_back_to_wall_thickness():
    est = estimate_scale(user_cm_per_px=None, door_widths_px=[], max_wall_thickness_px=20, **NO_EXTRAS)
    assert est.source == "wall_thickness"
    assert est.cm_per_px == pytest.approx(EXTERIOR_WALL_CM / 20)


def test_scale_bar_beats_page_format_and_doors():
    est = estimate_scale(
        user_cm_per_px=None, door_widths_px=[58], max_wall_thickness_px=20, scale_bar_px_per_m=67.5, page_cm_per_px=1.6
    )
    assert est.source == "scale_bar"
    assert est.cm_per_px == pytest.approx(100 / 67.5)


def test_page_format_beats_doors_when_they_roughly_agree():
    est = estimate_scale(
        user_cm_per_px=None, door_widths_px=[75], max_wall_thickness_px=30, scale_bar_px_per_m=None, page_cm_per_px=1.485
    )
    assert (est.cm_per_px, est.source) == (1.485, "page_format")


@pytest.mark.parametrize("field", ["scale_bar_px_per_m", "page_cm_per_px"])
def test_assumptions_that_contradict_the_doors_are_dropped(field):
    """E.g. a bar whose steps are 5 m, or a 1:50 plan on A4: doors are ~3x off from the assumption."""
    extras = {**NO_EXTRAS, field: 67.5 if field == "scale_bar_px_per_m" else 1.485}
    est = estimate_scale(user_cm_per_px=None, door_widths_px=[160, 170], max_wall_thickness_px=40, **extras)
    assert est.source == "doors"


def test_assumptions_are_used_when_there_are_no_doors():
    est = estimate_scale(
        user_cm_per_px=None, door_widths_px=[], max_wall_thickness_px=20, scale_bar_px_per_m=None, page_cm_per_px=1.485
    )
    assert est.source == "page_format"


@pytest.mark.parametrize(
    ("size", "expected"),
    [((1413, 2000), 2970 / 2000), ((2000, 1414), 2970 / 2000), ((2480, 3508), 2970 / 3508), ((1000, 1000), None)],
)
def test_page_format_assumes_a4_at_1_to_100(size, expected):
    result = page_format_scale(*size)
    assert result == (None if expected is None else pytest.approx(expected))
