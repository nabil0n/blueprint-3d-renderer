from pathlib import Path

import numpy as np
import pytest

from blueprint3d.parsing.errors import ParseError
from blueprint3d.parsing.image_io import binarize
from blueprint3d.parsing.learned.learned_parser import TARGET_WALL_PX, LearnedParser
from blueprint3d.parsing.learned.segmentation import CubiCasaModel, Segmentation, model_available
from blueprint3d.parsing.scale import DOOR_WIDTH_CM
from blueprint3d.parsing.walls import extract_wall_mask
from tests.parsing.learned.toy_model import ICON_CLASSES, ROOM_CLASSES, write_toy_model
from tests.parsing.synthetic import DOOR_PX, EXTERIOR_PX, blank, draw_two_room_plan, encode_png
from tests.parsing.test_ocr import no_network  # noqa: F401 - fixture

REPO = Path(__file__).resolve().parents[3]
REAL_MODEL_DIR = REPO / "models"


class ThickStrokesAreWalls:
    """Stands in for the network: thick dark strokes are walls, nothing else is recognised."""

    def __init__(self) -> None:
        self.scales: list[float] = []

    def segment(self, image: np.ndarray, scale: float = 1.0) -> Segmentation:
        self.scales.append(scale)
        walls = extract_wall_mask(binarize(image)).mask > 0
        rooms = np.where(walls, ROOM_CLASSES.index("Wall"), 0).astype(np.uint8)
        return Segmentation(rooms, np.zeros_like(rooms), tuple(ROOM_CLASSES), tuple(ICON_CLASSES))


@pytest.fixture(scope="module")
def result():
    return LearnedParser(model=ThickStrokesAreWalls()).parse(encode_png(draw_two_room_plan()))


def test_names_itself_in_the_result(result):
    assert result.meta.parser == "cubicasa"


def test_finds_the_rooms_and_names_them(result):
    named = sorted((r.name, r.kind) for r in result.plan.rooms)
    assert named == [("Bedroom", "bedroom"), ("Living", "living_room")]


def test_estimates_the_scale_from_its_doors(result):
    assert result.meta.scale_source == "doors"
    assert result.meta.cm_per_px == pytest.approx(DOOR_WIDTH_CM / DOOR_PX, rel=0.1)


def test_marks_the_outer_walls_exterior(result):
    exterior = [w for w in result.plan.walls if w.exterior]
    assert len(exterior) == 4
    assert all(not w.exterior for w in result.plan.walls if w not in exterior)


def test_runs_the_model_at_the_wall_thickness_it_was_trained_on():
    model = ThickStrokesAreWalls()
    LearnedParser(model=model).parse(encode_png(draw_two_room_plan()))
    (scale,) = model.scales
    assert TARGET_WALL_PX / EXTERIOR_PX <= scale <= 1.0


def test_a_user_scale_is_kept():
    parser = LearnedParser(model=ThickStrokesAreWalls())
    result = parser.parse(encode_png(draw_two_room_plan()), cm_per_px=1.0)
    assert (result.meta.scale_source, result.meta.cm_per_px) == ("user", 1.0)


def test_a_page_without_walls_is_a_parse_error():
    with pytest.raises(ParseError):
        LearnedParser(model=ThickStrokesAreWalls()).parse(encode_png(blank()))


def test_loads_the_model_from_its_folder_and_runs_offline(tmp_path, no_network):  # noqa: F811
    write_toy_model(tmp_path)
    result = LearnedParser(model_dir=tmp_path, read_text=None).parse(encode_png(draw_two_room_plan()))
    assert result.plan.walls


@pytest.mark.skipif(not model_available(REAL_MODEL_DIR), reason="exported CubiCasa model not present")
@pytest.mark.parametrize("path", sorted((REPO / "data").glob("*.jpg")), ids=lambda p: p.name)
def test_real_plans_with_the_real_model(path):
    result = LearnedParser(model=CubiCasaModel(REAL_MODEL_DIR)).parse(path.read_bytes())
    assert len(result.plan.walls) >= 8
    assert len(result.plan.rooms) >= 4
    assert any(o.kind == "door" for o in result.plan.openings)
    assert any(o.kind == "window" for o in result.plan.openings)
    assert 0.5 < result.meta.cm_per_px < 5
