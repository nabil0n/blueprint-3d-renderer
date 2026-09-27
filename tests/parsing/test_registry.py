from blueprint3d.parsing.learned.learned_parser import LearnedParser
from blueprint3d.parsing.opencv_parser import OpenCvParser
from blueprint3d.parsing.registry import DEFAULT_PARSER, available_parsers, parser_catalogue
from tests.parsing.learned.toy_model import write_toy_model


def test_opencv_is_always_available(tmp_path):
    parsers = available_parsers(tmp_path)
    assert list(parsers) == ["opencv"]
    assert isinstance(parsers["opencv"], OpenCvParser)
    assert DEFAULT_PARSER == "opencv"


def test_the_learned_parser_is_available_once_its_model_is_exported(tmp_path):
    write_toy_model(tmp_path)
    parsers = available_parsers(tmp_path)
    assert list(parsers) == ["opencv", "cubicasa"]
    assert isinstance(parsers["cubicasa"], LearnedParser)


def test_catalogue_lists_every_parser_with_its_availability(tmp_path):
    catalogue = parser_catalogue(tmp_path)
    assert [(p.name, p.available) for p in catalogue] == [("opencv", True), ("cubicasa", False)]
    assert "setup_cubicasa" in catalogue[1].description
