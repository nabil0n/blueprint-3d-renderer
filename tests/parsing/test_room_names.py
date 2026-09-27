import pytest

from blueprint3d.parsing.ocr import TextBox
from blueprint3d.parsing.room_names import RoomLabel, label_rooms, match_label

SQUARE = ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0))
RIGHT_SQUARE = ((100.0, 0.0), (200.0, 0.0), (200.0, 100.0), (100.0, 100.0))


def text(value: str, x: float = 50, y: float = 50, score: float = 1.0) -> TextBox:
    return TextBox(text=value, score=score, center=(x, y))


@pytest.mark.parametrize(
    ("printed", "name", "kind"),
    [
        ("Sovrum", "Sovrum", "bedroom"),
        ("SOVRUM", "Sovrum", "bedroom"),
        ("Sovrum 2", "Sovrum 2", "bedroom"),
        ("kammare", "Kammare", "bedroom"),
        ("Vardagsrum", "Vardagsrum", "living_room"),
        ("KÖK", "Kök", "kitchen"),
        ("BAD", "Bad", "bathroom"),
        ("Wc/dusch", "Wc/dusch", "bathroom"),
        ("Hall", "Hall", "hallway"),
        ("ENTRÉ", "Entré", "hallway"),
        ("Balkong", "Balkong", "balcony"),
        ("Bedroom", "Bedroom", "bedroom"),
        ("matrum", "Matrum", "other"),
    ],
)
def test_reads_printed_room_names(printed, name, kind):
    assert match_label(printed) == RoomLabel(name=name, kind=kind)


@pytest.mark.parametrize(
    ("misread", "name", "kind"),
    [
        ("S0VFum", "Sovrum", "bedroom"),  # scanned: 0 for o, F for r
        ("Famb.", "Tamb", "hallway"),  # "tamb." (tambur) with a broken t
        ("kők", "Kök", "kitchen"),
        ("SOVCUH", "Sovrum", "bedroom"),  # stylised hand lettering
        ("VACDAGSEUH", "Vardagsrum", "living_room"),
    ],
)
def test_misread_names_fall_back_to_the_closest_word(misread, name, kind):
    assert match_label(misread) == RoomLabel(name=name, kind=kind)


@pytest.mark.parametrize("marker", ["G", "g", "L", "S", "ST", "K", "F", "DM", "HS", "KYL", "(DM)", "6.", "网"])
def test_cupboard_and_appliance_markers_are_not_room_names(marker):
    assert match_label(marker) is None


@pytest.mark.parametrize(
    "other", ["Obj.nr 5403-0474", "HALLANDSGATAN 5", "Heymansgatan", "SKALA 1:100", "76,9 m2"]
)
def test_other_page_text_is_not_a_room_name(other):
    assert match_label(other) is None


def test_labels_go_to_the_room_they_are_printed_in():
    boxes = [text("Kök", 50, 50), text("Sovrum", 150, 50), text("Hall", 500, 500)]
    labels = label_rooms((SQUARE, RIGHT_SQUARE), boxes)
    assert labels == (RoomLabel("Kök", "kitchen"), RoomLabel("Sovrum", "bedroom"))


def test_room_without_a_label_stays_unnamed():
    assert label_rooms((SQUARE, RIGHT_SQUARE), [text("G", 150, 50)]) == (None, None)


def test_open_plan_room_lists_every_name_and_takes_the_main_kind():
    boxes = [text("KÖK", 10, 50), text("ENTRÉ", 50, 50), text("VARDAGSRUM", 90, 50)]
    (label,) = label_rooms((SQUARE,), boxes)
    assert label == RoomLabel("Vardagsrum / Kök / Entré", "living_room")


def test_the_same_name_printed_twice_is_listed_once():
    (label,) = label_rooms((SQUARE,), [text("Hall", 20, 20), text("HALL", 80, 80)])
    assert label == RoomLabel("Hall", "hallway")


def test_low_confidence_text_is_ignored():
    assert label_rooms((SQUARE,), [text("Sovrum", score=0.3)]) == (None,)
