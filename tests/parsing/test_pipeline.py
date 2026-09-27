from blueprint3d.parsing.pipeline import room_kinds_and_names
from blueprint3d.parsing.room_names import RoomLabel


def test_printed_labels_name_rooms_and_set_their_kind():
    kinds, names = room_kinds_and_names((RoomLabel("Kök", "kitchen"), None))
    assert kinds == {0: "kitchen"}
    assert names == {0: "Kök"}


def test_a_printed_label_beats_a_guessed_kind():
    labels = (RoomLabel("Sovrum", "bedroom"), None)
    kinds, _ = room_kinds_and_names(labels, guessed={0: "kitchen", 1: "bathroom"})
    assert kinds == {0: "bedroom", 1: "bathroom"}


def test_a_balcony_by_its_railings_stays_a_balcony():
    kinds, names = room_kinds_and_names((RoomLabel("Vardagsrum", "living_room"),), balconies={0})
    assert kinds == {0: "balcony"}
    assert names == {0: "Vardagsrum"}
