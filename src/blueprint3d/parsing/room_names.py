"""Naming rooms from the labels printed inside them (SOVRUM, KÖK, BAD ...)."""

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher

import cv2
import numpy as np

from blueprint3d.parsing.geometry import PxPoint
from blueprint3d.parsing.ocr import TextBox
from blueprint3d.schema import RoomKind

VOCABULARY: dict[str, RoomKind] = {
    # Swedish
    "vardagsrum": "living_room", "allrum": "living_room",
    "sovrum": "bedroom", "kammare": "bedroom",
    "kök": "kitchen", "kokvrå": "kitchen",
    "bad": "bathroom", "badrum": "bathroom", "wc": "bathroom", "dusch": "bathroom", "toalett": "bathroom",
    "tvätt": "bathroom",
    "hall": "hallway", "entré": "hallway", "tamb": "hallway", "tambur": "hallway", "kapprum": "hallway",
    "klädkammare": "closet", "klk": "closet", "förråd": "closet",
    "balkong": "balcony", "uteplats": "balcony", "altan": "balcony", "terrass": "balcony",
    "matrum": "other", "arbetsrum": "other", "rum": "other",
    # English
    "living": "living_room", "lounge": "living_room",
    "bedroom": "bedroom",
    "kitchen": "kitchen",
    "bathroom": "bathroom", "bath": "bathroom", "shower": "bathroom",
    "hallway": "hallway", "entrance": "hallway",
    "closet": "closet", "storage": "closet",
    "balcony": "balcony", "terrace": "balcony",
    "dining": "other", "study": "other", "room": "other",
}  # fmt: skip
"""Words that name a room, printed spelling -> kind. "rum"/"room" only complete other words."""

ABBREVIATIONS = {"vardr": "vardagsrum", "vardrum": "vardagsrum", "sovr": "sovrum", "balk": "balkong"}
"""Common plan abbreviations (VARD.R, VARD.RUM, SOVR., BALK.), dots removed -> the word they stand for."""

KIND_PRIORITY: tuple[RoomKind, ...] = (
    "living_room", "kitchen", "bedroom", "bathroom", "hallway", "closet", "balcony", "other"
)  # fmt: skip
"""An open-plan room with several labels takes the first of their kinds in this order."""

MIN_SCORE = 0.5
"""OCR lines less confident than this are ignored."""
MIN_FUZZY_LENGTH = 4
MIN_SIMILARITY = 0.65
"""Words this long may be misread (scans, hand lettering) and still match a vocabulary word this
similar. Shorter ones must match exactly: cupboard markers (G, L, ST, KYL) are short too."""
_LOOKALIKES = str.maketrans("015", "ols")
_TOKEN = re.compile(r"[^\W_]+")
_ABBREVIATION_DOT = re.compile(r"(?<=[^\W\d_])\.(?=[^\W\d_])")
"""A dot between letters, as in VARD.RUM: removed, so the abbreviation reads as one word."""


@dataclass(frozen=True)
class RoomLabel:
    name: str
    kind: RoomKind


def _plain(word: str) -> str:
    """Lowercase, without diacritics, digits inside words read as the letters they resemble."""
    decomposed = unicodedata.normalize("NFKD", word.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c)).translate(_LOOKALIKES)


_PLAIN_VOCABULARY = {_plain(word): word for word in VOCABULARY}


def _closest_word(token: str) -> tuple[str, bool] | None:
    """(vocabulary word, printed exactly as in the vocabulary) or None."""
    if token.lower() in VOCABULARY:
        return token.lower(), True
    plain = _plain(token)
    if plain in ABBREVIATIONS:
        return ABBREVIATIONS[plain], False
    if plain in _PLAIN_VOCABULARY:
        return _PLAIN_VOCABULARY[plain], False
    if len(plain) < MIN_FUZZY_LENGTH:
        return None
    similarity, word = max(
        (SequenceMatcher(None, plain, key).ratio(), word) for key, word in _PLAIN_VOCABULARY.items()
    )
    return (word, False) if similarity >= MIN_SIMILARITY else None


def match_label(text: str) -> RoomLabel | None:
    """A room label if every word of `text` names a room (numbers such as "Sovrum 2" allowed).
    Labels printed as-is keep their spelling; misread ones take the vocabulary's."""
    tokens = _join_stray_letters(_TOKEN.findall(_ABBREVIATION_DOT.sub("", text)))
    words = [t for t in tokens if not t.isdigit()]
    matches = [_closest_word(w) for w in words]
    if not words or any(m is None for m in matches):
        return None
    found = [m for m in matches if m is not None]
    kinds = [VOCABULARY[word] for word, _ in found]
    kind = next((k for k in kinds if k != "other"), "other")
    if all(exact for _, exact in found):
        return RoomLabel(name=text.strip().capitalize(), kind=kind)
    numbers = [t for t in tokens if t.isdigit()]
    return RoomLabel(name=" ".join([*(word for word, _ in found), *numbers]).capitalize(), kind=kind)


def _join_stray_letters(tokens: list[str]) -> list[str]:
    """A single letter belongs to the word before it: OCR reads "SOVR." as "SOV R." at times."""
    joined: list[str] = []
    for token in tokens:
        if len(token) == 1 and token.isalpha() and joined and not joined[-1].isdigit():
            joined[-1] += token
        else:
            joined.append(token)
    return joined


def label_rooms(
    rooms: Iterable[tuple[PxPoint, ...]], boxes: Iterable[TextBox]
) -> tuple[RoomLabel | None, ...]:
    """The label for each room from the room names printed inside it, or None."""
    labelled = [
        (box.center, label)
        for box in boxes
        if box.score >= MIN_SCORE and (label := match_label(box.text)) is not None
    ]
    return tuple(
        _combine([label for center, label in labelled if _inside(polygon, center)]) for polygon in rooms
    )


def _inside(polygon: tuple[PxPoint, ...], point: tuple[float, float]) -> bool:
    return cv2.pointPolygonTest(np.array(polygon, np.float32), point, measureDist=False) >= 0


def _combine(labels: list[RoomLabel]) -> RoomLabel | None:
    """One label for an open-plan room printed with several names, main kind first."""
    if not labels:
        return None
    unique = list({label.name.lower(): label for label in labels}.values())
    ordered = sorted(unique, key=lambda label: KIND_PRIORITY.index(label.kind))
    return RoomLabel(name=" / ".join(label.name for label in ordered), kind=ordered[0].kind)
