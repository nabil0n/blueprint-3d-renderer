"""Synthetic floor plans drawn with OpenCV, with known ground truth.

The two-room plan mimics a rental-site drawing: thick black walls, a window drawn as thin lines
inside a wall gap, doors drawn as gaps with a thin leaf and swing arc, room labels, and page
decoration (a bold title and a filled legend icon) that must not be mistaken for walls.
"""

import cv2
import numpy as np

from blueprint3d.parsing.scale import DOOR_WIDTH_CM

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

DOOR_PX = 57
TRUE_CM_PER_PX = DOOR_WIDTH_CM / DOOR_PX
EXTERIOR_PX = 20
INTERIOR_PX = 12


def blank(width: int = 1000, height: int = 800) -> np.ndarray:
    return np.full((height, width, 3), 255, np.uint8)


def draw_two_room_plan(window_ink: int = 0, interior_px: int = INTERIOR_PX) -> np.ndarray:
    """`window_ink` is the gray level of the window lines; rental plans often draw them light gray."""
    img = blank()
    # Exterior walls: outer box x 100..899, y 100..699, 20 px thick.
    cv2.rectangle(img, (100, 100), (899, 119), BLACK, -1)
    cv2.rectangle(img, (100, 680), (899, 699), BLACK, -1)
    cv2.rectangle(img, (100, 100), (119, 699), BLACK, -1)
    cv2.rectangle(img, (880, 100), (899, 699), BLACK, -1)
    # Interior wall, INTERIOR_PX thick by default.
    left = 500 - interior_px // 2
    right = left + interior_px - 1
    cv2.rectangle(img, (left, 120), (right, 679), BLACK, -1)

    # Door in the interior wall (gap y 380..436) with leaf and swing arc.
    cv2.rectangle(img, (left, 380), (right, 380 + DOOR_PX - 1), WHITE, -1)
    cv2.line(img, (506, 380), (506 + DOOR_PX, 380), BLACK, 1)
    cv2.ellipse(img, (506, 380), (DOOR_PX, DOOR_PX), 0, 0, 90, BLACK, 1)

    # Entrance door in the south wall (gap x 250..306).
    cv2.rectangle(img, (250, 680), (250 + DOOR_PX - 1, 699), WHITE, -1)
    cv2.line(img, (250, 679), (250, 679 - DOOR_PX), BLACK, 1)
    cv2.ellipse(img, (250, 679), (DOOR_PX, DOOR_PX), 0, -90, 0, BLACK, 1)

    # Window in the north wall (gap x 600..749): frame lines on both faces plus glazing.
    cv2.rectangle(img, (600, 100), (749, 119), WHITE, -1)
    for y in (100, 109, 119):
        cv2.line(img, (600, y), (749, y), (window_ink,) * 3, 1)

    cv2.putText(img, "Living", (250, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLACK, 1)
    cv2.putText(img, "Bedroom", (620, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLACK, 1)

    # Page decoration outside the plan.
    cv2.putText(img, "Sample plan", (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, BLACK, 3)
    cv2.rectangle(img, (960, 760), (989, 789), BLACK, -1)
    return img


def encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def to_binary(img: np.ndarray) -> np.ndarray:
    """Dark pixels as 255, like the parser's binarisation."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return np.where(gray < 128, 255, 0).astype(np.uint8)
