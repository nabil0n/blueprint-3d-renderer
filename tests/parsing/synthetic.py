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


def draw_bay_and_balcony_plan() -> np.ndarray:
    """Two rooms; the north-west corner is cut off by an angled glazed wall (thin gray lines between
    two thick corner blocks, like a bay window), and a balcony with a thin railing sits south of
    the west room, reached through a door in the south wall."""
    img = blank(1000, 900)
    cv2.rectangle(img, (250, 100), (899, 119), BLACK, -1)  # north
    cv2.rectangle(img, (880, 100), (899, 699), BLACK, -1)  # east
    cv2.rectangle(img, (100, 680), (899, 699), BLACK, -1)  # south
    cv2.rectangle(img, (100, 250), (119, 699), BLACK, -1)  # west
    cv2.rectangle(img, (494, 120), (505, 679), BLACK, -1)  # interior
    cv2.rectangle(img, (494, 380), (505, 380 + DOOR_PX - 1), WHITE, -1)

    # Angled glazed wall: corner blocks plus three parallel light-gray lines between them.
    cv2.rectangle(img, (100, 222), (127, 249), BLACK, -1)
    cv2.rectangle(img, (222, 100), (249, 127), BLACK, -1)
    for offset in (-9, 0, 9):
        shift = round(offset / np.sqrt(2))
        cv2.line(img, (122 + shift, 229 + shift), (229 + shift, 122 + shift), (170, 170, 170), 1)

    # Door from the west room to the balcony, and the balcony railing (a single thin line).
    cv2.rectangle(img, (400, 680), (400 + DOOR_PX - 1, 699), WHITE, -1)
    cv2.rectangle(img, (300, 699), (600, 830), BLACK, 1)

    cv2.putText(img, "Living", (250, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLACK, 1)
    cv2.putText(img, "Balcony", (400, 780), cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLACK, 1)
    return img


def draw_decorated_plan() -> np.ndarray:
    """The two-room plan dressed like a modern rental export: a thick dark-green logo just above it,
    the east room filled light blue (like a bathroom), and a balcony outlined in light gray."""
    img = draw_two_room_plan()
    img = np.vstack([blank(1000, 120), img, blank(1000, 150)])  # room for logo and balcony
    cv2.putText(img, "LOGO", (120, 170), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (70, 90, 20), 16)
    cv2.rectangle(img, (506, 240), (879, 799), (240, 225, 200), -1)  # BGR light blue fill
    cv2.rectangle(img, (300, 820), (600, 940), (160, 160, 160), 2)
    cv2.rectangle(img, (400, 800), (400 + DOOR_PX - 1, 819), WHITE, -1)  # balcony door in the south wall
    return img


def draw_tick_scale_bar(
    img: np.ndarray, x0: int, y: int, px_per_m: float, metres: int = 5, ink: int = 0
) -> None:
    """Line with a tick per metre, finer ticks in the first metre and digit labels above
    (the style in drheymansgata5 / memoargatan)."""
    color = (ink, ink, ink)
    cv2.line(img, (x0, y), (round(x0 + metres * px_per_m), y), color, 1)
    for m in range(metres + 1):
        x = round(x0 + m * px_per_m)
        cv2.line(img, (x, y), (x, y - 14), color, 1)
        cv2.putText(img, str(m), (x - 6, y - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 1)
    for k in range(1, 10):
        x = round(x0 + k * px_per_m / 10)
        cv2.line(img, (x, y), (x, y - 5), color, 1)


def draw_checker_scale_bar(img: np.ndarray, x0: int, y: int, px_per_m: float, metres: int = 10) -> None:
    """A filled band with alternating white inserts per metre (the style in hallandsgatan5)."""
    x1 = round(x0 + metres * px_per_m)
    cv2.rectangle(img, (x0, y), (x1, y + 13), BLACK, -1)
    for m in range(metres):
        a, b = round(x0 + m * px_per_m) + 2, round(x0 + (m + 1) * px_per_m) - 2
        top = y + 2 if m % 2 == 0 else y + 7
        cv2.rectangle(img, (a, top), (b, top + 4), WHITE, -1)


def encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def to_binary(img: np.ndarray) -> np.ndarray:
    """Dark pixels as 255, like the parser's binarisation."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return np.where(gray < 128, 255, 0).astype(np.uint8)
