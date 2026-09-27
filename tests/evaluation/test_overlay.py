import numpy as np

from blueprint3d.evaluation.overlay import contact_sheet


def test_wide_tiles_are_scaled_down_to_fit_their_cell():
    tall = np.zeros((1000, 700, 3), np.uint8)
    wide = np.zeros((100, 5000, 3), np.uint8)
    sheet = contact_sheet([(tall, ["tall"]), (wide, ["wide"])], columns=2, tile_height=500)
    assert sheet.shape[1] <= 2 * 500 * 1.6
