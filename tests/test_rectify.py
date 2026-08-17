import numpy as np
import cv2
from conftest import draw_table, warp_page, WARP_DST
from segmentation import (find_table_quad, rectify_page, ensure_upright,
                          CANON_W, CANON_H)


def test_find_table_quad_on_warped_page():
    img, _ = draw_table()
    warped = warp_page(img)
    quad = find_table_quad(warped)
    assert quad is not None and quad.shape == (4, 2)
    for found, expected in zip(quad, WARP_DST):
        assert np.linalg.norm(found - expected) < 12   # px tolerance


def test_find_table_quad_none_on_blank_page():
    blank = np.full((850, 1200), 235, np.uint8)
    assert find_table_quad(blank) is None


def test_rectify_maps_table_to_canonical_size():
    img, _ = draw_table()
    warped = warp_page(img)
    quad = find_table_quad(warped)
    rect = rectify_page(warped, quad)
    assert rect.shape[:2] == (CANON_H, CANON_W)
    # border line should now hug the edges: dark pixels near x=0 and x=W-1
    assert rect[:, :6].min() < 100 and rect[:, -6:].min() < 100


def test_ensure_upright_flips_header_to_top():
    img, _ = draw_table()
    quad = find_table_quad(img)
    rect = rectify_page(img, quad)
    up, flipped = ensure_upright(rect)
    assert not flipped
    upside_down = cv2.rotate(rect, cv2.ROTATE_180)
    up2, flipped2 = ensure_upright(upside_down)
    assert flipped2
    assert np.array_equal(up2, rect)
