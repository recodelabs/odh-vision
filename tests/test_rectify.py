import numpy as np
import cv2
from conftest import draw_table, warp_page, WARP_DST
from segmentation import (find_table_quad, rectify_page, ensure_upright,
                          CANON_W, CANON_H, HEADER_FRAC)


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
    # margins=True draws the pre-rectification page-margin content
    # (dense legend line above the table, sparser one below) that the
    # production margin-ink signal actually keys on -- see
    # segmentation._margin_ink. A synthetic page without this content
    # can't exercise that signal at all.
    img, _ = draw_table(margins=True)
    quad = find_table_quad(img)
    rect = rectify_page(img, quad)
    up, flipped = ensure_upright(rect, orig_gray=img, quad=quad)
    assert not flipped
    assert np.array_equal(up, rect)

    # Simulate a page photographed upside-down: rotate the *raw* page
    # (margins included) before quad-finding/rectification, matching how
    # segment_page actually feeds ensure_upright. This independently
    # re-runs quad detection on a distinct (rotated) image, so its
    # contour approximation can land a pixel or two off the exact mirror
    # of `quad` -- not bit-identical to `rect` after flipping back, but
    # the dense header ticks (drawn only in the header band) should end
    # up concentrated near the top, same as the known-upright `rect`.
    upside_down_raw = cv2.rotate(img, cv2.ROTATE_180)
    quad2 = find_table_quad(upside_down_raw)
    rect2 = rectify_page(upside_down_raw, quad2)
    up2, flipped2 = ensure_upright(rect2, orig_gray=upside_down_raw, quad=quad2)
    assert flipped2

    def band_density(rect_img, top):
        band = int(rect_img.shape[0] * HEADER_FRAC)
        region = rect_img[:band] if top else rect_img[-band:]
        return (region < 128).sum() / region.size

    # dense header ticks concentrated in the top band should out-density
    # an equivalently-sized band at the bottom (evenly-spaced body grid
    # lines only), in both the known-upright rect and the corrected
    # (flipped-back) up2.
    assert band_density(rect, top=True) > band_density(rect, top=False)
    assert band_density(up2, top=True) > band_density(up2, top=False)
