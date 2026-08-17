import numpy as np
import cv2

from segmentation import HEADER_FRAC   # keep synthetic fixture in sync with
                                        # the production constant it exercises

PAGE_W, PAGE_H = 1200, 850            # synthetic "photo" size
TABLE = (60, 40, 1140, 810)           # left, top, right, bottom of drawn table
N_RECORDS = 5

def draw_table(n_records=N_RECORDS):
    """Grey page with a black grid: border, dense header band, 15 body rows,
    6 column lines. Returns (image, truth) where truth holds the known
    geometry for assertions."""
    img = np.full((PAGE_H, PAGE_W), 235, np.uint8)
    l, t, r, b = TABLE
    cv2.rectangle(img, (l, t), (r, b), 0, 3)
    header_bottom = t + int((b - t) * HEADER_FRAC)
    cv2.line(img, (l, header_bottom), (r, header_bottom), 0, 2)
    for x in range(l + 6, r - 6, 10):              # dense printed "text"
        cv2.line(img, (x, t + 6), (x, header_bottom - 6), 0, 1)
    ys = np.linspace(header_bottom, b, n_records * 3 + 1).astype(int)
    for y in ys[1:-1]:
        cv2.line(img, (l, int(y)), (r, int(y)), 0, 2)
    col_fracs = (0.08, 0.20, 0.35, 0.50, 0.70, 0.85)
    xs = [l + int((r - l) * fx) for fx in col_fracs]
    for x in xs:
        cv2.line(img, (x, t), (x, b), 0, 2)
    truth = {"table": TABLE, "header_bottom": header_bottom,
             "body_ys": [int(y) for y in ys], "col_xs": xs}
    return img, truth

WARP_DST = np.float32([[90, 70], [1130, 55], [1150, 800], [70, 815]])

def warp_page(img):
    """Apply a known mild perspective warp (simulates a photographed page)."""
    l, t, r, b = TABLE
    src = np.float32([[l, t], [r, t], [r, b], [l, b]])
    H = cv2.getPerspectiveTransform(src, WARP_DST)
    return cv2.warpPerspective(img, H, (PAGE_W, PAGE_H), borderValue=235)
