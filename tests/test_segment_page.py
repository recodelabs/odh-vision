import os
import json
import cv2
import numpy as np
from conftest import draw_table, warp_page
from segmentation import segment_page, N_RECORDS, CANON_W


def _page_on_disk(tmp_path, transform=None):
    img, _ = draw_table()
    if transform:
        img = transform(img)
    path = str(tmp_path / "reg_p1.png")
    cv2.imwrite(path, img)
    return path


def test_segment_page_ok(tmp_path):
    path = _page_on_disk(tmp_path, warp_page)
    out = str(tmp_path / "seg")
    m = segment_page(path, out)
    assert m["status"] == "ok"
    assert len(m["records"]) == N_RECORDS
    for rec in m["records"]:
        strip = cv2.imread(os.path.join(out, rec["strip"]),
                           cv2.IMREAD_GRAYSCALE)
        assert strip is not None
        assert strip.shape[1] == CANON_W
        # strip = header band + record block
        assert strip.shape[0] == m["header_band"][1] + (rec["y1"] - rec["y0"])
    assert os.path.isfile(os.path.join(out, "reg_p1_full.png"))
    assert os.path.isfile(os.path.join(out, "reg_p1_debug.jpg"))
    with open(os.path.join(out, "reg_p1.json")) as f:
        assert json.load(f)["stem"] == "reg_p1"
    assert len(m["col_x"]) >= 6


def test_segment_page_handles_portrait_input(tmp_path):
    rot = lambda im: cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
    m = segment_page(_page_on_disk(tmp_path, rot), str(tmp_path / "seg"))
    assert m["status"] == "ok"
    assert len(m["records"]) == N_RECORDS


def test_segment_page_needs_review_on_blank(tmp_path):
    blank = str(tmp_path / "blank_p1.png")
    cv2.imwrite(blank, np.full((850, 1200), 235, np.uint8))
    out = str(tmp_path / "seg")
    m = segment_page(blank, out)
    assert m["status"] == "needs_review"
    assert m["records"] == []
    assert not [f for f in os.listdir(out) if "_rec" in f]   # no strips
    assert os.path.isfile(os.path.join(out, "blank_p1.json"))
