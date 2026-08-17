import os
import json
import cv2
import numpy as np
from conftest import draw_table, warp_page
import segmentation
from segmentation import segment_page, N_RECORDS, CANON_W, STRIP_PAD


def _page_on_disk(tmp_path, transform=None):
    img, _ = draw_table(margins=True)
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
    # page is upright by construction (margins=True draws the true
    # above-table legend denser than the below-table one) -- no flip
    # should have been detected
    assert not any("rotated 180" in w for w in m["warnings"])
    for rec in m["records"]:
        # strips are now 3-channel (red true-boundary lines drawn on them)
        strip = cv2.imread(os.path.join(out, rec["strip"]))
        assert strip is not None
        assert strip.shape[1] == CANON_W
        assert strip.shape[2] == 3
        # strip = header band + padded record block
        assert strip.shape[0] == (m["header_band"][1]
                                  + (rec["y1"] - rec["y0"])
                                  + rec["pad_top"] + rec["pad_bottom"])
    assert os.path.isfile(os.path.join(out, "reg_p1_full.png"))
    assert os.path.isfile(os.path.join(out, "reg_p1_debug.jpg"))
    with open(os.path.join(out, "reg_p1.json")) as f:
        assert json.load(f)["stem"] == "reg_p1"
    assert len(m["col_x"]) >= 6


def test_segment_page_strip_padding_and_boundary_lines(tmp_path):
    path = _page_on_disk(tmp_path, warp_page)
    out = str(tmp_path / "seg")
    m = segment_page(path, out)
    assert m["status"] == "ok"

    records = m["records"]
    # rec1's y0 == header_bottom, so its top padding is clamped to 0.
    assert records[0]["y0"] == m["header_band"][1]
    assert records[0]["pad_top"] == 0

    # Middle records (not touching header_bottom or the page bottom) get
    # the full pad on both sides.
    for rec in records[1:-1]:
        assert rec["pad_top"] == STRIP_PAD
        assert rec["pad_bottom"] == STRIP_PAD

    # Each strip carries red pixels at the rows corresponding to the true
    # y0/y1 boundaries (BGR: blue<100, green<100, red>200).
    header_h = m["header_band"][1]
    for rec in records:
        strip = cv2.imread(os.path.join(out, rec["strip"]))
        assert strip is not None
        line_y0 = header_h + rec["pad_top"]
        line_y1 = header_h + rec["pad_top"] + (rec["y1"] - rec["y0"])
        for line_y in (line_y0, line_y1):
            row = strip[line_y]
            red_pixels = ((row[:, 0] < 100) & (row[:, 1] < 100)
                         & (row[:, 2] > 200))
            assert red_pixels.sum() > 0.5 * row.shape[0]


def test_segment_page_handles_portrait_input(tmp_path):
    rot = lambda im: cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
    m = segment_page(_page_on_disk(tmp_path, rot), str(tmp_path / "seg"))
    assert m["status"] == "ok"
    assert len(m["records"]) == N_RECORDS


def test_segment_page_needs_review_on_non_monotonic_records(tmp_path, monkeypatch):
    # group_records is contractually expected to always return monotonic
    # (y0 < y1) blocks, but segment_page must not trust that blindly --
    # exercise its own gate directly by forcing a degenerate return value.
    path = _page_on_disk(tmp_path, warp_page)
    out = str(tmp_path / "seg")

    def fake_group_records(h_lines, table_h, **kwargs):
        header_bottom = int(table_h * segmentation.HEADER_FRAC)
        degenerate = [(header_bottom, header_bottom)] * N_RECORDS
        return header_bottom, degenerate, []

    monkeypatch.setattr(segmentation, "group_records", fake_group_records)
    m = segmentation.segment_page(path, out)

    assert m["status"] == "needs_review"
    assert m["records"] == []
    assert any("non-monotonic record bounds" in w for w in m["warnings"])
    assert not [f for f in os.listdir(out) if "_rec" in f]   # no strips


def test_segment_page_needs_review_on_blank(tmp_path):
    blank = str(tmp_path / "blank_p1.png")
    cv2.imwrite(blank, np.full((850, 1200), 235, np.uint8))
    out = str(tmp_path / "seg")
    m = segment_page(blank, out)
    assert m["status"] == "needs_review"
    assert m["records"] == []
    assert not [f for f in os.listdir(out) if "_rec" in f]   # no strips
    assert os.path.isfile(os.path.join(out, "blank_p1.json"))
