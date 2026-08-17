import numpy as np
from conftest import draw_table
from segmentation import (find_table_quad, rectify_page, detect_h_lines,
                          detect_v_lines, group_records, CANON_H, N_RECORDS)


def _rectified():
    img, truth = draw_table()
    rect = rectify_page(img, find_table_quad(img))
    return rect, truth


def test_detect_h_lines_finds_all_body_rows():
    rect, truth = _rectified()
    ys = detect_h_lines(rect)
    # border top+bottom, header line, and 14 interior body lines ≈ 17 lines
    assert len(ys) >= N_RECORDS * 3 + 1


def test_detect_v_lines_finds_columns():
    rect, truth = _rectified()
    xs = detect_v_lines(rect)
    assert len(xs) >= len(truth["col_xs"])            # 6 columns + 2 borders


def test_group_records_returns_five_snapped_blocks():
    rect, _ = _rectified()
    h_lines = detect_h_lines(rect)
    header_bottom, records, warnings = group_records(h_lines, CANON_H)
    assert len(records) == N_RECORDS
    assert not [w for w in warnings if "using ideal" in w]
    assert records[0][0] == header_bottom
    for (a0, a1), (b0, b1) in zip(records, records[1:]):
        assert a1 == b0                                # contiguous blocks
    heights = [y1 - y0 for y0, y1 in records]
    assert max(heights) - min(heights) < 0.02 * CANON_H


def test_group_records_warns_without_lines():
    header_bottom, records, warnings = group_records([], CANON_H)
    assert len(records) == N_RECORDS
    assert any("using ideal" in w for w in warnings)
