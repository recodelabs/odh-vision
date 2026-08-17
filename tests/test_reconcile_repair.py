import os

import cv2
import numpy as np

from extraction import FIELD_NAMES
from reconciliation import (has_clip_signature, build_repair_crop,
                            repair_fields, REPAIR_NOTE)


def F(**over):
    f = {n: {"value": "", "confidence": "high"} for n in FIELD_NAMES}
    for k, v in over.items():
        f[k] = {"value": v, "confidence": "high"}
    return f


def test_clip_signature():
    clipped = F(village="Bulaga", village_code="2", day="16", month="3")
    assert has_clip_signature(clipped)          # 8/8 CLIP_FIELDS empty
    normal = F(patient_name="X", sex="M", first_time_odh="Y",
               hh_owns_phone="Y", diagnosis="Malaria", full_cost="3500")
    assert not has_clip_signature(normal)       # only 2 empties


def test_build_repair_crop_geometry(tmp_path):
    H, W, header_bottom = 700, 1000, 100
    page = np.full((H, W), 200, np.uint8)
    full = str(tmp_path / "full.png")
    cv2.imwrite(full, page)
    y0, y1 = 400, 520                            # record: 120px, subrow 40
    out = str(tmp_path / "crop.png")
    path, top_slack, bottom_slack = build_repair_crop(
        full, header_bottom, y0, y1, out)
    img = cv2.imread(path)
    assert top_slack == 40 and bottom_slack == 40
    assert img.shape[0] == header_bottom + (y1 - y0) + top_slack + bottom_slack
    assert img.shape[2] == 3
    red = (img[:, :, 2] > 200) & (img[:, :, 0] < 100) & (img[:, :, 1] < 100)
    assert red.any()                             # boundary lines drawn


def test_build_repair_crop_clamps(tmp_path):
    H, W, header_bottom = 700, 1000, 100
    cv2.imwrite(str(tmp_path / "full.png"), np.full((H, W), 200, np.uint8))
    # last record touching the bottom: y1 = H-1
    _, top_slack, bottom_slack = build_repair_crop(
        str(tmp_path / "full.png"), header_bottom, 560, 699,
        str(tmp_path / "c.png"))
    assert bottom_slack == 1                     # clamped at page bottom


def test_repair_fills_only_empty_fields(tmp_path):
    H, W = 700, 1000
    cv2.imwrite(str(tmp_path / "full.png"), np.full((H, W), 200, np.uint8))
    fields = F(village="Bulaga", diagnosis="")    # clipped-ish
    reread = F(patient_name="Okwir Moses", sex="M", village="WRONG",
               diagnosis="Malaria")

    class Rec:
        def model_dump(self):
            return reread

    calls = []

    def extract_fn(image_path, record_index, context):
        calls.append((image_path, record_index, context))
        return Rec(), {"input_tokens": 5000, "output_tokens": 600,
                       "latency_s": 4.0}

    entry = {"index": 5, "y0": 400, "y1": 520}
    repaired, usage = repair_fields(fields, entry, 100,
                                    str(tmp_path / "full.png"),
                                    str(tmp_path), extract_fn, context="ctx")
    assert set(repaired) == {"patient_name", "sex", "diagnosis"}
    assert fields["patient_name"]["value"] == "Okwir Moses"
    assert fields["village"]["value"] == "Bulaga"        # never overwritten
    assert usage["input_tokens"] == 5000
    assert REPAIR_NOTE in calls[0][2] and "ctx" in calls[0][2]
    assert calls[0][1] == 5
