import json
import os

import numpy as np
import cv2

from extraction import FIELD_NAMES
from reconciliation import reconcile_page, _check_sequence


def F(**over):
    f = {n: {"value": "", "confidence": "high"} for n in FIELD_NAMES}
    for k, v in over.items():
        f[k] = {"value": v, "confidence": "high"}
    return f


def _page(tmp_path, stem="reg_p1", status="ok", records=None,
          extraction=None, model="m"):
    seg = tmp_path / "segments" / stem
    seg.mkdir(parents=True, exist_ok=True)
    n = len(extraction)
    man_records = [{"index": i, "y0": 100 + 120 * (i - 1),
                    "y1": 100 + 120 * i, "strip": f"{stem}_rec{i}.png",
                    "pad_top": 0, "pad_bottom": 0}
                   for i in range(1, n + 1)]
    (seg / f"{stem}.json").write_text(json.dumps(
        {"stem": stem, "status": status, "records": man_records,
         "header_band": [0, 100], "warnings": [], "col_x": []}))
    cv2.imwrite(str(seg / f"{stem}_full.png"),
                np.full((700, 1000), 200, np.uint8))
    ex = tmp_path / "ex" / model
    ex.mkdir(parents=True, exist_ok=True)
    (ex / f"{stem}.json").write_text(json.dumps(
        {"stem": stem, "model": model, "prompt_version": "3",
         "records": {str(i): {"fields": extraction[i - 1],
                              "usage": {"input_tokens": 1,
                                        "output_tokens": 1,
                                        "latency_s": 0}}
                     for i in range(1, n + 1)},
         "totals": {}}))
    return str(tmp_path / "segments"), str(tmp_path / "ex")


def test_reconcile_page_merges_and_writes(tmp_path):
    ex = [F(record_no="304", patient_name="Aciro Rose", sex="M",
            diagnosis="PID", treatment_line1="T1", full_cost="26000",
            first_time_odh="N", hh_owns_phone="Y", hh_owns_toilet="Y",
            result_pn="P"),
          F(treatment_line1="T2", tab_no="10"),
          F(record_no="305", patient_name="Namono Grace", sex="M",
            diagnosis="PUD", treatment_line1="T3", full_cost="28000",
            first_time_odh="N", hh_owns_phone="Y", hh_owns_toilet="N",
            result_pn="N")]
    seg, exd = _page(tmp_path, extraction=ex)
    out = str(tmp_path / "rec")
    r = reconcile_page("reg_p1", "m", segments_dir=seg, extractions_dir=exd,
                       out_base=out)
    assert len(r["patients"]) == 2
    p1 = r["patients"][0]
    assert [t["value"] for t in p1["fields"]["treatments"]] == ["T1", "T2"]
    assert p1["merged_from"] == [2]
    assert r["page_checks"]["record_no_sequence"] == "ok"
    saved = json.load(open(os.path.join(out, "m", "reg_p1.json")))
    assert saved["reconciler_version"] == "1"
    assert len(saved["patients"]) == 2


def test_reconcile_page_clip_without_repair_flags_review(tmp_path):
    clipped = F(village="Bulaga", village_code="2", day="16", month="3")
    seg, exd = _page(tmp_path, extraction=[clipped])
    r = reconcile_page("reg_p1", "m", segments_dir=seg, extractions_dir=exd,
                       out_base=str(tmp_path / "rec"), extract_fn=None)
    p = r["patients"][0]
    assert p["review"] is True
    assert any("clipped-no-repair" in str(w) for w in p["warnings"])


def test_reconcile_page_repairs_with_extract_fn(tmp_path):
    clipped = F(village="Bulaga", village_code="2", day="16", month="3")
    seg, exd = _page(tmp_path, extraction=[clipped])

    reread = F(patient_name="Okwir Moses", sex="M", first_time_odh="Y",
               hh_owns_phone="Y", hh_owns_toilet="Y", diagnosis="Malaria",
               full_cost="3500", village="IGNORED")

    class Rec:
        def model_dump(self):
            return reread

    def extract_fn(image_path, record_index, context):
        return Rec(), {"input_tokens": 5000, "output_tokens": 500,
                       "latency_s": 3.0}

    r = reconcile_page("reg_p1", "m", segments_dir=seg, extractions_dir=exd,
                       out_base=str(tmp_path / "rec"), extract_fn=extract_fn)
    p = r["patients"][0]
    assert p["fields"]["patient_name"]["value"] == "Okwir Moses"
    assert p["fields"]["village"]["value"] == "Bulaga"     # not overwritten
    assert set(p["repaired_fields"]) >= {"patient_name", "sex",
                                         "first_time_odh"}
    assert r["repair_usage"]["calls"] == 1
    assert r["repair_usage"]["input_tokens"] == 5000


def test_reconcile_page_refusals_and_resume(tmp_path):
    ex = [F(record_no="304", patient_name="A", sex="M", diagnosis="X",
            full_cost="100", first_time_odh="N", hh_owns_phone="Y",
            hh_owns_toilet="Y", result_pn="P")]
    seg, exd = _page(tmp_path, status="needs_review", extraction=ex)
    r = reconcile_page("reg_p1", "m", segments_dir=seg, extractions_dir=exd,
                       out_base=str(tmp_path / "rec"))
    assert r["refused"] == "needs_review"

    seg2, exd2 = _page(tmp_path / "b", extraction=ex)
    out = str(tmp_path / "b" / "rec")
    reconcile_page("reg_p1", "m", segments_dir=seg2, extractions_dir=exd2,
                   out_base=out)
    r2 = reconcile_page("reg_p1", "m", segments_dir=seg2,
                        extractions_dir=exd2, out_base=out)
    assert r2.get("skipped") is True


def test_check_sequence():
    assert _check_sequence(["304", "305", "306"]) == "ok"
    assert _check_sequence(["304", "306"]) == "gap"
    assert _check_sequence(["304", "304"]) == "duplicate"
    assert _check_sequence(["304", "abc"]) == "non-numeric"
