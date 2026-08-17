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
            diagnosis="PID", treatment_line1="T1", full_cost="27500",
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


def test_reconcile_page_recno_mismatch_name_match_merges_no_review(tmp_path):
    # Two strips, same true patient, matching names but a misread
    # record_no ("307" vs "3067") — must merge into ONE patient with a
    # recno-mismatch-name-match warning, and review must stay False.
    ex = [F(record_no="307", patient_name="Aciro Rose", sex="M",
            diagnosis="PID", treatment_line1="T1", full_cost="27500",
            first_time_odh="N", hh_owns_phone="Y", hh_owns_toilet="Y",
            result_pn="P"),
          F(record_no="3067", patient_name="Aciro Rose",
            treatment_line1="T2", tab_no="10")]
    seg, exd = _page(tmp_path, extraction=ex)
    r = reconcile_page("reg_p1", "m", segments_dir=seg, extractions_dir=exd,
                       out_base=str(tmp_path / "rec"))
    assert len(r["patients"]) == 1
    p = r["patients"][0]
    assert p["review"] is False
    assert p["record_no"] == "307"
    assert any(w.get("reason") == "recno-mismatch-name-match"
              for w in p["warnings"])
    assert [t["value"] for t in p["fields"]["treatments"]] == ["T1", "T2"]


def test_reconcile_page_possible_split_flags_both_patients(tmp_path):
    # Two adjacent, fully-classified primaries whose record_no values are
    # a plausible misread of each other ("307"/"3067", edit distance 1) —
    # neither has a clip signature and their names are dissimilar enough
    # that classify_strips never treats them as a continuation, so they
    # land as two separate patients. possible_split must flag BOTH for
    # review without merging them.
    ex = [F(record_no="307", patient_name="Xyz Alpha", sex="M",
            first_time_odh="N", hh_owns_phone="Y", hh_owns_toilet="Y",
            result_pn="P", diagnosis="D1", full_cost="100",
            treatment_line1="T1"),
          F(record_no="3067", patient_name="Totally Different Person",
            sex="F", first_time_odh="Y", hh_owns_phone="N",
            hh_owns_toilet="N", result_pn="N", diagnosis="D2",
            full_cost="200", treatment_line1="T2")]
    seg, exd = _page(tmp_path, extraction=ex)
    r = reconcile_page("reg_p1", "m", segments_dir=seg, extractions_dir=exd,
                       out_base=str(tmp_path / "rec"))
    assert len(r["patients"]) == 2
    p1, p2 = r["patients"]
    assert p1["review"] is True and p2["review"] is True
    assert any(w.get("reason") == "possible-same-patient"
              for w in p1["warnings"])
    assert any(w.get("reason") == "possible-same-patient"
              for w in p2["warnings"])


def test_reconcile_page_no_possible_split_for_distinct_sequential_patients(
        tmp_path):
    # Two ordinary sequential patients (304, 305) must NOT be flagged —
    # same-length record_nos differing by one digit is the normal shape
    # of consecutive real patients, not a misread.
    ex = [F(record_no="304", patient_name="Aciro Rose", sex="M",
            first_time_odh="N", hh_owns_phone="Y", hh_owns_toilet="Y",
            result_pn="P", diagnosis="PID", full_cost="27500",
            treatment_line1="T1"),
          F(record_no="305", patient_name="Okello Sam", sex="M",
            first_time_odh="N", hh_owns_phone="Y", hh_owns_toilet="N",
            result_pn="N", diagnosis="PUD", full_cost="28000",
            treatment_line1="T2")]
    seg, exd = _page(tmp_path, extraction=ex)
    r = reconcile_page("reg_p1", "m", segments_dir=seg, extractions_dir=exd,
                       out_base=str(tmp_path / "rec"))
    p1, p2 = r["patients"]
    assert p1["review"] is False and p2["review"] is False
    assert not any(w.get("reason") == "possible-same-patient"
                  for w in p1["warnings"] + p2["warnings"])


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
               full_cost="3200", village="IGNORED")

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


def test_reconcile_page_missing_extraction_flags_review(tmp_path):
    # Manifest has 2 record slots; the extraction only has key "1" — record
    # 2 was silently dropped somewhere upstream. That must surface as a
    # page-level warning and force page_checks.review, not vanish quietly.
    stem = "reg_p1"
    seg = tmp_path / "segments" / stem
    seg.mkdir(parents=True, exist_ok=True)
    man_records = [{"index": 1, "y0": 100, "y1": 220,
                    "strip": f"{stem}_rec1.png", "pad_top": 0, "pad_bottom": 0},
                   {"index": 2, "y0": 220, "y1": 340,
                    "strip": f"{stem}_rec2.png", "pad_top": 0, "pad_bottom": 0}]
    (seg / f"{stem}.json").write_text(json.dumps(
        {"stem": stem, "status": "ok", "records": man_records,
         "header_band": [0, 100], "warnings": [], "col_x": []}))
    cv2.imwrite(str(seg / f"{stem}_full.png"),
               np.full((700, 1000), 200, np.uint8))
    ex_dir = tmp_path / "ex" / "m"
    ex_dir.mkdir(parents=True, exist_ok=True)
    rec1 = F(record_no="304", patient_name="A", sex="M", diagnosis="X",
             full_cost="100", first_time_odh="N", hh_owns_phone="Y",
             hh_owns_toilet="Y", result_pn="P")
    (ex_dir / f"{stem}.json").write_text(json.dumps(
        {"stem": stem, "model": "m", "prompt_version": "3",
         "records": {"1": {"fields": rec1,
                           "usage": {"input_tokens": 1, "output_tokens": 1,
                                    "latency_s": 0}}},
         "totals": {}}))
    r = reconcile_page(stem, "m", segments_dir=str(tmp_path / "segments"),
                       extractions_dir=str(tmp_path / "ex"),
                       out_base=str(tmp_path / "rec"))
    assert any(w.get("reason") == "missing-extraction" and w.get("strip") == 2
              for w in r["page_checks"]["warnings"])
    assert r["page_checks"]["review"] is True


def test_reconcile_page_no_patients_flags_review(tmp_path):
    # A non-empty manifest whose sole record is entirely blank yields zero
    # patients — that must not pass silently as a clean, empty page.
    ex = [F()]
    seg, exd = _page(tmp_path, extraction=ex)
    r = reconcile_page("reg_p1", "m", segments_dir=seg, extractions_dir=exd,
                       out_base=str(tmp_path / "rec"))
    assert r["patients"] == []
    assert any(w.get("reason") == "no-patients"
              for w in r["page_checks"]["warnings"])
    assert r["page_checks"]["review"] is True


def test_reconcile_page_possible_split_survives_repair_masking(tmp_path):
    # patient_b is a single-strip clip-shaped fragment (checkbox cluster
    # empty, only patient_name populated among CLIP_FIELDS) that must be
    # flagged possible-same-patient. Its clip signature also triggers a
    # repair re-read that fills the checkbox cluster from REPAIRABLE_FIELDS
    # — if possible_split evaluated the POST-repair fields, the now-filled
    # fragment would look complete and the split would go silent. It must
    # still be flagged, using the pre-repair snapshot.
    ex = [F(record_no="204", patient_name="Xyz Alpha", sex="M",
            first_time_odh="Y", hh_owns_phone="Y", hh_owns_toilet="Y",
            result_pn="P", diagnosis="Flu", full_cost="100",
            treatment_line1="T1"),
          F(record_no="861", patient_name="Qrs Beta",
            treatment_line1="T2")]
    seg, exd = _page(tmp_path, extraction=ex)

    reread = F(sex="M", first_time_odh="Y", hh_owns_phone="Y",
               hh_owns_toilet="Y")

    class Rec:
        def model_dump(self):
            return reread

    def extract_fn(image_path, record_index, context):
        return Rec(), {"input_tokens": 100, "output_tokens": 20,
                       "latency_s": 1.0}

    r = reconcile_page("reg_p1", "m", segments_dir=seg, extractions_dir=exd,
                       out_base=str(tmp_path / "rec"), extract_fn=extract_fn)
    p1, p2 = r["patients"]
    # Repair actually fired and filled the checkbox cluster on p2 — proves
    # this test is not vacuous: without the pre-repair-snapshot fix the
    # live (post-repair) fields would show the cluster as populated.
    assert set(p2["repaired_fields"]) >= {"sex", "first_time_odh"}
    assert p2["fields"]["sex"]["value"] == "M"
    assert p1["review"] is True and p2["review"] is True
    assert any(w.get("reason") == "possible-same-patient"
              for w in p2["warnings"])
