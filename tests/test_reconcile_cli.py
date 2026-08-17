import importlib
import json
import os

import cv2
import numpy as np

from extraction import FIELD_NAMES


def F(**over):
    f = {n: {"value": "", "confidence": "high"} for n in FIELD_NAMES}
    for k, v in over.items():
        f[k] = {"value": v, "confidence": "high"}
    return f


def _page(tmp_path, stem, extraction, model="gemini-3.7-flash"):
    seg = tmp_path / "segments" / stem
    seg.mkdir(parents=True, exist_ok=True)
    n = len(extraction)
    man = [{"index": i, "y0": 100 + 120 * (i - 1), "y1": 100 + 120 * i,
            "strip": f"{stem}_rec{i}.png", "pad_top": 0, "pad_bottom": 0}
           for i in range(1, n + 1)]
    (seg / f"{stem}.json").write_text(json.dumps(
        {"stem": stem, "status": "ok", "records": man,
         "header_band": [0, 100], "warnings": [], "col_x": []}))
    cv2.imwrite(str(seg / f"{stem}_full.png"),
                np.full((700, 1000), 200, np.uint8))
    ex = tmp_path / "ex" / model
    ex.mkdir(parents=True, exist_ok=True)
    (ex / f"{stem}.json").write_text(json.dumps(
        {"stem": stem, "model": model, "prompt_version": "3",
         "records": {str(i): {"fields": extraction[i - 1],
                              "usage": {"input_tokens": 1, "output_tokens": 1,
                                        "latency_s": 0}}
                     for i in range(1, n + 1)}, "totals": {}}))


def test_cli_no_repair_review_exit2(tmp_path, capsys):
    _page(tmp_path, "reg_p1",
          [F(village="Bulaga", village_code="2", day="16", month="3")])
    cli = importlib.import_module("1e_reconcile")
    rc = cli.main(["reg_p1", "--no-repair",
                   "--segments-dir", str(tmp_path / "segments"),
                   "--extractions-dir", str(tmp_path / "ex"),
                   "--out", str(tmp_path / "rec")])
    out = capsys.readouterr().out
    assert rc == 2
    assert "review" in out.lower()
    assert os.path.isfile(
        str(tmp_path / "rec" / "gemini-3.7-flash" / "reg_p1.json"))


def test_cli_clean_page_exit0(tmp_path, capsys):
    _page(tmp_path, "reg_p2",
          [F(record_no="304", patient_name="A B", sex="M", diagnosis="X",
             full_cost="100", first_time_odh="N", hh_owns_phone="Y",
             hh_owns_toilet="Y", result_pn="P", treatment_line1="T1"),
           F(treatment_line2="T2", tab_no="4")])
    cli = importlib.import_module("1e_reconcile")
    rc = cli.main(["reg_p2", "--no-repair",
                   "--segments-dir", str(tmp_path / "segments"),
                   "--extractions-dir", str(tmp_path / "ex"),
                   "--out", str(tmp_path / "rec")])
    assert rc == 0
    saved = json.load(open(
        str(tmp_path / "rec" / "gemini-3.7-flash" / "reg_p2.json")))
    assert len(saved["patients"]) == 1
    assert len(saved["patients"][0]["fields"]["treatments"]) == 2
