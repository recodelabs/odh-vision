import csv
import importlib
import json
import os

from extraction import FIELD_NAMES

TREATMENT_LINE_KEYS = ("treatment_line1", "treatment_line2",
                       "treatment_line3", "tab_no")


def PF(treatments=None, tab_nos=None, **over):
    """Merged-patient fields dict: full field set (Reading dicts) minus the
    raw per-strip treatment_line1-3/tab_no keys, plus consolidated
    treatments/tab_nos lists — the shape reconciliation.merge_patient
    actually produces (see tests/test_reconcile_page.py's F() for the
    per-strip analogue this adapts). Override a field with a plain value
    for confidence "high", or a (value, confidence) tuple."""
    names = [n for n in FIELD_NAMES if n not in TREATMENT_LINE_KEYS]
    f = {}
    for n in names:
        if n in over:
            v = over[n]
            val, conf = v if isinstance(v, tuple) else (v, "high")
            f[n] = {"value": val, "confidence": conf}
        else:
            f[n] = {"value": "", "confidence": "high"}
    f["treatments"] = treatments or []
    f["tab_nos"] = tab_nos or []
    return f


def PT(seq, record_no="", review=False, warnings=None, repaired_fields=None,
      source_strips=None, merged_from=None, fields=None):
    return {
        "fields": fields if fields is not None else PF(),
        "source_strips": source_strips if source_strips is not None else [seq],
        "merged_from": merged_from or [],
        "filled_from_continuation": [],
        "resolved_conflicts": [],
        "warnings": warnings or [],
        "review": review,
        "repaired_fields": repaired_fields or [],
        "seq": seq,
        "record_no": record_no,
    }


def PAGE(stem, model, patients):
    return {
        "stem": stem, "model": model, "reconciler_version": "1",
        "patients": patients,
        "page_checks": {"record_no_sequence": "ok", "warnings": [],
                        "review": False},
        "repair_usage": {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                         "est_cost_usd": 0.0},
    }


def _write_reconciled(tmp_path, model, stem, page):
    d = tmp_path / "rec" / model
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{stem}.json").write_text(json.dumps(page))


EXPECTED_COLUMNS = [
    "record_id", "page_stem", "seq", "record_no", "patient_name", "sex",
    "age_yrs", "village", "village_code", "day", "month", "time",
    "first_time_odh", "first_voucher_use", "last_care", "group_appt",
    "hh_owns_phone", "hh_owns_toilet", "hoh_education", "tests",
    "result_pn", "malaria", "sev_malaria", "weight_kg", "diagnosis",
    "art_dose", "voucher_na", "voucher_color", "voucher_id", "treatments",
    "tab_nos", "full_cost", "balance", "cost_after_discount",
    "low_confidence_fields", "review", "review_reasons", "repaired_fields",
    "source_strips", "merged", "model", "reconciler_version",
]


def _build_pages(tmp_path, model="gemini-3.7-flash"):
    # reg_p2: two patients. seq 1 = merged, reviewed, blank time.
    # seq 2 = has treatments + a low-confidence field + a low-confidence
    # treatment entry.
    p2a = PT(1, record_no="304", review=True,
             warnings=[{"field": "patient_name", "primary": "Aciro Rose",
                       "continuation": "mwende Betty"},
                      {"reason": "clipped-no-repair", "strip": 1}],
             repaired_fields=["village"], source_strips=[1, 2],
             merged_from=[2],
             fields=PF(patient_name="Aciro Rose", sex="M", village="Bulaga"))
    p2b = PT(2, record_no="305", review=False,
             fields=PF(patient_name="Namono Grace", sex="F",
                       diagnosis=("PUD", "low"),
                       treatments=[{"value": "T1", "confidence": "high"},
                                  {"value": "T2", "confidence": "low"}],
                       time_hh="8", time_mm="05", am_pm="AM"))
    reg_p2 = PAGE("reg_p2", model, [p2a, p2b])

    # reg_p10: one patient — exists to prove natural (not lexicographic)
    # page-stem ordering: "reg_p2" must sort before "reg_p10".
    p10a = PT(1, record_no="410", review=False,
             fields=PF(patient_name="Okello Sam", sex="M"))
    reg_p10 = PAGE("reg_p10", model, [p10a])

    _write_reconciled(tmp_path, model, "reg_p2", reg_p2)
    _write_reconciled(tmp_path, model, "reg_p10", reg_p10)
    return model


def _run_export(tmp_path, model, out=None):
    cli = importlib.import_module("1f_export")
    out = out or str(tmp_path / "export.csv")
    rc = cli.main(["--model", model, "--out", out,
                  "--reconciled-dir", str(tmp_path / "rec")])
    return rc, out


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_export_row_count_and_column_order(tmp_path):
    model = _build_pages(tmp_path)
    rc, out = _run_export(tmp_path, model)
    assert rc == 0
    assert os.path.isfile(out)
    with open(out, newline="") as f:
        header = next(csv.reader(f))
    assert header == EXPECTED_COLUMNS
    rows = _rows(out)
    assert len(rows) == 3


def test_export_natural_page_sort_and_seq_order(tmp_path):
    model = _build_pages(tmp_path)
    _, out = _run_export(tmp_path, model)
    rows = _rows(out)
    assert [r["page_stem"] for r in rows] == ["reg_p2", "reg_p2", "reg_p10"]
    assert [r["seq"] for r in rows] == ["1", "2", "1"]
    assert [r["record_id"] for r in rows] == [
        "reg_p2-1", "reg_p2-2", "reg_p10-1"]


def test_export_treatments_join(tmp_path):
    model = _build_pages(tmp_path)
    _, out = _run_export(tmp_path, model)
    rows = _rows(out)
    p2b = rows[1]
    assert p2b["treatments"] == "T1; T2"


def test_export_low_confidence_fields(tmp_path):
    model = _build_pages(tmp_path)
    _, out = _run_export(tmp_path, model)
    rows = _rows(out)
    p2a, p2b = rows[0], rows[1]
    assert p2a["low_confidence_fields"] == ""
    assert p2b["low_confidence_fields"] == "diagnosis,treatments[1]"


def test_export_review_flag_and_reasons(tmp_path):
    model = _build_pages(tmp_path)
    _, out = _run_export(tmp_path, model)
    rows = _rows(out)
    p2a, p2b, p10a = rows
    assert p2a["review"] == "TRUE"
    assert p2a["review_reasons"] == "field:patient_name; clipped-no-repair"
    assert p2b["review"] == "FALSE"
    assert p2b["review_reasons"] == ""


def test_export_merged_and_source_strips(tmp_path):
    model = _build_pages(tmp_path)
    _, out = _run_export(tmp_path, model)
    rows = _rows(out)
    p2a = rows[0]
    assert p2a["merged"] == "TRUE"
    assert p2a["source_strips"] == "1+2"
    assert p2a["repaired_fields"] == "village"
    p10a = rows[2]
    assert p10a["merged"] == "FALSE"
    assert p10a["source_strips"] == "1"


def test_export_time_assembly(tmp_path):
    model = _build_pages(tmp_path)
    _, out = _run_export(tmp_path, model)
    rows = _rows(out)
    p2a, p2b = rows[0], rows[1]
    assert p2a["time"] == ""          # blank-safe: all three empty
    assert p2b["time"] == "8:05 AM"


def test_export_model_and_reconciler_version_columns(tmp_path):
    model = _build_pages(tmp_path)
    _, out = _run_export(tmp_path, model)
    rows = _rows(out)
    assert all(r["model"] == model for r in rows)
    assert all(r["reconciler_version"] == "1" for r in rows)


def test_export_default_out_path_is_gitignored_tree(tmp_path):
    model = _build_pages(tmp_path)
    cli = importlib.import_module("1f_export")
    rc = cli.main(["--model", model,
                  "--reconciled-dir", str(tmp_path / "rec")])
    assert rc == 0
    default_out = str(tmp_path / "rec" / model / "export.csv")
    assert os.path.isfile(default_out)


def test_export_summary_printed(tmp_path, capsys):
    model = _build_pages(tmp_path)
    _run_export(tmp_path, model)
    out = capsys.readouterr().out.lower()
    assert "2" in out and "pages" in out
    assert "3" in out and "patient" in out
    assert "review" in out
