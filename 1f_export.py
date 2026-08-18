#!/usr/bin/env python3
"""
1f_export.py — Per-patient CSV export from reconciled pages.

Usage:
    python 1f_export.py
    python 1f_export.py --model gemini-3.7-flash --out /tmp/export.csv
    python 1f_export.py --reconciled-dir _reconciled

Reads every _reconciled/<model>/<stem>.json and writes ONE CSV, one row per
patient, pages in natural (numeric) stem order and patients in seq order.
Default output lives inside the gitignored _reconciled/<model>/ tree
(rows carry patient data) — never default to a tracked path.
"""

import argparse
import csv
import glob as globmod
import json
import os
import re
import sys

from config import RECONCILED_DIR

COLUMNS = [
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

LOW_CONFIDENCE = ("low", "illegible")


def _natural_key(s):
    """Sort key that orders embedded digit runs numerically (p2 < p10)."""
    return [int(chunk) if chunk.isdigit() else chunk
            for chunk in re.split(r"(\d+)", s)]


def discover_stems(reconciled_dir, model):
    d = os.path.join(reconciled_dir, model)
    stems = [os.path.splitext(os.path.basename(p))[0]
             for p in globmod.glob(os.path.join(d, "*.json"))]
    return sorted(stems, key=_natural_key)


def _val(fields, name):
    r = fields.get(name)
    return r["value"] if r else ""


def _assemble_time(fields):
    hh = _val(fields, "time_hh")
    mm = _val(fields, "time_mm")
    am_pm = _val(fields, "am_pm")
    if not (hh or mm or am_pm):
        return ""
    return f"{hh}:{mm} {am_pm}".strip()


def _low_confidence_fields(fields):
    """Scalar field names with confidence low/illegible, plus per-treatment
    entries as treatments[i] (in field-declaration, then treatment order)."""
    out = []
    for name, reading in fields.items():
        if name in ("treatments", "tab_nos"):
            continue
        if isinstance(reading, dict) and reading.get("confidence") in LOW_CONFIDENCE:
            out.append(name)
    for i, t in enumerate(fields.get("treatments", [])):
        if t.get("confidence") in LOW_CONFIDENCE:
            out.append(f"treatments[{i}]")
    return out


def _review_reasons(warnings):
    """Compact, "; "-joined reasons: a warning's "reason" verbatim, or
    "field:<name>" for merge-conflict entries that only carry a field."""
    parts = []
    for w in warnings:
        if "reason" in w:
            parts.append(str(w["reason"]))
        elif "field" in w:
            parts.append(f"field:{w['field']}")
    return "; ".join(parts)


def patient_row(stem, model, reconciler_version, patient):
    fields = patient["fields"]
    seq = patient["seq"]
    treatments = "; ".join(t["value"] for t in fields.get("treatments", []))
    tab_nos = "; ".join(t["value"] for t in fields.get("tab_nos", []))
    return {
        "record_id": f"{stem}-{seq}",
        "page_stem": stem,
        "seq": seq,
        "record_no": patient.get("record_no", ""),
        "patient_name": _val(fields, "patient_name"),
        "sex": _val(fields, "sex"),
        "age_yrs": _val(fields, "age_yrs"),
        "village": _val(fields, "village"),
        "village_code": _val(fields, "village_code"),
        "day": _val(fields, "day"),
        "month": _val(fields, "month"),
        "time": _assemble_time(fields),
        "first_time_odh": _val(fields, "first_time_odh"),
        "first_voucher_use": _val(fields, "first_voucher_use"),
        "last_care": _val(fields, "last_care"),
        "group_appt": _val(fields, "group_appt"),
        "hh_owns_phone": _val(fields, "hh_owns_phone"),
        "hh_owns_toilet": _val(fields, "hh_owns_toilet"),
        "hoh_education": _val(fields, "hoh_education"),
        "tests": _val(fields, "tests"),
        "result_pn": _val(fields, "result_pn"),
        "malaria": _val(fields, "malaria"),
        "sev_malaria": _val(fields, "sev_malaria"),
        "weight_kg": _val(fields, "weight_kg"),
        "diagnosis": _val(fields, "diagnosis"),
        "art_dose": _val(fields, "art_dose"),
        "voucher_na": _val(fields, "voucher_na"),
        "voucher_color": _val(fields, "voucher_color"),
        "voucher_id": _val(fields, "voucher_id"),
        "treatments": treatments,
        "tab_nos": tab_nos,
        "full_cost": _val(fields, "full_cost"),
        "balance": _val(fields, "balance"),
        "cost_after_discount": _val(fields, "cost_after_discount"),
        "low_confidence_fields": ",".join(_low_confidence_fields(fields)),
        "review": "TRUE" if patient.get("review") else "FALSE",
        "review_reasons": _review_reasons(patient.get("warnings", [])),
        "repaired_fields": ",".join(patient.get("repaired_fields", [])),
        "source_strips": "+".join(str(k) for k in
                                  patient.get("source_strips", [])),
        "merged": "TRUE" if patient.get("merged_from") else "FALSE",
        "model": model,
        "reconciler_version": reconciler_version,
    }


def export(reconciled_dir, model, out_path):
    stems = discover_stems(reconciled_dir, model)
    rows = []
    n_review = 0
    for stem in stems:
        path = os.path.join(reconciled_dir, model, f"{stem}.json")
        with open(path) as f:
            page = json.load(f)
        page_model = page.get("model", model)
        reconciler_version = page.get("reconciler_version", "")
        for patient in page.get("patients", []):
            row = patient_row(stem, page_model, reconciler_version, patient)
            rows.append(row)
            if row["review"] == "TRUE":
                n_review += 1

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return {"pages": len(stems), "patients": len(rows), "review": n_review,
            "out_path": out_path}


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Export reconciled pages to one per-patient CSV.")
    p.add_argument("--model", default="gemini-3.7-flash")
    p.add_argument("--out", default=None,
                   help="Output CSV path (default: "
                        "<reconciled-dir>/<model>/export.csv)")
    p.add_argument("--reconciled-dir", default=RECONCILED_DIR)
    args = p.parse_args(argv)

    out_path = args.out or os.path.join(
        args.reconciled_dir, args.model, "export.csv")

    summary = export(args.reconciled_dir, args.model, out_path)
    print(f"[{args.model}] {summary['pages']} pages, "
          f"{summary['patients']} patients, {summary['review']} review "
          f"-> {summary['out_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
