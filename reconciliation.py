"""
reconciliation.py — Per-strip extractions → per-patient records.

Phase 3 of odh-vision: continuation merge, validator-based conflict
resolution, boundary repair, page checks. Pure logic except the repair
re-read, which is injected (extract_fn) so tests never touch a network.
See docs/superpowers/specs/2026-08-17-reconciliation-design.md.
"""

import json
import os
import re

import cv2
import numpy as np

from config import SEGMENTS_DIR, EXTRACTIONS_DIR, RECONCILED_DIR
from segmentation import SUBROWS_PER_RECORD
from extraction import FIELD_NAMES, estimate_cost

RECONCILER_VERSION = "1"

CLIP_MIN_EMPTY = 5
CLIP_FIELDS = ["patient_name", "sex", "first_time_odh", "hh_owns_phone",
               "hh_owns_toilet", "result_pn", "diagnosis", "full_cost"]
CONTENT_FIELDS = ["treatment_line1", "treatment_line2", "treatment_line3",
                  "tab_no", "full_cost", "balance", "cost_after_discount",
                  "diagnosis"]
IDENTITY_FIELDS = ["sex", "age_yrs", "village"]
TREATMENT_FIELDS = ("treatment_line1", "treatment_line2", "treatment_line3")

REPAIR_NOTE = ("REPAIR RE-READ: this crop extends slightly beyond one "
               "record. Exactly one complete patient record lies between "
               "the red lines; fragments of adjacent records may appear at "
               "the extreme top/bottom edges. Transcribe the single "
               "complete record.")


def _v(fields, name):
    """Stripped value of a field, or empty string."""
    r = fields.get(name)
    return r["value"].strip() if r else ""


def _norm(s):
    return " ".join(str(s).split()).casefold()


# ─── Validators ──────────────────────────────────────────────────────────────

def _int_range(lo, hi):
    return lambda v: v.isdigit() and lo <= int(v) <= hi


def _numeric(v):
    return bool(re.fullmatch(r"\d+(\.\d+)?", v))


VALIDATORS = {
    "day": _int_range(1, 31),
    "month": _int_range(1, 12),
    "time_hh": _int_range(0, 12),
    "time_mm": _int_range(0, 59),
    "record_no": lambda v: v.isdigit(),
    "age_yrs": _numeric,
    "weight_kg": _numeric,
    "full_cost": _numeric,
    "balance": _numeric,
    "cost_after_discount": _numeric,
    "sex": lambda v: v in ("M", "F"),
    "result_pn": lambda v: v in ("P", "N"),
    "am_pm": lambda v: v in ("AM", "PM"),
}


# ─── Classification ──────────────────────────────────────────────────────────

def is_continuation(primary_fields, cur):
    """True if *cur* looks like an overflow block of *primary_fields*."""
    rn_p, rn_c = _v(primary_fields, "record_no"), _v(cur, "record_no")
    if rn_c and rn_p and rn_c != rn_p:
        return False
    n_p, n_c = _norm(_v(primary_fields, "patient_name")), \
        _norm(_v(cur, "patient_name"))
    if n_c and n_p and n_c != n_p:
        return False
    for fld in IDENTITY_FIELDS:
        a, b = _norm(_v(primary_fields, fld)), _norm(_v(cur, fld))
        if a and b and a != b:
            return False
    return any(_v(cur, f) for f in CONTENT_FIELDS)


def continuation_conflict(primary_fields, cur):
    """Reason string when *cur* half-matches the primary (needs review)."""
    rn_p, rn_c = _v(primary_fields, "record_no"), _v(cur, "record_no")
    n_p, n_c = _norm(_v(primary_fields, "patient_name")), \
        _norm(_v(cur, "patient_name"))
    if rn_c and rn_p and rn_c == rn_p and n_c and n_p and n_c != n_p:
        return "ambiguous-continuation"
    return None


def classify_strips(records):
    """[(key, "primary"|"continuation"|"empty", review_reason|None)]."""
    out = []
    current_primary = None
    for key in sorted(records, key=int):
        fields = records[key]["fields"]
        if not any(_v(fields, n) for n in FIELD_NAMES):
            out.append((int(key), "empty", None))
            continue
        if current_primary is not None:
            reason = continuation_conflict(current_primary, fields)
            if reason:
                out.append((int(key), "primary", reason))
                current_primary = fields
                continue
            if is_continuation(current_primary, fields):
                out.append((int(key), "continuation", None))
                continue
        out.append((int(key), "primary", None))
        current_primary = fields
    return out


# ─── Merge ───────────────────────────────────────────────────────────────────

def merge_patient(strips):
    """Merge [(key, fields), ...] (first = primary) into one patient dict."""
    prim_key, prim = strips[0]
    fields = {n: dict(prim[n]) for n in FIELD_NAMES
              if n not in TREATMENT_FIELDS and n != "tab_no"}
    treatments = [dict(prim[t]) for t in TREATMENT_FIELDS if _v(prim, t)]
    tab_nos = [dict(prim["tab_no"])] if _v(prim, "tab_no") else []

    filled, resolved, warnings = [], [], []
    review = False
    for key, cont in strips[1:]:
        treatments += [dict(cont[t]) for t in TREATMENT_FIELDS if _v(cont, t)]
        if _v(cont, "tab_no"):
            tab_nos.append(dict(cont["tab_no"]))
        for n in fields:
            cv = _v(cont, n)
            if not cv:
                continue
            pv = fields[n]["value"].strip()
            if not pv:
                fields[n] = dict(cont[n])
                filled.append(n)
                continue
            if _norm(pv) == _norm(cv):
                continue
            valid = VALIDATORS.get(n)
            if valid:
                pok, cok = valid(pv), valid(cv)
                if cok and not pok:
                    resolved.append({"field": n, "kept": cv, "rejected": pv,
                                     "how": "validator"})
                    fields[n] = dict(cont[n])
                    continue
                if pok and not cok:
                    resolved.append({"field": n, "kept": pv, "rejected": cv,
                                     "how": "validator"})
                    continue
            warnings.append({"field": n, "primary": pv, "continuation": cv})
            review = True

    fields["treatments"] = treatments
    fields["tab_nos"] = tab_nos
    return {"fields": fields,
            "source_strips": [k for k, _ in strips],
            "merged_from": [k for k, _ in strips[1:]],
            "filled_from_continuation": filled,
            "resolved_conflicts": resolved,
            "warnings": warnings,
            "review": review}
