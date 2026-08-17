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


# ─── Boundary repair ─────────────────────────────────────────────────────────

def has_clip_signature(fields):
    """True when so many core fields are empty the strip was likely clipped."""
    return sum(1 for f in CLIP_FIELDS if not _v(fields, f)) >= CLIP_MIN_EMPTY


def build_repair_crop(full_page_path, header_bottom, y0, y1, out_path):
    """Expanded crop from the rectified page: one sub-row of slack each side,
    header band stitched, red lines at the EXPANDED bounds. Returns
    (out_path, top_slack, bottom_slack)."""
    img = cv2.imread(full_page_path, cv2.IMREAD_GRAYSCALE)
    H = img.shape[0]
    subrow = max(1, (y1 - y0) // SUBROWS_PER_RECORD)
    top = max(header_bottom, y0 - subrow)
    bottom = min(H, y1 + subrow)
    strip = np.vstack([img[0:header_bottom], img[top:bottom]])
    vis = cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR)
    w = vis.shape[1]
    cv2.line(vis, (0, header_bottom), (w, header_bottom), (0, 0, 255), 2)
    cv2.line(vis, (0, vis.shape[0] - 2), (w, vis.shape[0] - 2),
             (0, 0, 255), 2)
    cv2.imwrite(out_path, vis)
    return out_path, y0 - top, bottom - y1


def repair_fields(fields, rec_entry, header_bottom, full_png, tmp_dir,
                  extract_fn, context=""):
    """One targeted re-read; fill ONLY empty fields. Returns
    (repaired field names, usage)."""
    crop = os.path.join(tmp_dir, f"repair_rec{rec_entry['index']}.png")
    build_repair_crop(full_png, header_bottom,
                      rec_entry["y0"], rec_entry["y1"], crop)
    full_context = f"{context}. {REPAIR_NOTE}" if context else REPAIR_NOTE
    rec, usage = extract_fn(crop, rec_entry["index"], full_context)
    new = rec.model_dump() if hasattr(rec, "model_dump") else rec
    repaired = []
    for n in FIELD_NAMES:
        if not _v(fields, n) and _v(new, n):
            fields[n] = dict(new[n])
            repaired.append(n)
    return repaired, usage


# ─── Page orchestration ──────────────────────────────────────────────────────

def _check_sequence(record_nos):
    if not all(rn.isdigit() for rn in record_nos):
        return "non-numeric"
    nums = [int(rn) for rn in record_nos]
    if len(set(nums)) != len(nums):
        return "duplicate"
    if any(b - a != 1 for a, b in zip(nums, nums[1:])):
        return "gap"
    return "ok"


def _atomic_write(path, payload):
    tmp = path + ".wtmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def reconcile_page(stem, model, segments_dir=SEGMENTS_DIR,
                   extractions_dir=EXTRACTIONS_DIR, out_base=RECONCILED_DIR,
                   extract_fn=None, force=False, context="", tmp_dir=None):
    """Reconcile one page's strip extractions into patient records."""
    man_path = os.path.join(segments_dir, stem, f"{stem}.json")
    with open(man_path) as f:
        manifest = json.load(f)
    if manifest.get("status") != "ok":
        return {"stem": stem, "refused": manifest.get("status", "unknown")}
    ex_path = os.path.join(extractions_dir, model, f"{stem}.json")
    if not os.path.isfile(ex_path):
        return {"stem": stem, "refused": "no-extraction"}

    out_dir = os.path.join(out_base, model)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{stem}.json")
    if os.path.isfile(out_path) and not force:
        with open(out_path) as f:
            page = json.load(f)
        page["skipped"] = True
        return page

    with open(ex_path) as f:
        extraction = json.load(f)
    records = extraction["records"]
    man_by_index = {r["index"]: r for r in manifest["records"]}
    header_bottom = manifest["header_band"][1]
    full_png = os.path.join(segments_dir, stem, f"{stem}_full.png")
    tmp_dir = tmp_dir or os.path.join(out_dir, ".repair")
    os.makedirs(tmp_dir, exist_ok=True)

    repair_usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
    kinds = classify_strips(records)

    # boundary repair pass (primaries only)
    repaired_by_key = {}
    for key, kind, _reason in kinds:
        if kind != "primary":
            continue
        fields = records[str(key)]["fields"]
        if not has_clip_signature(fields):
            continue
        if extract_fn is None:
            repaired_by_key[key] = None          # flag later
            continue
        repaired, usage = repair_fields(
            fields, man_by_index[key], header_bottom, full_png, tmp_dir,
            extract_fn, context=context)
        repaired_by_key[key] = repaired
        repair_usage["calls"] += 1
        repair_usage["input_tokens"] += usage.get("input_tokens", 0)
        repair_usage["output_tokens"] += usage.get("output_tokens", 0)

    # group primaries with their continuations, in order
    patients = []
    group = []
    groups = []
    for key, kind, reason in kinds:
        if kind == "empty":
            continue
        if kind == "primary":
            if group:
                groups.append(group)
            group = [(key, reason)]
        else:
            group.append((key, None))
    if group:
        groups.append(group)

    for seq, grp in enumerate(groups, start=1):
        strips = [(k, records[str(k)]["fields"]) for k, _ in grp]
        merged = merge_patient(strips)
        primary_key, primary_reason = grp[0]
        rep = repaired_by_key.get(primary_key, [])
        if rep is None:
            merged["review"] = True
            merged["warnings"].append(
                {"reason": "clipped-no-repair", "strip": primary_key})
            rep = []
        if primary_reason:
            merged["review"] = True
            merged["warnings"].append(
                {"reason": primary_reason, "strip": primary_key})
        merged["repaired_fields"] = rep
        merged["seq"] = seq
        merged["record_no"] = merged["fields"]["record_no"]["value"]
        patients.append(merged)

    seq_status = _check_sequence(
        [p["record_no"] for p in patients if p["record_no"]])
    repair_usage["est_cost_usd"] = round(estimate_cost(
        model, repair_usage["input_tokens"], repair_usage["output_tokens"]), 6)

    page = {"stem": stem, "model": model,
            "reconciler_version": RECONCILER_VERSION,
            "patients": patients,
            "page_checks": {"record_no_sequence": seq_status, "warnings": []},
            "repair_usage": repair_usage}
    _atomic_write(out_path, page)
    return page
