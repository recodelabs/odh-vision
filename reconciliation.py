"""
reconciliation.py — Per-strip extractions → per-patient records.

Phase 3 of odh-vision: continuation merge, validator-based conflict
resolution, boundary repair, page checks. Pure logic except the repair
re-read, which is injected (extract_fn) so tests never touch a network.
See docs/superpowers/specs/2026-08-17-reconciliation-design.md.
"""

import difflib
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
# A real register record virtually always has "sex" checked (and, in
# practice, the other three checkbox-cluster fields too). If ALL of these
# are empty regardless of the rest of the strip, that is itself a clip
# signature even when CLIP_MIN_EMPTY isn't reached (e.g. patient_name /
# result_pn / diagnosis / full_cost were captured lower on the strip,
# below the clip line) — see live validation 2026-08-17, record 314.
CHECKBOX_CLUSTER_FIELDS = ["sex", "first_time_odh", "hh_owns_phone",
                           "hh_owns_toilet"]
CONTENT_FIELDS = ["treatment_line1", "treatment_line2", "treatment_line3",
                  "tab_no", "full_cost", "balance", "cost_after_discount",
                  "diagnosis"]
IDENTITY_FIELDS = ["sex", "age_yrs", "village"]
TREATMENT_FIELDS = ("treatment_line1", "treatment_line2", "treatment_line3")

# Repairs may only fill fields that are structurally safe to re-derive from
# a wider crop of the SAME record: identity/header fields and the
# checkbox/name cluster. Treatment lines, tab_no, balance and
# cost_after_discount are explicitly excluded — live validation 2026-08-17
# found repair re-reads fabricating/duplicating treatment-line content and
# leaking values from neighboring strips into these fields.
REPAIRABLE_FIELDS = CLIP_FIELDS + ["record_no", "village", "village_code",
                                   "age_yrs", "day", "month", "time_hh",
                                   "time_mm", "am_pm"]

# Margin record_no digits are proven unreliable (clipped, misread
# inconsistently across a single patient's own strips — live validation
# 2026-08-17). Name similarity is the primary continuation signal.
NAME_MATCH_THRESHOLD = 0.75

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

def _name_similarity(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def is_continuation(primary_fields, cur):
    """True if *cur* looks like an overflow block of *primary_fields*.

    Name is the primary identity signal when both sides have one: a fuzzy
    match (ratio >= NAME_MATCH_THRESHOLD) is enough, and in that case a
    differing record_no no longer blocks classification — margin
    record_no digits are proven unreliable, misread inconsistently even
    across a single patient's own strips. When either side lacks a name,
    fall back to the record_no equality check.
    """
    n_p, n_c = _norm(_v(primary_fields, "patient_name")), \
        _norm(_v(cur, "patient_name"))
    if n_p and n_c:
        if _name_similarity(n_p, n_c) < NAME_MATCH_THRESHOLD:
            return False
    else:
        rn_p, rn_c = _v(primary_fields, "record_no"), _v(cur, "record_no")
        if rn_c and rn_p and rn_c != rn_p:
            return False
    for fld in IDENTITY_FIELDS:
        a, b = _norm(_v(primary_fields, fld)), _norm(_v(cur, fld))
        if not a or not b:
            continue
        if fld == "village":
            # Village names are hand-lettered free text, prone to the same
            # kind of near-miss OCR spelling drift as patient names (e.g.
            # "Bulago" vs "Bulaga") — fuzzy-match rather than requiring an
            # exact string match.
            if _name_similarity(a, b) < NAME_MATCH_THRESHOLD:
                return False
        elif a != b:
            return False
    return any(_v(cur, f) for f in CONTENT_FIELDS)


def continuation_conflict(primary_fields, cur):
    """Reason string when *cur* half-matches the primary (needs review).

    Applies only when record_no matches AND the names are dissimilar
    (ratio < NAME_MATCH_THRESHOLD) — a same-record_no strip whose name
    also matches (or fuzzy-matches) is a normal continuation, not a
    conflict.
    """
    rn_p, rn_c = _v(primary_fields, "record_no"), _v(cur, "record_no")
    n_p, n_c = _norm(_v(primary_fields, "patient_name")), \
        _norm(_v(cur, "patient_name"))
    if rn_c and rn_p and rn_c == rn_p and n_c and n_p:
        if _name_similarity(n_p, n_c) < NAME_MATCH_THRESHOLD:
            return "ambiguous-continuation"
    return None


def classify_strips(records):
    """[(key, "primary"|"continuation"|"empty", reason|None)].

    For continuations, *reason* is normally None, but may be
    "recno-mismatch-name-match" when the strip was classified as a
    continuation on name similarity despite a differing record_no — a
    warning-worthy fact for the caller to surface, not a review-forcing
    one (record_no is known-unreliable; the name match is trusted).
    """
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
                cont_reason = None
                rn_p = _v(current_primary, "record_no")
                rn_c = _v(fields, "record_no")
                if rn_p and rn_c and rn_p != rn_c:
                    cont_reason = "recno-mismatch-name-match"
                out.append((int(key), "continuation", cont_reason))
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
            if n == "record_no":
                # Known-unreliable field (margins clipped/misread across a
                # patient's own strips); mismatch is surfaced separately as
                # a "recno-mismatch-name-match" warning by reconcile_page,
                # not as a blocking merge conflict here. Keep the primary's
                # reading silently.
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
    """True when so many core fields are empty the strip was likely clipped.

    Two independent triggers:
    - the whole checkbox cluster (sex, first_time_odh, hh_owns_phone,
      hh_owns_toilet) is empty — a real register record virtually always
      has at least "sex" checked, so this is itself a clip signature
      regardless of how many other CLIP_FIELDS were captured elsewhere on
      the strip (e.g. below the clip line);
    - OR the pre-existing >= CLIP_MIN_EMPTY rule over all CLIP_FIELDS.
    """
    if all(not _v(fields, f) for f in CHECKBOX_CLUSTER_FIELDS):
        return True
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
        if n not in REPAIRABLE_FIELDS:
            continue
        if not _v(fields, n) and _v(new, n):
            fields[n] = dict(new[n])
            repaired.append(n)
    return repaired, usage


# ─── Possible-split detection ────────────────────────────────────────────────

def _edit_distance(a, b):
    """Levenshtein distance between two strings."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                        prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def _recno_similar(a, b):
    """True if two record_no strings are plausibly the same misread digits:
    one is a substring of the other (e.g. clipped "31"/"314"), or a
    dropped/extra digit puts them within Levenshtein distance 1 (e.g.
    "307"/"3067"). Equal-length strings that merely differ by one digit
    (e.g. "304"/"305") are NOT flagged here — that's the ordinary shape
    of two genuinely different, sequential patients, not a misread; the
    OCR failure mode this catches is a dropped/inserted/clipped digit,
    which always changes the string length.
    """
    if not a or not b:
        return False
    if a == b:
        return True
    if len(a) == len(b):
        return False
    if a in b or b in a:
        return True
    return _edit_distance(a, b) <= 1


def _patient_content(fields):
    """CONTENT_FIELDS-equivalent check on an already-merged patient's
    fields dict, where treatment_line1..3/tab_no have been consolidated
    into the "treatments"/"tab_nos" lists by merge_patient."""
    if fields.get("treatments") or fields.get("tab_nos"):
        return True
    return any(_v(fields, f) for f in
              ("full_cost", "balance", "cost_after_discount", "diagnosis"))


def possible_split(patient_a, patient_b, pre_repair_b=None):
    """True if two ADJACENT already-merged patients on a page might
    actually be one patient a classification bug split in two. Never
    auto-merged — this only flags for human review (never-guess).

    Two independent triggers:
    - their record_no values are non-empty and plausibly the same misread
      digits (_recno_similar: substring or edit-distance <= 1); OR
    - patient_b is built from a single strip whose content is
      continuation-shaped (>=1 CONTENT_FIELDS-equivalent value present)
      AND the whole checkbox cluster is empty AND at most 2 of
      CLIP_FIELDS are non-empty — the same clip-signature shape that
      would mark it a boundary-clipped fragment rather than a genuine
      new patient, just not recognized as a continuation of patient_a.

    *pre_repair_b*, when given, is a {"checkbox_empty": bool,
    "clip_populated": int} snapshot of patient_b's single strip taken
    BEFORE the boundary-repair pass ran. Boundary repair can fill exactly
    the checkbox-cluster/CLIP_FIELDS this shape check inspects (both are
    in REPAIRABLE_FIELDS), which would silently mask a genuine split by
    making a repaired fragment look complete. When provided, the shape
    check uses the pre-repair snapshot instead of patient_b's (possibly
    repaired) live fields, so repair can never suppress this flag.
    """
    rn_a = _v(patient_a["fields"], "record_no")
    rn_b = _v(patient_b["fields"], "record_no")
    if rn_a and rn_b and _recno_similar(rn_a, rn_b):
        return True
    if len(patient_b["source_strips"]) == 1:
        fb = patient_b["fields"]
        if pre_repair_b is not None:
            checkbox_empty = pre_repair_b["checkbox_empty"]
            clip_populated = pre_repair_b["clip_populated"]
        else:
            checkbox_empty = all(not _v(fb, f) for f in CHECKBOX_CLUSTER_FIELDS)
            clip_populated = sum(1 for f in CLIP_FIELDS if _v(fb, f))
        if _patient_content(fb) and checkbox_empty and clip_populated <= 2:
            return True
    return False


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

    # Pre-repair snapshot of every primary's clip/fragment shape — the
    # boundary-repair pass below fills exactly the checkbox-cluster/
    # CLIP_FIELDS that possible_split's branch (b) inspects (both are in
    # REPAIRABLE_FIELDS), which would silently mask a genuine split.
    # Snapshotting BEFORE repair mutates fields in place lets the
    # possible-split scan evaluate the PRE-repair shape.
    pre_repair_snapshot = {}
    for key, kind, _reason in kinds:
        if kind != "primary":
            continue
        fields = records[str(key)]["fields"]
        pre_repair_snapshot[key] = {
            "checkbox_empty": all(not _v(fields, f)
                                  for f in CHECKBOX_CLUSTER_FIELDS),
            "clip_populated": sum(1 for f in CLIP_FIELDS if _v(fields, f)),
        }

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
            group.append((key, reason))
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
        # Continuation reasons (e.g. "recno-mismatch-name-match") are
        # informational: surfaced in warnings but never force review on
        # their own — record_no is known-unreliable, and the strip was
        # already trusted as a continuation on name similarity.
        for cont_key, cont_reason in grp[1:]:
            if cont_reason:
                merged["warnings"].append(
                    {"reason": cont_reason, "strip": cont_key})
        merged["repaired_fields"] = rep
        merged["seq"] = seq
        merged["record_no"] = _v(merged["fields"], "record_no")
        patients.append(merged)

    # Possible-split scan: adjacent patients that might be one patient a
    # classification bug split in two. Never auto-merged (never-guess) —
    # flag both sides for the human review queue instead.
    for i in range(len(patients) - 1):
        a, b = patients[i], patients[i + 1]
        b_key = b["source_strips"][0] if len(b["source_strips"]) == 1 else None
        pre_b = pre_repair_snapshot.get(b_key) if b_key is not None else None
        if possible_split(a, b, pre_repair_b=pre_b):
            a["review"] = True
            b["review"] = True
            a["warnings"].append(
                {"reason": "possible-same-patient", "with_seq": b["seq"]})
            b["warnings"].append(
                {"reason": "possible-same-patient", "with_seq": a["seq"]})

    seq_status = _check_sequence(
        [p["record_no"] for p in patients if p["record_no"]])
    repair_usage["est_cost_usd"] = round(estimate_cost(
        model, repair_usage["input_tokens"], repair_usage["output_tokens"]), 6)

    # Missing-strip detection: a manifest record index absent from the
    # extraction means a patient was silently dropped rather than merged
    # or reviewed. Diff manifest indices against extraction record keys
    # and surface every gap as a page-level warning. An all-empty page
    # (manifest non-empty but zero patients built) is the same failure
    # mode by a different route and gets the same treatment.
    man_indices = {r["index"] for r in manifest["records"]}
    ex_indices = {int(k) for k in records}
    page_warnings = [{"reason": "missing-extraction", "strip": idx}
                     for idx in sorted(man_indices - ex_indices)]
    if manifest["records"] and not patients:
        page_warnings.append({"reason": "no-patients"})
    page_checks = {"record_no_sequence": seq_status,
                  "warnings": page_warnings, "review": bool(page_warnings)}

    page = {"stem": stem, "model": model,
            "reconciler_version": RECONCILER_VERSION,
            "patients": patients,
            "page_checks": page_checks,
            "repair_usage": repair_usage}
    _atomic_write(out_path, page)
    return page
