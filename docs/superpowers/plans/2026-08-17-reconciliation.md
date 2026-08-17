# Reconciliation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn per-strip extractions into per-patient records: continuation merge, validator-based conflict resolution, boundary repair via targeted re-reads, page checks — with full provenance and review flags, never silent guessing.

**Architecture:** Pure-logic library `reconciliation.py` (classification, merge, validators, checks) with an injected `extract_fn` for boundary-repair re-reads (stubbed in tests, wired to Gemini in the CLI `1e_reconcile.py`). Consumes `_segments/` manifests + `_extractions/<model>/`; produces `_reconciled/<model>/<stem>.json`.

**Tech Stack:** Python 3, cv2/numpy (crop building only), pytest. Reuses `extraction.FIELD_NAMES`/`estimate_cost`/`extract_strip`, `segmentation.SUBROWS_PER_RECORD`.

**Spec:** `docs/superpowers/specs/2026-08-17-reconciliation-design.md`

## Global Constraints

- Repair fills ONLY empty fields; existing values are never overwritten. Ambiguity → `review: true` with a recorded reason, never a silent fix.
- Merged patients: `treatment_line1..3` → `treatments` list, `tab_no` → `tab_nos` list (strip order, empties dropped); all other fields keep `{value, confidence}` shape.
- Output path `_reconciled/<model>/<stem>.json`, atomic write, resume unless `--force`. `RECONCILER_VERSION = "1"`. Default model `gemini-3.7-flash`.
- Clip signature: ≥ `CLIP_MIN_EMPTY = 5` empties among `CLIP_FIELDS = [patient_name, sex, first_time_odh, hh_owns_phone, hh_owns_toilet, result_pn, diagnosis, full_cost]`, on non-continuation strips only.
- Repair crop: full sub-row slack each side (`(y1-y0)//SUBROWS_PER_RECORD`), clamped to `[header_bottom, H]`; header band stitched; red lines at the EXPANDED bounds.
- Offline tests only (stub `extract_fn`); one credential-gated live task at the end. Suite baseline: 57 offline (`python -m pytest --ignore=tests/test_live_extraction.py`). Never run the live extraction test; never print/commit credentials; never commit gitignored artifacts.
- Live acceptance (human ground truth 2026-08-17): p081 → exactly 3 patients, 304's treatments ≥ 4 entries; 314 repaired to name containing "Okwir", sex M, first_time_odh Y, hh_owns_phone Y, hh_owns_toilet Y; 304's first_voucher_use/group_appt stay empty; record_no continuity 304..314 ok.

---

### Task 1: Core module — helpers, classification, merge, validators

**Files:**
- Create: `reconciliation.py`
- Test: `tests/test_reconcile_merge.py`

**Interfaces:**
- Produces: `RECONCILER_VERSION`, `CLIP_MIN_EMPTY`, `CLIP_FIELDS`, `CONTENT_FIELDS`, `IDENTITY_FIELDS`, `TREATMENT_FIELDS`, `VALIDATORS`;
  `_v(fields, name) -> str` (stripped value or ""); `_norm(s) -> str`;
  `is_continuation(primary_fields, cur_fields) -> bool`;
  `continuation_conflict(primary_fields, cur_fields) -> str | None` (reason like "ambiguous-continuation" when record_no matches but names differ non-empty);
  `classify_strips(records: dict[str, dict]) -> list[tuple[int, str, str|None]]` — per strip key (int order): kind `"primary"|"continuation"|"empty"`, review-reason or None;
  `merge_patient(strips: list[tuple[int, dict]]) -> dict` with keys `fields` (incl. `treatments`, `tab_nos` lists), `source_strips`, `merged_from`, `filled_from_continuation`, `resolved_conflicts`, `warnings`, `review`.

- [ ] **Step 1: Write the failing tests** (`tests/test_reconcile_merge.py`)

```python
import pytest

from extraction import FIELD_NAMES
from reconciliation import (classify_strips, is_continuation, merge_patient,
                            continuation_conflict, VALIDATORS)


def F(**over):
    """All-empty extraction fields dict with overrides: F(patient_name='X')."""
    f = {n: {"value": "", "confidence": "high"} for n in FIELD_NAMES}
    for k, v in over.items():
        f[k] = {"value": v, "confidence": "high"}
    return f


PRIMARY = dict(record_no="304", patient_name="Aciro Rose", sex="M",
               age_yrs="24", village="Katuru", day="15", month="3",
               diagnosis="PID", treatment_line1="T. O cef 2g stat",
               treatment_line2="T. O centa stat", tab_no="6",
               full_cost="27500", balance="22800")


def test_is_continuation_true_for_treatment_overflow():
    prim = F(**PRIMARY)
    cont = F(treatment_line1="T. Nitro 100mg bd/7",
             treatment_line2="T. Ibuprofen 400mg", tab_no="10")
    assert is_continuation(prim, cont)


def test_is_continuation_false_on_new_identity():
    prim = F(**PRIMARY)
    other = F(record_no="305", patient_name="Namono Grace",
              treatment_line1="T. Cipro")
    assert not is_continuation(prim, other)


def test_continuation_conflict_same_recno_different_name():
    prim = F(**PRIMARY)
    odd = F(record_no="304", patient_name="Someone Else",
            treatment_line1="T. X")
    assert continuation_conflict(prim, odd) == "ambiguous-continuation"


def test_classify_strips_primary_cont_empty():
    records = {
        "1": {"fields": F(**PRIMARY)},
        "2": {"fields": F(treatment_line1="T. Nitro", tab_no="10")},
        "3": {"fields": F(record_no="305", patient_name="Namono Grace",
                          diagnosis="PUD", full_cost="28000")},
        "4": {"fields": F()},                       # blank block
    }
    kinds = classify_strips(records)
    assert [(k, kind) for k, kind, _ in kinds] == [
        (1, "primary"), (2, "continuation"), (3, "primary"), (4, "empty")]


def test_merge_concats_treatments_and_fills_empties():
    prim = F(**PRIMARY)
    cont = F(treatment_line1="T. Nitro 100mg", treatment_line3="T. Pcm",
             tab_no="10", cost_after_discount="20000")
    m = merge_patient([(1, prim), (2, cont)])
    tvals = [t["value"] for t in m["fields"]["treatments"]]
    assert tvals == ["T. O cef 2g stat", "T. O centa stat",
                     "T. Nitro 100mg", "T. Pcm"]
    assert [t["value"] for t in m["fields"]["tab_nos"]] == ["6", "10"]
    assert m["fields"]["cost_after_discount"]["value"] == "20000"
    assert "cost_after_discount" in m["filled_from_continuation"]
    assert m["source_strips"] == [1, 2] and m["merged_from"] == [2]
    assert m["review"] is False


def test_merge_resolves_invalid_day_by_validator():
    prim = F(**{**PRIMARY, "day": "86"})            # ink blot
    cont = F(day="16", treatment_line1="T. X")
    m = merge_patient([(1, prim), (2, cont)])
    assert m["fields"]["day"]["value"] == "16"
    assert m["resolved_conflicts"][0]["field"] == "day"
    assert m["resolved_conflicts"][0]["rejected"] == "86"
    assert m["review"] is False


def test_merge_flags_unresolvable_conflict():
    prim = F(**PRIMARY)                              # full_cost 27500
    cont = F(full_cost="99000", treatment_line1="T. X")   # both valid, differ
    m = merge_patient([(1, prim), (2, cont)])
    assert m["fields"]["full_cost"]["value"] == "27500"   # primary kept
    assert m["review"] is True
    assert m["warnings"][0]["field"] == "full_cost"


def test_validators_shape():
    assert VALIDATORS["day"]("16") and not VALIDATORS["day"]("86")
    assert VALIDATORS["month"]("3") and not VALIDATORS["month"]("48")
    assert VALIDATORS["sex"]("F") and not VALIDATORS["sex"]("X")
    assert VALIDATORS["full_cost"]("27500") and not VALIDATORS["full_cost"]("27k")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_reconcile_merge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'reconciliation'`

- [ ] **Step 3: Create `reconciliation.py`**

```python
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
```

(`RECONCILED_DIR` does not exist yet in config — Task 3 adds it. For THIS task only, so the module imports cleanly, add it to `config.py` now: `RECONCILED_DIR = os.path.join(PROJECT_ROOT, "_reconciled")` + `os.makedirs(RECONCILED_DIR, exist_ok=True)` below EXTRACTIONS_DIR, and append `_reconciled/` to `.gitignore`. Task 3's brief mentions the same lines — treat as already done then.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_reconcile_merge.py -v`
Expected: 8 passed. Full offline suite: 65.

- [ ] **Step 5: Commit**

```bash
git add reconciliation.py config.py .gitignore tests/test_reconcile_merge.py
git commit -m "feat: reconciliation core — classification, merge, validators"
```

---

### Task 2: Clip signature, repair crop, repair merge

**Files:**
- Modify: `reconciliation.py` (append)
- Test: `tests/test_reconcile_repair.py`

**Interfaces:**
- Produces: `has_clip_signature(fields) -> bool`;
  `build_repair_crop(full_page_path, header_bottom, y0, y1, out_path) -> tuple[str, int, int]` (path, applied top slack, applied bottom slack);
  `repair_fields(fields, rec_entry, header_bottom, full_png, tmp_dir, extract_fn, context="") -> tuple[list[str], dict]` — fills ONLY empty fields from the re-read, returns (repaired field names, usage dict). `extract_fn(image_path, record_index, context) -> (obj_with_model_dump_or_dict, usage)`.

- [ ] **Step 1: Write the failing tests** (`tests/test_reconcile_repair.py`)

```python
import os

import cv2
import numpy as np

from extraction import FIELD_NAMES
from reconciliation import (has_clip_signature, build_repair_crop,
                            repair_fields, REPAIR_NOTE)
from tests.test_reconcile_merge import F if False else None  # placeholder


def F(**over):
    f = {n: {"value": "", "confidence": "high"} for n in FIELD_NAMES}
    for k, v in over.items():
        f[k] = {"value": v, "confidence": "high"}
    return f


def test_clip_signature():
    clipped = F(village="Bulaga", village_code="2", day="16", month="3")
    assert has_clip_signature(clipped)          # 8/8 CLIP_FIELDS empty
    normal = F(patient_name="X", sex="M", first_time_odh="Y",
               hh_owns_phone="Y", diagnosis="Malaria", full_cost="3200")
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
```

(Remove the placeholder import line — define the local `F` helper as shown.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_reconcile_repair.py -v`
Expected: FAIL with ImportError (`has_clip_signature` not defined)

- [ ] **Step 3: Append to `reconciliation.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_reconcile_repair.py -v`
Expected: 4 passed. Full offline suite: 69.

- [ ] **Step 5: Commit**

```bash
git add reconciliation.py tests/test_reconcile_repair.py
git commit -m "feat: clip signature, repair crop, fill-only-empty repair merge"
```

---

### Task 3: Page orchestration `reconcile_page`

**Files:**
- Modify: `reconciliation.py` (append)
- Test: `tests/test_reconcile_page.py`

**Interfaces:**
- Produces: `reconcile_page(stem, model, segments_dir=SEGMENTS_DIR, extractions_dir=EXTRACTIONS_DIR, out_base=RECONCILED_DIR, extract_fn=None, force=False, context="", tmp_dir=None) -> dict` — output written to `<out_base>/<model>/<stem>.json` (atomic) with keys `stem, model, reconciler_version, patients, page_checks, repair_usage` per the spec schema; refusal dicts `{"stem", "refused": reason}` for non-ok manifest / missing extraction; resume unless force (transient `"skipped": True` on resume); `extract_fn=None` disables repair (clip signature → patient `review` with reason `clipped-no-repair`); `_check_sequence(record_nos: list[str]) -> str` ("ok"|"gap"|"duplicate"|"non-numeric").

- [ ] **Step 1: Write the failing tests** (`tests/test_reconcile_page.py`)

```python
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


def test_reconcile_page_clip_without_repair_flags_review(tmp_path):
    clipped = F(village="Bulaga", village_code="2", day="16", month="3")
    seg, exd = _page(tmp_path, extraction=[clipped])
    r = reconcile_page("reg_p1", "m", segments_dir=seg, extractions_dir=exd,
                       out_base=str(tmp_path / "rec"), extract_fn=None)
    p = r["patients"][0]
    assert p["review"] is True
    assert "clipped-no-repair" in p["warnings_text"] \
        if "warnings_text" in p else any(
            "clipped-no-repair" in str(w) for w in p["warnings"])


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_reconcile_page.py -v`
Expected: FAIL with ImportError (`reconcile_page` not defined)

- [ ] **Step 3: Append to `reconciliation.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_reconcile_page.py -v` → 5 passed. Full offline suite: 74.

- [ ] **Step 5: Commit**

```bash
git add reconciliation.py tests/test_reconcile_page.py
git commit -m "feat: reconcile_page orchestration with repair pass and page checks"
```

---

### Task 4: CLI `1e_reconcile.py`

**Files:**
- Create: `1e_reconcile.py`
- Test: `tests/test_reconcile_cli.py`

**Interfaces:**
- Produces: CLI `python 1e_reconcile.py [stems...] [--all] [--model M] [--no-repair] [--force] [--center X] [--year Y] [--out D] [--segments-dir D] [--extractions-dir D]`; default model `gemini-3.7-flash`; `main(argv=None)`; builds the Gemini client lazily (only when the first repair re-read is actually needed) via `_make_extract_fn(model, center, year)` returning a closure over `extraction.make_client` + a repair call using `extraction.extract_strip`; `--no-repair` passes `extract_fn=None`. Prints per page `patients / merged / repaired / review` counts and a summary with repair spend. Exit codes: 0 clean, 2 if any patient `review`, 1 on errors.

- [ ] **Step 1: Write the failing test** (`tests/test_reconcile_cli.py`) — offline `--no-repair` path over the fabricated fixtures from Task 3's `_page` helper (copy the helper), asserting: exit code 2 when a clipped page produces a review patient, "review" appears in stdout, output file exists; and a clean two-patient page exits 0.

```python
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
             hh_owns_toilet="Y", result_pn="P"),
           F(treatment_line1="T2", tab_no="4")])
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
```

- [ ] **Step 2: Run tests to verify they fail** — module not found.

- [ ] **Step 3: Write `1e_reconcile.py`**

```python
#!/usr/bin/env python3
"""
1e_reconcile.py — Per-strip extractions → per-patient records.

Usage:
    python 1e_reconcile.py --all                       # gemini-3.7-flash
    python 1e_reconcile.py reg_p1 reg_p2 --no-repair
    python 1e_reconcile.py --all --force --center "Kameno" --year 2026

Boundary-repair re-reads call Gemini (credentials from .env) and are built
lazily — pages needing no repair never touch the network. --no-repair
disables re-reads entirely (clip signatures become review flags).
Output: _reconciled/<model>/<stem>.json. Exit 0 clean, 2 review queue
non-empty, 1 errors.
"""

import argparse
import glob as globmod
import os
import sys

from config import SEGMENTS_DIR, EXTRACTIONS_DIR, RECONCILED_DIR
from reconciliation import reconcile_page


def discover_stems(extractions_dir, model):
    d = os.path.join(extractions_dir, model)
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in globmod.glob(os.path.join(d, "*.json"))
                  if not os.path.basename(p).startswith("compare_"))


def _make_extract_fn(model, context_unused):
    """Lazy Gemini-backed repair re-reader; client built on first call."""
    state = {"client": None}

    def extract_fn(image_path, record_index, context):
        if state["client"] is None:
            from extraction import make_client
            state["client"] = make_client()
        from extraction import extract_strip
        return extract_strip(state["client"], model, image_path,
                             record_index, context=context)

    return extract_fn


def main(argv=None):
    p = argparse.ArgumentParser(description="Reconcile strips into patients.")
    p.add_argument("stems", nargs="*")
    p.add_argument("--all", action="store_true")
    p.add_argument("--model", default="gemini-3.7-flash")
    p.add_argument("--no-repair", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--center", default="")
    p.add_argument("--year", default="")
    p.add_argument("--segments-dir", default=SEGMENTS_DIR)
    p.add_argument("--extractions-dir", default=EXTRACTIONS_DIR)
    p.add_argument("--out", default=RECONCILED_DIR)
    args = p.parse_args(argv)

    stems = list(args.stems)
    if args.all:
        stems += [s for s in discover_stems(args.extractions_dir, args.model)
                  if s not in stems]
    if not stems:
        p.error("no stems given (pass stems or --all)")

    context = ", ".join(x for x in
                        ([f"Center: {args.center}"] if args.center else []) +
                        ([f"Year: {args.year}"] if args.year else []))
    extract_fn = None if args.no_repair else _make_extract_fn(args.model,
                                                              context)

    total_patients = reviews = errors = repairs = 0
    rcost = 0.0
    for stem in stems:
        try:
            r = reconcile_page(stem, args.model,
                               segments_dir=args.segments_dir,
                               extractions_dir=args.extractions_dir,
                               out_base=args.out, extract_fn=extract_fn,
                               force=args.force, context=context)
        except Exception as err:
            errors += 1
            print(f"  {stem}: ERROR {err}")
            continue
        if "refused" in r:
            print(f"  {stem}: refused ({r['refused']})")
            continue
        if r.get("skipped"):
            print(f"  {stem}: skipped (existing; use --force)")
            continue
        n_rev = sum(1 for pt in r["patients"] if pt["review"])
        n_merged = sum(1 for pt in r["patients"] if pt["merged_from"])
        reviews += n_rev
        repairs += r["repair_usage"]["calls"]
        rcost += r["repair_usage"].get("est_cost_usd", 0.0)
        total_patients += len(r["patients"])
        print(f"  {stem}: {len(r['patients'])} patients, {n_merged} merged, "
              f"{r['repair_usage']['calls']} repaired, {n_rev} review, "
              f"seq={r['page_checks']['record_no_sequence']}")

    print(f"\n[{args.model}] {total_patients} patients, {repairs} repair "
          f"re-reads (est ${rcost:.4f}), {reviews} review, {errors} errors")
    if errors:
        return 1
    return 2 if reviews else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests** — `python -m pytest tests/test_reconcile_cli.py -v` → 2 passed. Full offline suite: 76.

- [ ] **Step 5: Commit**

```bash
git add 1e_reconcile.py tests/test_reconcile_cli.py
git commit -m "feat: 1e_reconcile CLI with lazy repair client and review exit code"
```

---

### Task 5: Live validation against human ground truth

**Files:**
- Modify (results only): `docs/superpowers/specs/2026-08-17-reconciliation-design.md`
- Commit also: `docs/superpowers/plans/2026-08-17-reconciliation.md` (this plan, not yet tracked)

- [ ] **Step 1:** Run `python 1e_reconcile.py 20260319_053700_KAM_Stlhb_p1 20260319_053700_KAM_Stlhb_p2 20260319_053700_KAM_Stlhb_p3 --center "Kameno" --year 2026` (repairs enabled; expect 1-3 re-reads, ≤$0.05).

- [ ] **Step 2: Verify the acceptance list** (spec §Acceptance) against `_reconciled/gemini-3.7-flash/*.json`:
p081 = exactly 3 patients (304/305/306), 304 treatments ≥ 4; p083 patient 314 has repaired_fields covering the human-validated values (name contains "Okwir", sex M, first_time_odh Y, hh_owns_phone Y, hh_owns_toilet Y) — verify each against the repair crop image with Read (you have vision); 304 first_voucher_use/group_appt still empty; record continuity 304..314 `ok` per page; ink-blot date field resolved or review-flagged, never invalid-and-silent.

- [ ] **Step 3:** If a repair re-read returns values contradicting the human ground truth, record the discrepancy verbatim in the report and DO NOT tune prompts/thresholds beyond one retry — report DONE_WITH_CONCERNS.

- [ ] **Step 4:** Append `## Live validation (2026-08-17)` to the spec: per-page patient counts, repairs performed + spend, acceptance-list outcomes, discrepancies.

- [ ] **Step 5: Commit** the spec update + this plan file. Verify `git status` stages nothing gitignored/PHI-bearing.

```bash
git add docs/superpowers/specs/2026-08-17-reconciliation-design.md docs/superpowers/plans/2026-08-17-reconciliation.md
git commit -m "test: reconciliation live validation against human ground truth"
```

---

## Self-review notes

- Spec coverage: classification/merge/validators (T1), clip signature + repair (T2), orchestration/checks/output schema (T3), CLI with lazy client + exit codes (T4), live acceptance (T5). Cross-page identity and FHIR are spec'd out of scope.
- Type consistency: `classify_strips` returns int keys used by T3's grouping and `records[str(key)]` lookups; `merge_patient` input `[(int, fields)]` matches T3; `repair_fields` mutates fields in place and returns names+usage, consumed in T3's repair pass; `_page` fixture manifest/extraction shapes match phases 1-2 output contracts (header_band, records index/y0/y1, fields/usage).
- Judgment calls: repairs run on primaries only (continuations with clip signatures are rare and would merge anyway); `time_hh` validator allows 0-12 (register uses 12h clock with AM/PM); repair crop's red lines sit at the expanded bounds (search area), unlike normal strips (true bounds) — REPAIR_NOTE explains the difference to the model.
