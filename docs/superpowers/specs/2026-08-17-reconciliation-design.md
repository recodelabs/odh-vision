# Reconciliation — Design Spec

**Date:** 2026-08-17
**Status:** Approved (design confirmed in session with mberg)
**Context:** Phase 3 of odh-vision. Phase 1 produces per-strip images + manifests (`_segments/`); phase 2 produces per-strip extractions (`_extractions/<model>/<stem>.json`). This phase turns per-STRIP extractions into per-PATIENT records, repairing the failure classes human validation confirmed on 2026-08-17.

## Goal

One correct record per patient, with three repairs applied and full provenance:
1. **Boundary repair** — recover fields lost when a record straddles a strip crop (validated: record 314 lost name/sex/4 checkboxes; the ink exists in the adjacent strip's zone on the full rectified page).
2. **Continuation merge** — fold treatment-overflow blocks into their patient (validated: page 081 has 5 blocks but 3 patients; the legacy CSV's 86 patients vs our 110 blocks).
3. **Consistency checks** — intra-patient conflict resolution with validators (validated: ink-blot date read "86" on a primary row while the continuation block held the true 16), plus record-number continuity and range checks.

Never silently guess: every repair/merge/resolution is recorded in provenance; anything ambiguous sets `review: true` with a reason instead of being "fixed".

## Architecture

```
_segments/<stem>/ (manifest + _full.png)   reconciliation.py (new library)
_extractions/<model>/<stem>.json      ──▶  1e_reconcile.py (CLI)  ──▶  _reconciled/<model>/<stem>.json
```

New files: `reconciliation.py`, `1e_reconcile.py`, `tests/test_reconcile_merge.py`, `tests/test_reconcile_checks.py`, `tests/test_reconcile_repair.py`, `tests/test_reconcile_cli.py`. `config.py` gains `RECONCILED_DIR`. `.gitignore` gains `_reconciled/`. No changes to phases 1–2 except reuse of `extraction.extract_strip`-adjacent machinery via a thin injected function.

## Pipeline per page (given model M, stem S)

1. Load manifest (`_segments/S/S.json`) and extraction (`_extractions/M/S.json`). Refuse pages whose manifest isn't `ok` or whose extraction is missing/has `strip_errors`-era gaps (missing record keys → `review` note, not silent skip).
2. **Classify strips in page order**: PRIMARY (starts a patient) vs CONTINUATION vs EMPTY (all fields blank → dropped with a note). Detection is conservative (below); ambiguity → treat as primary + `review: true`.
3. **Boundary-repair scan** (before merging): any strip whose extraction shows the *clip signature* gets one targeted re-read from the full rectified page with expanded bounds; only EMPTY fields are filled from the re-read (existing values are never overwritten), recorded in `repaired_fields`.
4. **Merge** continuations into their primary; resolve intra-patient conflicts with validators.
5. **Checks**: field range validators (day 1–31, month 1–12, numeric record_no/costs), per-page record-number continuity (expected +1 steps; gaps/duplicates → warnings), unresolved conflicts → `review: true`.
6. Write `_reconciled/M/S.json` atomically.

## Detection rules

**Continuation** (strip k relative to the patient built so far), all on normalized `.value`:
- `record_no` empty OR equal to the primary's; AND
- `patient_name` empty OR norm-equal to the primary's; AND
- identity fields `sex, age_yrs, village` each empty or equal; AND
- the strip has content in at least one of `treatment_line1..3, tab_no, full_cost, balance, cost_after_discount, diagnosis` (otherwise it's EMPTY).
- Conflict case (e.g. same record_no but a different non-empty name): classify PRIMARY and set `review: true` with reason `ambiguous-continuation`.

**Clip signature** (candidate for boundary repair): strip is not a detected continuation AND ≥ `CLIP_MIN_EMPTY = 5` of `CLIP_FIELDS = [patient_name, sex, first_time_odh, hh_owns_phone, hh_owns_toilet, result_pn, diagnosis, full_cost]` are empty.

## Boundary repair

- Crop from `_segments/S/S_full.png` (grayscale rectified page): rows `[max(header_bottom, y0 − subrow) : min(H, y1 + subrow)]` where `subrow = (y1 − y0) // SUBROWS_PER_RECORD` — a full sub-row of slack each side, because validated mis-snaps are sub-row-scale, beyond the 25px strip buffer.
- Assemble like a normal strip (header band stitched on top, BGR, red lines at the *expanded* crop bounds) and re-read with the standard extraction prompt plus a repair addendum: exactly one complete patient record lies between the red lines; fragments of adjacent records may appear at the extreme edges; transcribe the complete record.
- Re-read via an injected `extract_fn(image_path, record_index, context) -> (RecordExtraction, usage)` so tests stub it and the CLI wires it to a thin wrapper over `extraction.extract_strip` machinery (same model M, temperature 0). Repair usage/cost is accounted in the page output.
- Merge policy: fill only fields that were EMPTY in the original strip extraction; never overwrite; list filled names in `repaired_fields`. `--no-repair` disables re-reads (classification/merge still run; clip signature becomes a `review` reason instead).

## Merge & conflict policy

- **Treatments/tabs become lists** (a merged patient can exceed 3 lines — validated: Aciro Rose has 5): reconciled fields replace `treatment_line1..3` with `treatments: [Reading...]` and `tab_no` with `tab_nos: [Reading...]`, in strip order, empties dropped. All other fields keep the `{value, confidence}` shape.
- Scalar fields: primary's non-empty value wins; if primary empty, continuation's value fills it (recorded in `filled_from_continuation`); if both non-empty and different → run the field's validator: if exactly one value is valid (e.g. day "86" vs "16"), take the valid one and record in `resolved_conflicts`; otherwise keep the primary's and set `review: true` with the conflict recorded.
- Validators: day ∈ 1..31, month ∈ 1..12, record_no numeric, costs numeric-ish (digits, optional trailing ".0"), sex ∈ {M,F}, checkboxes ∈ {Y,N,""}. Fields without a validator resolve conservatively (keep primary + review on conflict).

## Output — `_reconciled/<model>/<stem>.json`

```json
{
  "stem": "..._p1", "model": "...", "reconciler_version": "1",
  "patients": [
    {
      "seq": 1, "record_no": "304",
      "fields": { "patient_name": {"value": "...", "confidence": "high"},
                   "treatments": [{"value": "...", "confidence": "high"}, ...],
                   "tab_nos": [...] , "...": {} },
      "source_strips": [1, 2], "merged_from": [2],
      "repaired_fields": [], "filled_from_continuation": ["balance"],
      "resolved_conflicts": [{"field": "day", "kept": "16", "rejected": "86", "how": "validator"}],
      "warnings": [], "review": false
    }
  ],
  "page_checks": {"record_no_sequence": "ok|gap|duplicate", "warnings": []},
  "repair_usage": {"calls": 0, "input_tokens": 0, "output_tokens": 0, "est_cost_usd": 0.0}
}
```

## CLI — `1e_reconcile.py`

```
python 1e_reconcile.py --all [--model gemini-3.7-flash] [--no-repair] [--force]
python 1e_reconcile.py <stem> [...]
```
Default model `gemini-3.7-flash` (the validated reader). Prints per page: patients, merges, repairs, reviews; summary with repair spend. Exit 0 clean, 2 if any patient has `review: true` (queue non-empty), 1 on errors. Resume: skips stems with existing output unless `--force`.

## Acceptance (live validation on p1–p3 against human ground truth of 2026-08-17)

- p081 yields exactly **3 patients** (304, 305, 306); 304's treatments list has ≥4 entries.
- Record **314 repaired**: patient_name contains "Okwir", sex = M, first_time_odh = Y, hh_owns_phone = Y, hh_owns_toilet = Y (human-validated values), each listed in `repaired_fields`.
- Record 304's `first_voucher_use`/`group_appt` remain **empty** (human-validated blanks — repair must not invent them).
- The p082 ink-blot date resolves to day 16 via validator (invalid "86"/"26" rejected) or is review-flagged — never silently kept invalid.
- Record-number continuity 304→314 reported `ok` across the three pages.

## Testing strategy

Pure-logic tests (classification, merge, validators, continuity) on fabricated extraction/manifest dicts; repair tested with a stubbed `extract_fn` (no network) plus a real-cv2 crop-geometry test on a synthetic page; CLI dry paths offline. One credential-gated live validation task performs the acceptance list above (repair re-reads ≈ 1–3 calls, ~$0.03).

## Out of scope

Cross-page patient identity (same person visiting twice), gazetteer snapping, FHIR emission, writing to the legacy workbook, batch-tier submission.

## Live validation (2026-08-17)

Ran `python 1e_reconcile.py 20260319_053700_KAM_Stlhb_p1 ..._p2 ..._p3 --center "Kameno" --year 2026` twice (initial run + one permitted retry with `--force`, to check reproducibility of a suspicious repair). Both runs exit 2 (reviews present, expected). Total spend across both runs: **$0.0378** (5 repair calls run 1 + retry combined: $0.0179 + $0.0199), under the $0.05 ceiling.

**Per-page patient counts:** p1 → 5 patients (307, 3067, 308, 308, 306); p2 → 4 patients (307, 307, 308, 309); p3 → 5 patients (310, 311, 312, 313, 31). `page_checks.record_no_sequence`: p1 `duplicate`, p2 `duplicate`, p3 `gap`.

**Repairs performed:** run 1 — p1: 2 calls (strips 2, 4); p2: 1 call (strip 2); p3: 0. Run 2 (retry) reproduced the same 3 repair triggers but with a larger fill set on p2/strip2 (LLM sampling variance at temperature 0, not a code change).

### Acceptance-list outcomes

1. **p081 = exactly 3 patients (304/305/306), 304 treatments ≥ 4 — FAIL (count/id), PASS (substance).** Reconciled output shows 5 patients, not 3. Visual inspection of `_segments/.../p1_full.png` confirms the physical page truly has only 3 patients — 304 "Aciro Rose" (2 strips, rows 11-13/21-23), 305 "Okello Sam" (2 strips, rows 31-33/41-43), 306 "Nabirye Christine" (1 strip) — matching human ground truth exactly. But the underlying v2-era-prompt strip extraction misread `record_no`/`patient_name` inconsistently across each patient's two strips ("Aciro Rose" → strip1 "Acire Rose"/307, strip2 "Adong Janet"/3067; "Okello Sam" → strip3 "Okelu Sam"/308, strip4 "Okello Sam"/308 flagged `ambiguous-continuation`). The continuation-detection heuristic requires matching `record_no`/name and could not merge these, so each strip became its own patient. Field-level content (sex, checkboxes, first_voucher_use/group_appt) is individually correct on the split fragments; treatment count across the two 304 fragments is 6 (≥4, satisfies the numeric floor) even though the true page has 5 distinct entries — a duplicated treatment line was fabricated in one fragment (see below).
2. **Record 314 repaired (name contains "Okwir", sex M, first_time_odh Y, hh_owns_phone Y, hh_owns_toilet Y) — FAIL, repair never triggered.** Verified via the CAUTION scenario exactly as described: `_segments/.../p3_rec5.png` shows the strip's top red line cutting through row 51, chopping off the `1st time ODH`/`Sex`/`HH owns phone`/`HH owns toilet` checkboxes (all visibly checked Y/M/Y/Y on the full page image, all read empty in the raw strip extraction). This is the textbook clip-signature case. But `has_clip_signature` counts only 4 of the 8 `CLIP_FIELDS` empty (`sex`, `first_time_odh`, `hh_owns_phone`, `hh_owns_toilet`) — `patient_name` ("Moses", partial), `result_pn` ("P"), `diagnosis` ("malaria"), and `full_cost` ("3500") were captured lower in the strip and count as non-empty — one field short of `CLIP_MIN_EMPTY = 5`. Repair was never called; `repaired_fields` is `[]`; all four target fields remain empty. Per task instructions this was not tuned (no threshold change made).
3. **Record 304's first_voucher_use/group_appt remain empty — PASS (in substance).** Both split fragments of the true 304 record (reconciled as `record_no 307` and `record_no 3067`) show `first_voucher_use` and `group_appt` empty, matching the human-validated blank boxes.
4. **p082 ink-blot date resolves to 16, never silently invalid — PASS.** In this extraction run, the raw v2-era strip extraction for both affected primaries (rows 11-13 "Auma/Aumo Betty" and rows 31-33 "Wandera Peter") already reads `day=16` (medium confidence) despite the ink blot being visibly ambiguous on the page image (could read "86"). No invalid day value ever entered the reconciled output; no repair or conflict resolution was needed for this field.
5. **Record-number continuity 304→314 "ok" across all three pages — FAIL, all three pages.** p1 `duplicate`, p2 `duplicate`, p3 `gap` — none report `ok`. Root cause for p1/p2 is the same record_no/name misread described in (1)/(above); for p3 the last record's `record_no` reads truncated as `"31"` instead of `"314"` (medium confidence), consistent with the same left-edge/No.-column legibility issue that also produced the row-51 clip signature.

### Additional discrepancy found (repair cross-contamination)

Not on the acceptance list, but found during image verification and reproduced on the `--force` retry: when the repair mechanism fires on a strip that is actually an orphaned continuation (mis-classified as `ambiguous-continuation`/standalone due to the record_no/name mismatch in finding 1), its expanded-bounds re-read pulls header-type field values (`last_care`, `age_yrs`, `tests`, `diagnosis`, `weight_kg`, `full_cost`, `result_pn`) from the **neighboring strip's block** rather than leaving them empty. Example: p1 strip 4 (second half of true patient 305, "Okello Sam") had `last_care`/`age_yrs`/`tests`/`diagnosis` repaired-in from row 33 — the tail of the *previous* strip (that field group is recorded once per patient, not per sub-row, so it is genuinely blank on strip 4's own rows). Coincidentally correct as real-world facts (same true patient), but the value did not come from the strip under repair. Separately, on p2 strip 2 and p1 strip 2, a genuinely blank treatment cell (row 23 / row 23 respectively, verified blank on the full page image) was filled by the repair re-read with a duplicate of the preceding treatment line instead of staying empty — a fabrication, not merely a leak. This reproduced (with a larger fill set) on the `--force` retry, so it is not a one-off sampling fluke.

**Net assessment:** The reconciliation *logic* (merge, validators, conflict resolution, continuity checks, clip-signature repair mechanics) behaves as designed and was exercised correctly — e.g., p2 strip 3/4's genuine time_hh/time_mm conflict was correctly left unresolved and flagged for review rather than guessed. The acceptance failures trace to (a) v2-era-prompt strip-extraction quality on `record_no`/`patient_name` for p1/p2 defeating the continuation-detection heuristic, and (b) the `CLIP_MIN_EMPTY = 5` threshold missing the one real repair case (314) it was designed to catch, by a single field. No prompts or thresholds were tuned to compensate, per task instructions. Reported **DONE_WITH_CONCERNS**.
