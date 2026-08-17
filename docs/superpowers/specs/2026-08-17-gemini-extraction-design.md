# Gemini Strip Extraction — Design Spec

**Date:** 2026-08-17
**Status:** Approved (design confirmed in session with mberg; per-strip extraction, Flash-tier models, Vertex AI auth)
**Context:** Phase 2 of the odh-vision pipeline. Phase 1 (spec `2026-08-17-record-segmentation-design.md`) produces per-record grayscale strips in `_segments/<stem>/` with JSON manifests. This phase reads each strip into structured JSON with Gemini.

## Goal

For every `ok` page, extract each record strip into a form-faithful, schema-validated JSON record with per-field confidence — scripted, resumable, cost-tracked — and provide an A/B harness comparing `gemini-3.5-flash-lite` against `gemini-3.7-flash` on accuracy and cost.

## Decisions (made with the user)

- **One API call per record strip**, all fields at once. No sub-section crops: column boundaries (`col_x`) are best-effort, handwriting overflows printed cells, and cross-field context aids reading. Sub-crops remain the *verification-phase* escalation path.
- **Models:** start `gemini-3.5-flash-lite` (default), compare against `gemini-3.7-flash`. Pricing (standard tier, per 1M tokens, verified 2026-08-17; 3.x rates double Jan 1 2027): flash-lite $0.25 in / $1.50 out; 3.7-flash $0.75 in / $3.75 out; batch tier halves both.
- **Auth: Vertex AI with a service account** (`GOOGLE_APPLICATION_CREDENTIALS` → SA JSON; project from `GOOGLE_CLOUD_PROJECT` or the SA file's `project_id`; location `GOOGLE_CLOUD_LOCATION`, default `global`). An API-key path (`GEMINI_API_KEY`) is supported as fallback for portability. Secrets live in a gitignored `.env` in the repo root; never committed, never printed.
- **Schema is form-faithful** (mirrors the printed register, not the legacy spreadsheet): richer capture for the later FHIR mapping; legacy columns are derivable from it.

## Architecture

```
_segments/<stem>/            extraction.py (new library)
  manifest + strips     ──▶  1c_extract_strips.py (CLI)  ──▶  _extractions/<model>/<stem>.json
                             1d_compare_models.py (CLI)  ──▶  agreement report + disagreements CSV
```

New files: `extraction.py`, `1c_extract_strips.py`, `1d_compare_models.py`, `tests/test_extraction.py`, `tests/test_compare.py`. Legacy scripts 2–8 untouched. `config.py` gains `EXTRACTIONS_DIR`. `.gitignore` gains `.env`, `.gcp-sa.json`, `_extractions/`. `requirements.txt` gains `google-genai`, `pydantic`.

### Record schema (pydantic; used directly as Gemini `response_schema`)

Every field is a `Reading {value: str, confidence: "high"|"medium"|"low"|"illegible"}`. `illegible` ⇒ `value` must be `""` — never guessed (same philosophy as phase 1's flags). Checkbox fields use value `"Y"`/`"N"`/`""`; enumerated cells use their printed domain (`am_pm`: AM/PM/"", `sex`: M/F/"", `result_pn`: P/N/"").

Fields (form order): `record_no, day, month, time_hh, time_mm, am_pm, voucher_na, voucher_color, voucher_id, patient_name, village, village_code, first_time_odh, first_voucher_use, sex, hh_owns_phone, hh_owns_toilet, last_care, group_appt, age_yrs, hoh_education, tests, result_pn, malaria, sev_malaria, weight_kg, diagnosis, art_dose, treatment_line1..3, tab_no, full_cost, balance, cost_after_discount` plus a free-text `row_notes` for anomalies (merged rows, crossed-out entries).

### Extraction call

`google-genai` SDK, `generate_content` with the strip PNG + prompt, `response_mime_type="application/json"`, `response_schema=RecordExtraction`, `temperature=0.0`, thinking budget 0 by default (transcription needs no reasoning tokens; if the model rejects the thinking config, retry once without it). Retries with exponential backoff on 429/5xx/RESOURCE_EXHAUSTED, 4 attempts. Usage (input tokens, output+thinking tokens, latency) captured from `usage_metadata` per call.

The prompt describes the strip's fixed layout (printed header band on top, one record = 3 sub-rows), the column map, checkbox semantics, and verbatim-transcription rules; optional `--center`/`--year` context strings are appended when supplied. `PROMPT_VERSION` is stamped into every output file.

### Output format — `_extractions/<model>/<stem>.json`

```json
{
  "stem": "..._p1", "model": "gemini-3.5-flash-lite", "prompt_version": "1",
  "records": {
    "1": {"fields": { "patient_name": {"value": "...", "confidence": "high"}, ... },
           "usage": {"input_tokens": 812, "output_tokens": 604, "latency_s": 3.1}}
  },
  "totals": {"input_tokens": ..., "output_tokens": ..., "est_cost_usd": ...}
}
```

One file per page per model; separate model directories keep A/B runs and future prompt-version re-runs clean. Written atomically (tmp + `os.replace`) after every record, so a crash never loses completed work.

### Operational rules

- **Resume-safe:** records already present in the output file are skipped unless `--force`.
- **Refuses `needs_review` pages** (no strips exist for them anyway; the refusal is explicit in output).
- `--dry-run` counts pending strips and prints a cost estimate from the pricing table without calling the API. `--limit N` caps strips per run for cheap trials.
- Run summary: strips extracted/skipped, tokens in/out, estimated cost from actual usage.

### A/B harness — `1d_compare_models.py`

Pure local (no API). Loads both models' extraction files for the same stems and reports: overall field agreement rate, per-field agreement (sorted worst-first), agreement by confidence pair, and writes `_extractions/compare_<m1>__<m2>.csv` with one row per disagreement (stem, record, field, value+confidence from each model) for human adjudication. Values are normalized before comparison (whitespace collapse + casefold); both verbatims are preserved in the CSV. Agreement-on-nonempty and both-empty are tracked separately so "both blank" doesn't inflate the score.

### Cost model (recorded for planning)

~1.5k tokens per strip round trip ⇒ full 312-PDF archive (~36k strips): flash-lite ≈ $27 standard / ≈ $14 batch; 3.7-flash ≈ $74 / ≈ $37. A 23-page A/B (both models, 110 strips) ≈ $0.35. Batch-tier submission is a future optimization, out of scope this phase.

### Testing strategy

- Unit tests run **fully offline** against a stub client object (same attribute surface as the SDK response: `.parsed`, `.text`, `.usage_metadata`): schema round-trip, env loading, retry/backoff behavior, resume + `--force`, needs_review refusal, atomic write, cost math, comparison stats/CSV.
- One **live smoke test** (`tests/test_live_extraction.py`) auto-skipped unless credentials are present: extracts one real strip and asserts a valid `RecordExtraction` comes back.
- Final integration: mini A/B over sample pages with both models once credentials land; results appended to this spec.

## Out of scope (later phases)

Batch API submission, the two-model ensemble as a *pipeline* stage (this phase only builds the comparison harness), verification re-reads from cell crops, gazetteer injection, terminology coding, FHIR output, and writing into the legacy Excel workbook.

## Live results (2026-08-17)

Live smoke test (`tests/test_live_extraction.py`) passed on the first real Vertex AI call (the configured GCP project, location `global`, service-account auth) — no permission/API-not-enabled/model-not-found issues. Full suite: 45 passed. Mini A/B run: pages p1–p3 (15 strips), both models, `--center "Kameno" --year 2026`. Total spend: **$0.137** (well under the $0.35 ceiling).

### Tokens/cost: actual vs planning figures

Planning figure was ~1000 in / 500 out tokens per strip round trip. Actuals came in very different:

| Model | avg in/strip | avg out/strip | avg latency | run total (15 strips) | actual $/strip |
|---|---|---|---|---|---|
| gemini-3.5-flash-lite | 5419 | 996 | 5.16s | $0.0427 | $0.00285 |
| gemini-3.7-flash | 5419 | 593 | 5.98s | $0.0943 | $0.00629 |

Input tokens are identical between models per strip (same image + prompt) and are **~5.4× the planning estimate** — the 300 DPI cropped-strip PNG costs far more in vision tokens than assumed. Output tokens are close to planned for 3.7-flash (593 vs 500) but flash-lite ran **~2×** planned (996 vs 500) despite `thinking_budget=0` on both — flash-lite's non-thinking output is simply more verbose per call; no thinking-config rejection/drop occurred on either model (0 errors, 0 retries across all 30 calls).

**Revised full-archive projection** (312 PDFs, ~36k strips, standard tier): flash-lite ≈ **$103** (vs $27 planned), 3.7-flash ≈ **$226** (vs $74 planned). Batch tier (half price) would bring these to ≈$51 / ≈$113. This is the single biggest surprise of the live run and should inform any full-archive budget decision — batch-tier submission (already noted as future work) becomes considerably more valuable at these actuals.

### Agreement (p1–p3, 15 records, both models)

Overall agreement: **63.1%** (366 non-empty field-pairs compared, +159 both-empty agreements excluded from the rate). 135 disagreements written to `_extractions/compare_gemini-3.5-flash-lite__gemini-3.7-flash.csv` (gitignored, not committed — contains PHI).

Worst fields (agreement rate): `voucher_na` 0% (2/2), `group_appt` 0% (5/5), `art_dose` 0% (6/6), `treatment_line1` 6.7% (1/15), `first_voucher_use` 14.3% (1/7), `treatment_line3` 18.2%, `voucher_color` 20%, `treatment_line2` 26.7%. Best fields: `month`, `time_hh`, `time_mm`, `am_pm`, `village_code`, `last_care` all 100%.

### Vision spot-check (3 strips, images vs JSON)

- **`..._p1_rec1`** (clean strip): both models read demographics, times, and costs correctly. flash-lite marked `result_pn` as `""` at **high confidence** when the strip clearly shows a handwritten "P"; 3.7-flash correctly read `result_pn: "P"` (high confidence). This is the concerning pattern: a **confident miss**, not a flagged low-confidence guess — the honesty contract (low confidence ⇒ hedge) held for illegible cells but not for this kind of clean-but-skipped read.
- **`..._p3_rec5`** (strip with its top sub-row cropped at the image edge by segmentation): 3.7-flash correctly left `treatment_line1` empty (unreadable/absent row) and placed the two visible treatment lines in `treatment_line2`/`treatment_line3`. flash-lite shifted everything up by one row — visible line 2's text went into `treatment_line1`, line 3's into `treatment_line2`, leaving `treatment_line3` empty — a genuine row-misalignment error, not a disagreement over illegible handwriting. Also on this strip flash-lite swapped `voucher_na`/`voucher_color` values (put the color-column digit into the NA field). This traces to a segmentation-strip boundary issue (record 5's top sub-row clipped) rather than a schema/prompt bug, but 3.7-flash handled the degraded input more gracefully.
- **`..._p2_rec3`** (clean strip, messy handwriting): disagreements here look like genuine handwriting ambiguity — flash-lite read age as "38" where 3.7-flash (and visual inspection) support "58"; flash-lite dropped a leading digit on `full_cost` ("1000" vs correct "11000"); flash-lite missed `voucher_color` ("" vs correct "1"). All were marked **high confidence** by flash-lite despite being wrong — again, confident misses rather than hedged guesses.
- **Illegible⇒empty check**: scanned all 1050 field readings across both models' p1–p3 output — **zero violations** (no reading has `confidence: "illegible"` with a non-empty `value`).

**Qualitative summary:** both models read clean, well-aligned printed/checkbox fields (dates, times, AM/PM, sex, last-care code) essentially perfectly. Both struggle on free-hand treatment lines (abbreviation-heavy prescriptions) — genuine ambiguity, not a defect. flash-lite's distinguishing weakness across all three spot-checked strips was **confident wrong answers** on fields it should have flagged (skipped a legible "P", misread a digit, dropped a digit, mis-assigned a checkbox) rather than honest low-confidence hedges, and it was more fragile on the one edge-clipped strip. 3.7-flash was more accurate on exactly these cases at ~2.2× the cost. For a production pipeline, the field-level agreement CSV plus per-field confidence should drive which fields get single-model (flash-lite) vs dual-model/human-adjudicated treatment — `voucher_na`, `group_appt`, `art_dose`, `treatment_line1-3`, `voucher_color` are the weakest and worth escalation.

### Other surprises

- No permission, API-enablement, or model-not-found issues on the very first live call — Vertex AI project config was correct out of the box.
- No retries were triggered across 31 live calls (1 smoke test + 30 A/B); no 429/5xx and no thinking-config rejections.
- Latency ~5-6s/call for both models, page-level runs (5 strips) took well under a minute each — no timeout concerns at this scale.

### Prompt v2 / buffered strips re-run (2026-08-17)

**What changed:** `segmentation.emit_record_strips` now pads each record's crop by `STRIP_PAD = 25px` above/below its true (y0, y1), clamped to `[header_bottom, H]`, and draws the true boundaries as red horizontal lines on the (now 3-channel) strip PNG. Manifest record entries gained `pad_top`/`pad_bottom` (actual applied padding after clamping). `extraction.PROMPT_VERSION` bumped to `"2"`; the prompt gained a rule telling the model the record lies between the two red lines and that content above/below belongs to adjacent records (use it only to complete crossing pen strokes, never transcribe it).

**Re-segmentation:** all 23 pages re-segmented with the new code — **22 ok / 1 needs_review (p6)**, identical to the pre-change baseline. No regression in ok-count.

**Visual check:** `..._p1_rec1` — red lines sit exactly at the header/body and body/next-record boundaries; padding shows a sliver of the adjacent record above/below as expected. `..._p3_rec5` — the padding **does** now include content from patient 314 (Okwir Moses)'s record, which the original (pre-buffer) extraction had entirely missed because it straddled the rec4/rec5 crop boundary (documented in `_extractions/claude_vs_gemini37_p1-3.md`, patient 314 row). Pixel inspection confirmed the true y0 boundary (the printed grid line) falls essentially through the middle of patient 314's first sub-row — the row's ink (name, diagnosis, treatment, cost) sits mostly *above* the line, so it now appears in rec5's `pad_top` zone (and, symmetrically, in rec4's `pad_bottom` zone) rather than being clipped by either crop.

**314-recovery outcome (re-extracted with prompt v2):** of the record's fields that were previously entirely absent from Gemini's output (`patient_name`, `sex`, `first_time_odh`, `hh_owns_phone`, `hh_owns_toilet`, `result_pn`, `diagnosis`, `treatment_line1`, `full_cost`), **gemini-3.7-flash's rec5 now recovers 5 of 9**: `patient_name` "Moses", `result_pn` "P", `diagnosis` "Malaria", `treatment_line1` "T. Act 1 bd x 3/2" (7 should read x3/7), `full_cost` "3500" — all verified correct against the strip image. **4 remain empty** (`sex`, `first_time_odh`, `hh_owns_phone`, `hh_owns_toilet`) despite the ground truth (checked via zoomed pixel inspection of the strip) showing `sex`=M, `first_time_odh`=Y, `hh_owns_phone`=Y, `hh_owns_toilet`=Y all marked. These four are the checkbox fields furthest left in the straddling row, right at/above the red line — the model appears to have applied the new "don't transcribe adjacent content" rule inconsistently across a single physical row (skipping the near-boundary checkboxes while still reading diagnosis/treatment/cost further right on the same row). No hallucination or leakage of *wrong* adjacent-record data was observed in either rec4 or rec5 — rec4 correctly kept patient 313 (Aber Susan)'s own fields and did not absorb patient 314's data. gemini-3.5-flash-lite showed much weaker recovery (misattributed the village name "Bulaga" as `patient_name`, `diagnosis`/`full_cost` still empty) — consistent with its generally lower accuracy documented above, not attributable to the padding change.

**Regression spot-check:** patient 306 (p1 rec5: name/sex/village/village_code/age_yrs) and patient 309 (p2 rec5: name/sex/hh_owns_phone/hh_owns_toilet/age_yrs) both match the pre-change adjudicated values exactly under gemini-3.7-flash — no boundary-line-induced regression, and no adjacent-record content leaked into either record.

**New agreement rate:** re-running `1d_compare_models.py --models gemini-3.5-flash-lite gemini-3.7-flash` on the re-extracted p1-p3 gives **54.7%** overall agreement (380 non-empty field-pairs, +145 both-empty), down from the prior **63.1%**. This is expected: the padded strips changed what content is visible near record boundaries (e.g. patient 314's row), and the two models diverge in how completely they use it — flash-lite's weaker recovery mostly stayed empty/wrong while 3.7-flash's partial recovery introduced new non-empty values with no counterpart to agree with. Per-field pattern is broadly similar to before (checkbox/treatment fields remain the weakest).

**Cost of this re-run:** segmentation regeneration was local/free. Re-extraction of p1-p3 with both models: gemini-3.5-flash-lite $0.0428 (82,061 in / 14,855 out tokens) + gemini-3.7-flash $0.0895 (82,061 in / 7,465 out tokens) = **$0.1323 total**, within the pre-approved ~$0.15 ceiling.

**Assessment:** the fix delivers a genuine, verifiable partial recovery of the motivating defect (5 of 9 previously-lost fields on patient 314, with correct values, no adjacent-record leakage) and no regression on previously-correct fields. It does not fully close the gap — some near-boundary checkbox fields are still dropped, apparently due to the model applying the new adjacency rule conservatively to the exact row that straddles the line. This is a partial-help outcome, not a hurting one (no case of a model transcribing genuinely adjacent/foreign-record data into a field was observed) — reported as-is per instructions, without prompt-tuning on top of the pre-approved change.

**Prompt v3 / origin-ownership rewording (2026-08-17):** the boundary rule was reworded from a strict "never transcribe values above/below the red lines" framing to an origin-ownership framing — content fully outside the red lines still belongs to adjacent records, but a mark or entry that *originates* inside this record's rows (including checkbox marks and text partly covered by a red line) belongs to this record even if it visually extends past the line. `PROMPT_VERSION` bumped to `"3"`. Re-extracting only `p3` with gemini-3.7-flash under prompt v3 ($0.0322, 5 strips) left patient 314 (rec5)'s four still-empty checkboxes (`sex`, `first_time_odh`, `hh_owns_phone`, `hh_owns_toilet`) **unchanged** — still empty. Pixel-level re-inspection of `_segments/.../..._p3_rec5.png` explains why: the checkbox marks live on patient 314's first sub-row's "Y" checkbox line, and that line's checkbox *cells* are almost entirely cropped out of the rec5 strip itself — the strip's `pad_top` sliver above the red line shows only blank cell borders with no ink, while the full marks (⊠Y first_time_odh, ⊠M sex, ⊠Y phone, ⊠Y toilet, all confirmed against the image) are visible only in `..._p3_rec4.png`'s `pad_bottom` zone. So this is not a prompt-wording problem — no amount of rule rewording can recover pixels the model was never shown — it is a segmentation padding-insufficiency problem (rec5's `pad_top` needs to be taller, or asymmetric, to include that checkbox line). A prompt-only fix cannot close this specific gap; a segmentation change would be the next candidate fix, out of scope for this pre-approved follow-up. Separately, this same re-run surfaced a regression to flag: record 4 (patient 313, Aber/Abeu Susan)'s `voucher_id` came back as `048239` (confidence `medium`) versus the correct `048231` (confirmed by direct pixel read of the strip, and matching the prior prompt-v1/v2 extraction) — name, village, and `hh_owns_toilet`=`Y` were unaffected. All five `p3` records' `prompt_version` now correctly stamp `"3"`, confirming the provenance re-stamp fix (extraction.py `extract_page`) took effect on this force re-run.
