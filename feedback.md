# ODH Vision Pipeline — Review & Feedback

**Reviewed:** 2026-08-17
**Scope:** all 10 pipeline files (`config.py`, `extraction_helpers.py`, scripts 1–8) plus the sample register `20260319_053700_KAM_Stlhb.pdf` (23 pages, Kameno center).

---

## 1. What this is

An 8-step pipeline for digitizing handwritten Ugandan OPD (outpatient department) registers: render PDF pages to images → enhance regions → extract records with a vision LLM → store into a master Excel workbook → verify flagged cells → audit random samples → report progress. The extraction step is currently **manual**: a human pastes the prompt + image into a Claude session and saves the JSON response.

The sample PDF is exactly the hard case: pre-printed A4 grid, one patient record spanning **three physical sub-rows**, dense checkboxes (1st time at ODH, voucher use, sex, HH phone/toilet, malaria), village-code legend printed in the page header, three cost columns (full cost / balance / cost after discount), blue paper, photographed (not flatbed) pages with shadows and mixed orientations.

---

## 2. Strengths

- **Clear pipeline decomposition.** Numbered scripts with one job each, a shared helpers module, and a single config module. Docstrings are genuinely good — each script documents usage, input formats, and JSON shapes. A new contributor could run this from the docstrings alone.
- **Honest uncertainty model.** The graduated confidence palette (`[verified]` / `[?]` / `[faded]` / `[??]` / `[illegible]`) with the explicit rule "illegible = blank, never guessed" is the right philosophy for health data. Most OCR projects skip this entirely.
- **Human-in-the-loop verification is a first-class step**, not an afterthought: list flagged cells → re-read at higher resolution → apply corrections with provenance coloring.
- **Smart audit sampling.** Weighting audits *toward* high-confidence PDFs (step 6) is a clever inversion — errors hiding in "trusted" data are the most dangerous, and low-confidence data is already flagged.
- **Concurrency care.** Advisory `flock` locking, atomic save via temp-file + `os.replace`, and automatic `.bak` backup around every workbook write. Rare to see in a scripts-grade project.
- **Progress and audit accuracy are measurable** (steps 7–8), so the project can make quantitative claims about extraction quality instead of vibes.

---

## 3. Weaknesses

### 3.1 Correctness bugs

- **Page rotation is assumed, not detected.** `1_render_pages.py` rotates any portrait page 90° counter-clockwise (PIL's default direction) with no orientation check. The sample PDF shows pages in both orientations, though that's mostly an artifact of the first few pages having been manually rotated — so this is less prevalent than it looks. Still worth a cheap verification (locate the printed header band / page number, or Tesseract OSD) as part of the OpenCV preprocessing pass (§4.6), where it comes nearly for free.
- **Re-running step 4 duplicates data.** `append_records()` blindly appends; there is no idempotency check on (Source PDF, Page, Record No.). One accidental re-run of `4_store_records.py` silently doubles a page's records, and the audit/progress counts inherit the corruption. Fix: refuse (or upsert) when the PDF+page already has rows, with a `--force` override.
- **Confidence markers pollute the data.** Appending `[?]`, `[verified]` etc. into cell *values* means "Cost (UGX)" is no longer numeric, names carry noise into any downstream analysis, and you now have **two sources of truth** (text markers and cell fill colors) that can disagree — `update_record_fields` colors from the marker, but initial storage in step 4 writes markers with *no* fill at all, so `flagged_cells_for_pdf` (color-based) misses every uncertainty from initial extraction unless someone colored cells by hand. Fix: one machine-readable confidence channel (see §4.1).
- **`cell_rgb` breaks on theme/indexed colors.** openpyxl returns theme-indexed color objects (not ARGB strings) for cells colored via Excel's standard palette; a human touching up cells in Excel produces fills this code can't read. Guard for `fill.fgColor.type == "rgb"`.
- **Progress "Verified" column is dead.** `update_progress` documents column E = Verified but never writes it; `8_progress_report.py` instead greps Notes for the magic string `"verified-done"`, which nothing in the codebase ever writes. Verification % will read 0% forever unless someone knows the secret handshake.
- **`update_progress` can't reset state.** All fields use truthy checks (`if extracted:`, `if records_count:`), so you can never set a status back to empty or a count to 0.
- **Prompt/form mismatches.** The prompt says "if a row spans two physical rows, merge" — but every record on this form spans **three** sub-rows. It asks for a full "Date" ("07/02/2026") when the form has only Day + Month cells (year is on the page header). It asks for one "Cost (UGX)" when the form has three cost columns. It never mentions checkboxes, which is most of the form. The model will improvise on all of these, differently on different days.

### 3.2 Design & robustness

- **Excel-as-database.** Every operation loads and rewrites the entire workbook; `update_record_fields` is an O(all-rows) scan per correction; state lives partly in cell colors. This works at 312 PDFs but is fragile (one corrupt save = whole dataset, no history, merge conflicts impossible to resolve). See §4.1.
- **`config.py` runs `pip install` at import time.** Auto-installing packages as a side effect of `import config` is surprising, breaks in managed/offline environments, and runs on every import. Also `_which()` shells out to `which` instead of using `shutil.which()`. Replace with a `requirements.txt` + a friendly error message.
- **The audit measures the wrong thing.** Step 7 compares stored values against a *re-read by the same model on the same image*. Correlated errors (the model misreads "Ssemwuda" the same way twice) count as MATCH, so accuracy is overstated. Better: a different model for the audit re-read, or human adjudication for the audit sample only.
- **No response validation.** `parse_extraction_response` grabs the first `[`…last `]` and `json.loads` it — no schema check. Misspelled keys are then *silently dropped* by `append_records` ("unknowns are silently ignored"), so a model that returns `"Village name"` instead of `"Village"` loses the whole column with no warning.
- **No version control, no tests, no manifest.** The directory isn't a git repo; there's no `requirements.txt`/`pyproject.toml`, no tests (the pure functions — `tag_confidence`, `parse_extraction_response`, `_fill_for_marker`, audit stats — are trivially testable), and cruft is lying around (`odh_pipeline_scripts.zip`, empty dir, `.DS_Store`). **Caution:** when you do init git, `.gitignore` the PDFs and workbook — they contain patient names + diagnoses (PHI). Same caution applies to which model APIs the images are sent to and under what data-processing terms.
- **Hardcoded `TOTAL_PDFS = 312`** — walk `OPTIMIZED_DIR` and count instead.
- **Enhancement upscales JPEGs.** `enhance_region` crops a 200-DPI JPEG and Lanczos-upscales it — that invents pixels rather than recovering detail. For re-reads, re-render the region *from the PDF* at 400–600 DPI (`pdftoppm -r 600 -x -y -W -H`) so the model actually sees more information. Also: 200 DPI JPEG is a lossy starting point for faint handwriting — render at 300 DPI PNG (or `-jpegopt quality=95`).

---

## 4. Improvements

### 4.1 Structured data → FHIR (the destination shapes everything upstream)

Since the target is FHIR, make the canonical store **structured records, not a spreadsheet**. Concretely:

- **Canonical store = SQLite (or JSONL) with a typed schema**; generate the Excel workbook as a *view* for the human verification workflow. Give every field three channels: `value_verbatim` (exactly what's on paper), `value_coded` (normalized/coded), and `confidence` (enum) — this kills the markers-in-values problem and gives FHIR clean typed inputs.
- **Resource mapping** (one register row → one small Bundle):
  - `Patient` — name, sex, approximate birthDate from age (flag as estimated), village → address
  - `Encounter` — visit date/time, `Location`/`Organization` = center (Kameno etc.)
  - `Condition` — Diagnosis, coded (see below), with `verificationStatus` driven by your confidence flag
  - `Observation` — MRDT/RPR/other test results, weight; malaria RDT has LOINC codes
  - `MedicationStatement` (or `MedicationRequest`) — one per treatment line, with parsed dose/form/duration
  - `Coverage`/`Claim` — voucher color/ID, costs (full/balance/after-discount as separate money fields)
  - `Provenance` — source PDF, page, record no., extraction model + prompt version, confidence, verifier. This is gold for auditability and you already track most of it.
- Emit NDJSON bundles and load with your existing OpenFn tooling into HAPI / Google Healthcare API. Validate with the FHIR validator in CI so garbage can't reach the server.
- **Do the typing early.** FHIR will force types eventually (dates, quantities, codes); validating at extraction time (pydantic model per record) means errors surface while the page image is still in front of you, not during a FHIR load six months later.

### 4.2 Terminology / code matching (currently absent — you're right that it's fluid)

Right now Diagnosis/Treatment/Tests are free text normalized only by prompt suggestion. Add a post-extraction **normalization pass**:

- Maintain small controlled vocabularies: `diagnoses.csv` (ODH canonical name → ICD-10 code — Uganda MoH uses ICD-10; add SNOMED CT if OpenSRP/analytics need it), `drugs.csv` (name → strength/form → WHO ATC), `tests.csv` (MRDT, RPR, urinalysis… → LOINC).
- Fuzzy-match extracted text against the vocabulary (rapidfuzz); above threshold → auto-code, below → flag for human pick-list review. Keep verbatim text alongside the code always.
- Parse treatment strings structurally (`C. Doxy 100mg bd x 5/7` → drug, dose, frequency, duration). A cheap LLM call with a JSON schema does this well; it's a text-only task, so it costs almost nothing.
- Built-in validators the form gives you for free: village **name vs. village code** (the code legend is printed on every page header — extract both and cross-check); malaria checkbox vs. Diagnosis="Malaria"; MRDT result P/N vs. malaria result column; record-number continuity per page (gaps = missed rows); date within the register's month; cost arithmetic where all three cost cells are present.

### 4.3 Learning strategy (corrections should compound)

Yes — build the feedback loop you described. Cheap version that works:

- **Gazetteers**: `names.txt`, `villages.txt` (seed from the printed village-code legends, per center), `diagnoses`, `drugs`. Every time a human verifies or corrects a value in step 5, append the confirmed form to the gazetteer. The hardcoded name list in the prompt ("Nakato, Katusiime…") becomes a *generated* section: inject the top-N most frequent confirmed names/villages **for that center** into the prompt. Villages especially are center-local — the sample page's legend is 12 villages — so per-center context is small and high-value.
- **Confusion log**: store every correction as a (misread → correct, field) pair. Frequent pairs ("Ssemwuda" misread as "Ssemwanda") become explicit prompt warnings and fuzzy-match priors.
- **Post-hoc snapping**: after extraction, snap names/villages to the nearest gazetteer entry above a similarity threshold; below it, flag rather than snap. This converts the gazetteer into accuracy even when the prompt is ignored.
- Track hit rates so you can see the loop working (auto-verified % should climb over time).

### 4.4 Row-level clipping — yes, it will help

Full-page extraction of a 5-record × ~20-column grid is where vision models fail most: **row drift** (value from record 3 assigned to record 4) and small-cell misreads. Recommended architecture:

1. **Segment by grid, not by guesswork.** The printed grid is high-contrast and regular — OpenCV morphological line detection (or even fixed geometry after deskew, since the form is standardized) finds the record boundaries reliably. Each record block = its 3 sub-rows.
2. **Feed one record-strip at a time**, with the column-header band stitched on top of each crop so the model always sees labels adjacent to values. This raises effective resolution per token enormously and makes row drift structurally impossible.
3. Keep a cheap **page-level pass** for global facts (center, year, page number, record-number range, village legend) and for the continuity check.
4. Per-record calls cost more per page — that's fine once extraction is scripted with a cheap model (§4.5), and it buys you caching + surgical re-reads (re-run one record, not one page).
5. For the verification step, go further: per-*cell* crops rendered from the PDF at 600 DPI.

### 4.5 Script it with cheap multimodal models (remove the human copy-paste)

The manual Cowork step is the bottleneck and the least reproducible part. Replace step 3's "print prompt, paste response" with an API script:

- **Primary**: Gemini Flash-class model (cheap, strong handwriting OCR) called per record-strip with a **structured-output JSON schema** (not "return only JSON" — schema-enforced output deletes the fence-stripping/parsing fragility entirely).
- **Ensemble as confidence**: run two cheap models (or the same model twice at different temperature/crops) and compare field-by-field. Agreement → auto-accept; disagreement → that field is flagged and escalated to a stronger model (Gemini Pro / Claude) or to the human queue. This replaces the subjective "[?] if unclear" self-assessment with a measurable signal, and fixes the self-audit problem in §3.2 for free (the audit re-read becomes a *different* model).
- Batch APIs give ~50% cost reduction; at 312 PDFs × ~23 pages × ~5 records this is thousands of calls, so it matters.
- Log model + prompt version per record (you already have the "Optimization used" column — extend it) so accuracy regressions are attributable.
- Classical OCR engines (Tesseract etc.) will not handle this handwriting; multimodal LLMs are the right tool. Don't spend time on a Tesseract path except for orientation detection.

### 4.6 Pre-processing — worth doing, in this order

What the sample pages actually suffer from: photographed (not scanned) pages → perspective keystone, uneven illumination/shadows, blue paper with color cast, mixed orientation, and faint pen strokes.

1. **Orientation check** (§3.1) — cheap and zero risk once the grid-detection code exists; the grid's header band tells you which way is up.
2. **Perspective correction + deskew** using the printed grid corners (OpenCV `findContours` → `warpPerspective`). This also makes fixed-geometry row clipping (§4.4) trivial.
3. **Illumination flattening**: divide the image by a heavily median-blurred copy of itself (background estimate). Removes shadows/gradients without touching strokes. Almost always a pure win on photos.
4. **Color-channel selection instead of naive grayscale**: the paper is blue and much ink is blue — test which channel maximizes ink/paper contrast per page (often the red channel for blue paper) rather than a fixed luminance grayscale.
5. **Contrast**: CLAHE (adaptive) rather than the current global `ImageEnhance.Contrast(1.8)`.
6. **Binarization — be careful.** The current global threshold at 140 will erase faint strokes (exactly the `[faded]` cells you care about). If you binarize, use adaptive Sauvola/Gaussian. But for *LLM* input, cleaned **grayscale usually beats hard black-and-white** — binarization destroys the stroke-weight cues models use. Suggested policy: grayscale+flattened as the primary input; binarized version only as a *second view* attached for faint regions.
7. Render at 300 DPI PNG for extraction; 600 DPI crops for verification (§3.2).
8. **Measure, don't assume**: you have the audit harness — run a 5-page A/B (raw vs. each preprocessing stage) and keep only stages that move field accuracy. This is a genuinely nice property of this project: preprocessing claims are testable in an afternoon.

### 4.7 Smaller items

- Init a git repo (with PHI-safe `.gitignore`), add `requirements.txt`, a short `README.md`, and unit tests for the pure functions.
- Delete `odh_pipeline_scripts.zip`, the empty `odh_pipeline_scripts/` dir, and `.DS_Store`s once git exists.
- Derive `TOTAL_PDFS` from the filesystem.
- Wire the Progress sheet "Verified" column for real, and make `update_progress` able to clear fields.
- Log the audit RNG seed (and drop `random.seed(int(time.time()))` — just use the default) so audit picks are reproducible.

---

## 5. Suggested priority order

| # | Change | Why first |
|---|--------|-----------|
| 1 | Fix dedup-on-store bug | Silent data corruption today |
| 2 | Git + requirements + PHI-safe gitignore | Safety net for everything else |
| 3 | Script extraction via cheap-model API with schema-enforced JSON (§4.5) | Removes the manual bottleneck; enables everything below |
| 4 | Grid segmentation + per-record strips (§4.4) with perspective/illumination preprocessing (§4.6) | Biggest accuracy lever |
| 5 | Typed canonical store with verbatim/coded/confidence channels (§4.1) | Prereq for FHIR and kills the markers-in-values problem |
| 6 | Two-model ensemble confidence + independent audit re-reads | Turns confidence from self-report into measurement |
| 7 | Gazetteer learning loop (§4.3) + terminology coding (§4.2) | Compounds accuracy over time; FHIR-ready codes |
| 8 | FHIR bundle emitter + OpenFn load (§4.1) | The destination |

The core judgment: this is a well-organized, thoughtfully documented v1 with an unusually mature attitude toward uncertainty — its main limits are that extraction is manual, confidence is self-reported rather than measured, and Excel is doing a database's job. All three are fixable without throwing anything away.
