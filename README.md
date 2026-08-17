# odh-vision

Digitizing handwritten Ugandan health-facility (ODH) OPD registers: photographed register pages go in, structured per-patient records come out.

The pipeline splits the problem in two:

1. **Deterministic image processing (OpenCV, free, local)** — rectify each photographed page, clean it, detect the printed table grid, and crop one image *strip per patient record*. ✅ implemented
2. **Handwriting extraction (multimodal LLM, the only paid step)** — read each record strip into structured JSON, with confidence tracking and human verification. 🔜 next phase (a manual Claude-assisted flow exists in scripts 3–8)

Feeding the model small, pre-aligned, single-record strips instead of whole page photos removes the classic failure mode of table OCR (values drifting between rows) and raises effective resolution per model call.

## How segmentation works

```
PDF ──1_render_pages.py──▶ 300 DPI PNGs ──1b_segment_records.py──▶ _segments/<page>/
```

Per page, `segmentation.py`:

1. **Finds the table** — adaptive threshold + largest 4-corner contour (the printed border).
2. **Rectifies** — perspective-warps the border onto a canonical 2000×1400 rectangle, fixing keystone, skew, and 90° rotation in one transform.
3. **Orients** — resolves the remaining 180° ambiguity by comparing ink in the page margin just above vs. below the table (the printed legend line sits above the table on this form). An in-table ink-density heuristic was tried first and empirically *doesn't work* — handwritten rows out-ink the printed header; see the spec's integration notes.
4. **Cleans** — background division (removes shadows from photographed pages), best-contrast color channel, CLAHE. Output stays grayscale — never binarized, since hard thresholding erases faint pen strokes.
5. **Detects the grid** — morphological long-line extraction, clustered into row/column coordinates; rows grouped into the form's 5 record blocks (3 sub-rows each).
6. **Emits** per record: a PNG strip with the printed column-header band stitched on top, plus a rectified full page, a debug overlay (detected grid drawn on the page), and a JSON manifest (record y-ranges, best-effort column x-coordinates, status, warnings).

**Fail-safe by design:** if the border isn't found or grid detection doesn't validate, the page emits *no strips* — only a `needs_review` manifest and the debug overlay. Bad segmentation can never silently feed the extraction step.

## Quickstart

Requires Python 3.10+, [poppler](https://poppler.freedesktop.org/) (`brew install poppler` / `apt install poppler-utils`).

```bash
pip install -r requirements.txt

# 1. Render a register PDF to page images (→ _output/)
python 1_render_pages.py path/to/register.pdf

# 2. Rectify + segment into record strips (→ _segments/)
python 1b_segment_records.py --glob "_output/register_p*.png" --contact-sheet
```

Then open `_segments/contact_sheet.jpg` — every page's debug overlay tiled in one image, green-bordered if segmentation succeeded, red if flagged for review. Exit code `0` = all pages ok, `2` = some pages need review.

Per-page output in `_segments/<page>/`:

| File | What it is |
|---|---|
| `<page>_rec1..5.png` | Record strips — the extraction inputs |
| `<page>_full.png` | Rectified, cleaned full page |
| `<page>_debug.jpg` | QA overlay (header line, record boxes, columns) |
| `<page>.json` | Manifest: status, record y-ranges, warnings, best-effort `col_x` |

On the 23-page validation register, 22/23 pages segment cleanly (every one verified correctly oriented); the remaining page is correctly flagged rather than guessed.

## Repository layout

- `segmentation.py` — the OpenCV library (pure functions, no I/O side effects beyond what `segment_page` writes)
- `1_render_pages.py`, `1b_segment_records.py` — pipeline steps 1 and 1b
- `2_enhance_regions.py` … `8_progress_report.py`, `config.py`, `extraction_helpers.py` — the original manual extraction workflow (Claude-in-the-loop reading of pages into a master Excel workbook, with confidence flags, verification, and audit sampling); being progressively replaced by scripted extraction
- `tests/` — 18 pytest tests, all running on synthetic register images with known geometry, so CI never needs real scans
- `docs/superpowers/` — design spec (including real-page integration results and tuning evidence) and the implementation plan
- `feedback.md` — project review and roadmap: scripted multimodal extraction with ensemble confidence, terminology coding, gazetteer learning loop, FHIR output

```bash
python -m pytest -v   # run the test suite
```

## Data privacy

Register scans contain patient information (names, villages, diagnoses). **No scan data is, or ever has been, in this repository**: source PDFs, rendered pages (`_output/`), and all segmentation outputs (`_segments/`) are gitignored, and the exclusions were in place before the first commit. Keep it that way — never commit rendered images, strips, contact sheets, or workbook files, and check `git status` before staging anything new.

## Roadmap

- [ ] Scripted extraction: cheap multimodal model per record strip, schema-enforced JSON output
- [ ] Two-model agreement as the confidence signal; disagreements escalate to a stronger model or human review
- [ ] Make manifest `col_x` reliable enough for per-cell crops (needed for the verification phase; currently best-effort — don't consume it)
- [ ] Gazetteers (names/villages per center) that grow from human corrections
- [ ] Diagnosis/drug/test coding (ICD-10 / ATC / LOINC) and FHIR bundle output
