# Record Segmentation & Page Cleanup — Design Spec

**Date:** 2026-08-17
**Status:** Approved (design confirmed in session with mberg)
**Context:** ODH scan-extraction pipeline (`~/github/odh-vision`). See `feedback.md` §4.4/§4.6 for motivation.

## Goal

Replace whole-page images as the OCR input with clean, rectified, **per-record crop strips**, produced deterministically by OpenCV. Each strip contains one patient record (its 3 physical sub-rows) with the printed column-header band stitched on top. This structurally eliminates row-drift errors in the extraction step and raises effective resolution per model call.

## Scope

**In scope:**
- Page rectification (perspective/keystone/skew/orientation) using the printed table border.
- Image cleanup: illumination flattening, ink-contrast channel selection, CLAHE. Output is grayscale — no binarization of OCR input.
- Grid line detection → record-block segmentation → strip emission.
- Per-page JSON manifest with record y-ranges and column x-boundaries (enables lazy cell-level crops later without emitting them now).
- Debug overlays + contact sheet for human QA.
- Render step changes: 300 DPI PNG, remove blind rotation.
- Repo housekeeping: git init, PHI-safe `.gitignore`, `requirements.txt`.

**Out of scope (next phases):** the OCR/model calls themselves, cell-level crop emission, gazetteers, FHIR mapping, changes to scripts 3–8.

## Source document facts (from sample `20260319_053700_KAM_Stlhb.pdf`)

- A4 pages, photographed (not flatbed): keystone, shadows, mixed 90°/180° orientations possible.
- Blue paper, blue/black ink, pre-printed black grid.
- One table per page: a printed header band (~3 dense printed rows) + 5 record blocks, each block = 3 sub-rows.
- Village-code legend and center/year live in the page margin above the table; page number printed in the top-right table cell.

## Architecture

```
1_render_pages.py (modified)        segmentation.py (new library)
  PDF → 300dpi PNG per page   →   1b_segment_records.py (new CLI)
                                     per page:
                                       rectify → clean → detect grid →
                                       emit strips + debug overlay + manifest
                                     output: _segments/<stem>/
```

### Module: `segmentation.py`

Pure functions, OpenCV + numpy, no workbook/Excel dependencies.

| Function | Responsibility |
|---|---|
| `find_table_quad(gray)` | Locate the table's outer border as an ordered 4-corner quad (or `None`). |
| `order_corners(pts)` | Order 4 points as tl, tr, br, bl. |
| `rectify_page(img, quad)` | Perspective-warp the quad to the canonical `CANON_W × CANON_H` landscape rectangle. |
| `ensure_upright(rect_gray)` | Fix residual 180° flip: the dense printed header band must be at the top. Compares small-mark ink density (long grid lines removed) in the top vs bottom band; rotates 180° if bottom wins. |
| `best_channel(bgr)` | Pick the color channel with the highest ink/paper contrast (std). |
| `flatten_illumination(gray)` | Divide by a median-blurred background estimate → removes shadows/gradients. |
| `clean_page(bgr)` | Compose: best channel → flatten → CLAHE. Returns grayscale. |
| `detect_h_lines(gray)` / `detect_v_lines(gray)` | Morphological long-line extraction, clustered to line center coordinates. |
| `group_records(h_lines, header_bottom, table_bottom, n_records=5)` | Split the body into record blocks: ideal equal split, snapped to nearest detected line within tolerance; unsnapped boundaries produce warnings. |
| `emit_record_strips(rect_gray, header_band, records, out_dir, stem)` | Save `<stem>_rec<K>.png` strips (header band vstacked on record block). |
| `save_debug_overlay(rect_img, grid, path)` | Rectified page with detected lines/blocks drawn. |
| `segment_page(image_path, out_dir)` | Orchestrates all of the above; writes manifest JSON; returns the manifest dict. |

### Canonical geometry constants (in `segmentation.py`)

- `CANON_W, CANON_H = 2000, 1400` — rectified table size in px.
- `HEADER_FRAC = 0.13` — expected header-band fraction of table height (snap target, not hard-coded cut).
- `N_RECORDS = 5`, `SUBROWS_PER_RECORD = 3`.
- `SNAP_TOL = 0.02` — boundary snap tolerance as fraction of table height.

These are starting values; the integration task (contact-sheet QA over the 23-page sample) is where they get tuned.

### Orientation handling

1. If the input image is portrait, rotate 90° (either direction) before detection — the table is landscape.
2. Rectification maps whatever skew/keystone remains onto the canonical rectangle.
3. `ensure_upright` resolves the remaining 180° ambiguity via header-band position.

The blind `rotate(90)` in `1_render_pages.py` is removed; orientation is owned here.

### Manifest schema (`_segments/<stem>/<stem>.json`)

```json
{
  "source_image": "_output/foo_p3.png",
  "stem": "foo_p3",
  "status": "ok",
  "canonical_size": [2000, 1400],
  "header_band": [0, 182],
  "records": [
    {"index": 1, "y0": 182, "y1": 425, "strip": "foo_p3_rec1.png"}
  ],
  "col_x": [0, 55, 130, 210],
  "warnings": []
}
```

- `status`: `"ok"` or `"needs_review"`.
- `col_x`: detected vertical line x-coordinates on the rectified page — the contract that lets a later verification step crop any single cell from the rectified full page without this step emitting cell files.

### Failure behavior

If the border quad isn't found, or detected rows fail validation, the page emits **no strips** — only the debug overlay and a `needs_review` manifest with warnings. Bad segmentation must never silently feed the OCR step.

### Outputs per page (in `_segments/<stem>/`)

- `<stem>_rec1..5.png` — record strips (grayscale, cleaned)
- `<stem>_full.png` — rectified cleaned full page
- `<stem>_debug.jpg` — overlay for QA
- `<stem>.json` — manifest

### CLI: `1b_segment_records.py`

```
python 1b_segment_records.py _output/foo_p1.png [_output/foo_p2.png ...]
python 1b_segment_records.py --glob "_output/foo_p*.png" --contact-sheet
```

Prints a per-page summary line and a final count of `ok` vs `needs_review`. `--contact-sheet` tiles the debug overlays into one JPEG for fast eyeballing.

### Changes to existing files

- `config.py`: `RENDER_DPI = 300`; add `SEGMENTS_DIR`. (The pip-auto-install behavior is left alone for now; new code relies on `requirements.txt`.)
- `1_render_pages.py`: `-png` instead of `-jpeg`; `.png` filenames; delete the portrait-rotation block.

### Testing strategy

- **Unit tests on synthetic pages**: a fixture draws a known grid (border, dense header band, 15 body rows, columns), optionally perspective-warps it with a known homography and adds an illumination gradient. Detection results are asserted against the known coordinates within tolerance. No sample-PDF dependency, runs anywhere.
- **Integration**: render all 23 sample pages, segment, build the contact sheet. Acceptance: all 23 pages `status: ok` with 5 records each (or explicitly understood exceptions), verified by human eyeball of the contact sheet.

### Dependencies

`requirements.txt`: `opencv-python-headless`, `numpy`, `Pillow`, `openpyxl`, `pytest`. Repo initialized as git with `.gitignore` excluding PDFs, `ODHFILESCANS*` workbook files, `_output/`, `_segments/` (PHI-bearing artifacts must not be committed).

## Integration results (2026-08-17)

Task 8 ran the full pipeline against all 23 pages of the real sample PDF
(`20260319_053700_KAM_Stlhb.pdf`) across two rounds.

**Round 1 (reverted): grid constants alone were unsafe.** Applying only
`HEADER_FRAC=0.177` / `SNAP_TOL=0.03` (measured correct, see below) without
fixing `ensure_upright` raised the `ok` count to 6/23, but 4 of those 6 were
verified **upside-down despite `status: "ok"`** — the evenly-spaced 5-record
grid snaps just as cleanly in the wrong orientation, so an accurate grid fit
alone cannot be trusted to prove correct orientation. `ensure_upright`'s
original heuristic (compare "small marks" ink density in the top vs. bottom
15% of the *rectified table*) is essentially uncorrelated with true
orientation on this document: handwriting-dense body rows routinely out-ink
the comparatively sparse printed header, the opposite of what the heuristic
assumed. Round 1 was reverted to the safe (0/23 ok) baseline rather than ship
a silent-failure regression. Full round-1 trajectory and evidence are in
`.superpowers/sdd/2026-08-17-record-segmentation/task-8-report.md`.

**Round 2 (this fix): replaced the orientation signal with pre-rectification
margin ink, applied the verified grid constants, and lowered the line-detection
threshold.** Final result: **22/23 pages `status: ok`, 1/23 `needs_review`**
(p6 — see below), with every `ok` page individually visually confirmed
correctly oriented (debug overlays, full resolution).

- **Orientation signal:** `ensure_upright` (in `segmentation.py`) now takes
  optional `orig_gray`/`quad` parameters (pre-rectification grayscale image
  and its detected table quad; `segment_page` always supplies them). When
  present, it compares ink density in the *page margin* immediately above vs.
  below the table border, capped to 5% of the raw image height
  (`MARGIN_CAP_FRAC`). The printed margin carries two distinct legends: a
  single long "Center / Year / Village codes" line above the table, and two
  shorter "Last care / Voucher color" lines below it. The above-table legend
  is consistently denser than the two below-table lines, so whichever margin
  has more ink is the true top — a signal that doesn't depend on how much a
  given page's *body* happens to be filled in, unlike any in-table ink
  comparison. Ground truth was established by reading all 23 raw pages
  directly (Read tool, full resolution): p1 needed no flip, all of p2–p23
  needed a 180° flip. This margin signal was validated against that ground
  truth and scores **23/23** across a `MARGIN_CAP_FRAC` range of 0.025–0.06
  (0.05 chosen, comfortably centered). Several other candidates were tried
  and empirically rejected first: inverted in-table ink density (21/23,
  fails on the two pages with unusually sparse last-record content),
  small-component mean-area (20/23), and a layout-template match using
  `group_records`' own snap-warning count (0/23 — confirms the round-1
  finding that the regular grid fits almost identically well in both
  orientations, so grid fit cannot itself be used as an orientation oracle).
  The original in-table density heuristic is kept as a fallback only for
  callers that omit `orig_gray`/`quad` (none exist in production; kept so
  `ensure_upright(rect)` remains callable with just a rectified image).
- **Grid constants:** `HEADER_FRAC = 0.177` and `SNAP_TOL = 0.03`, both
  re-verified against `detect_h_lines` output on multiple confirmed-upright
  real pages (unchanged from round 1's measurement).
- **Line-detection threshold:** with orientation now reliable, round 1's
  objection to lowering `min_frac` (it would convert more upside-down pages
  into false "ok" positives) no longer applies. Direct measurement showed
  real grid lines present in heavily-written lower rows at 30–42% width span
  — below `detect_h_lines`' default 0.5 threshold — because handwriting and
  checkbox marks fragment the printed line into runs shorter than the
  long-line morphological kernel, even though the line is visibly continuous
  to the eye. Added `H_LINE_MIN_FRAC = 0.27`, used only in `segment_page`'s
  `detect_h_lines` call (the function's own default stays 0.5 for other
  callers/tests). Verified empirically across all 23 pages before picking
  0.27: values from 0.26–0.5 were swept, with ok-count rising monotonically
  from 6/23 (at 0.5) to 23/23 (at 0.26); 0.27 (22/23) was chosen over 0.26
  (23/23) for a small margin of safety against spurious handwriting-line
  false positives, while still clearing the "aim 20+" bar with headroom.
- **Residual `needs_review`:** p6 — one body sub-row boundary (`y=1146`) has
  no detected grid line within `SNAP_TOL` in either orientation; its debug
  overlay shows the other four boundaries snapped correctly and the record
  boxes look visually right, but the pipeline correctly declines to mark it
  `ok` on an unconfirmed boundary rather than assume the ideal fallback
  position is correct. Safe, conservative behavior per the Failure behavior
  invariant.
- **Test changes:** `tests/conftest.py::draw_table` gained an optional
  `margins=True` mode that draws the same above/below-table ink asymmetry the
  production signal keys on (dense line above, sparse marks below).
  `tests/test_rectify.py::test_ensure_upright_flips_header_to_top` now builds
  its "upside-down" case by rotating the *raw* synthetic page 180° and
  re-running quad-detection/rectification (matching how `segment_page` really
  drives `ensure_upright`), rather than rotating an already-rectified image;
  its final assertions were changed from bit-exact pixel equality (no longer
  guaranteed once quad-detection is independently re-run on a rotated image)
  to a density check tied to the fixture's known header-tick geometry.

Unit suite: `python -m pytest -v` → 16/16 passed.
