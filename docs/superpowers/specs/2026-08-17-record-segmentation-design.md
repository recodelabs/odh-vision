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
