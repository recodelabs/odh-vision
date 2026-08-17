# Record Segmentation & Page Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rectify photographed OPD register pages with OpenCV and crop them into per-record strips (3 sub-rows + printed header band) with a JSON manifest, replacing whole-page images as the OCR input.

**Architecture:** A pure-function library `segmentation.py` (rectify → clean → detect grid → emit strips + manifest) driven by a new CLI `1b_segment_records.py`, slotting between the existing render step and the extraction step. Unit tests run on synthetic grid images with known coordinates; a final integration pass runs the 23-page sample PDF and produces a contact sheet for human QA.

**Tech Stack:** Python 3, OpenCV (`opencv-python-headless`), numpy, Pillow, pytest. Existing pipeline files: `config.py`, `1_render_pages.py`.

**Spec:** `docs/superpowers/specs/2026-08-17-record-segmentation-design.md`

## Global Constraints

- Canonical rectified size: `CANON_W, CANON_H = 2000, 1400`.
- Form geometry: `N_RECORDS = 5` records/page, `SUBROWS_PER_RECORD = 3`, `HEADER_FRAC = 0.13`, `SNAP_TOL = 0.02`.
- OCR input stays **grayscale** — never binarize emitted strips.
- Failed pages emit **no strips**: debug overlay + `"status": "needs_review"` manifest only.
- PHI safety: `.gitignore` must exclude `*.pdf`, `ODHFILESCANS*`, `_output/`, `_segments/` before the first commit.
- Run tests as `python -m pytest` from the repo root (so `segmentation.py` is importable).

---

### Task 1: Git init, requirements, PHI-safe gitignore

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`

**Interfaces:**
- Produces: a git repo all later tasks commit into; installed `cv2`, `numpy`, `pytest` importable by all later tasks.

- [ ] **Step 1: Write `.gitignore`**

```gitignore
# PHI-bearing source data and derived artifacts — never commit
*.pdf
ODHFILESCANS*
_output/
_segments/

# Python
__pycache__/
*.pyc
.venv/

# OS / editor cruft
.DS_Store
*.zip
```

- [ ] **Step 2: Write `requirements.txt`**

```text
opencv-python-headless>=4.9
numpy>=1.26
Pillow>=10.0
openpyxl>=3.1
pytest>=8.0
```

- [ ] **Step 3: Install dependencies and verify imports**

Run: `python -m pip install -r requirements.txt && python -c "import cv2, numpy; print(cv2.__version__)"`
Expected: prints an OpenCV version like `4.10.x`, no errors.

- [ ] **Step 4: Initialize repo and make the first commit**

```bash
git init
git add .gitignore requirements.txt config.py extraction_helpers.py \
    1_render_pages.py 2_enhance_regions.py 3_extract_records.py \
    4_store_records.py 5_verify_fields.py 6_audit_pick.py \
    7_audit_compare.py 8_progress_report.py feedback.md docs/
git commit -m "chore: init repo with existing pipeline, requirements, PHI-safe gitignore"
```

- [ ] **Step 5: Verify no PHI is tracked**

Run: `git ls-files | grep -iE '\.pdf$|ODHFILESCANS|_output|_segments' || echo CLEAN`
Expected: `CLEAN`

---

### Task 2: Render step changes (300 DPI PNG, no blind rotation)

**Files:**
- Modify: `config.py` (RENDER_DPI line, ~line 127; add SEGMENTS_DIR near OUTPUT_DIR, ~line 67)
- Modify: `1_render_pages.py` (pdftoppm args ~line 50; suffix handling ~line 56; delete rotation block lines 65–69)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `config.SEGMENTS_DIR` (str path, dir auto-created) used by Tasks 6–8; step-1 output files now named `<stem>_p<N>.png`.

- [ ] **Step 1: Update `config.py`**

Change `RENDER_DPI  = 200` to:

```python
RENDER_DPI  = 300
```

Below the `OUTPUT_DIR` block add:

```python
SEGMENTS_DIR = os.path.join(PROJECT_ROOT, "_segments")
os.makedirs(SEGMENTS_DIR, exist_ok=True)
```

- [ ] **Step 2: Update `1_render_pages.py`**

In `render_pdf_pages`, replace the pdftoppm call and suffix handling to use PNG:

```python
        subprocess.run(
            ["pdftoppm", "-png", "-r", str(dpi),
             "-f", str(p), "-l", str(p),
             pdf_path, prefix],
            check=True)

        matches = sorted(glob.glob(f"{prefix}*.png"))
        if not matches:
            print(f"  WARNING: no output for page {p}", file=sys.stderr)
            continue
        raw = matches[0]
        final = f"{prefix}.png"
        if raw != final:
            os.rename(raw, final)

        im = Image.open(final)
        print(f"  page {p}: {im.size[0]}x{im.size[1]}  → {final}")
        output_files.append(final)
```

(This deletes the `Auto-rotate portrait → landscape` block entirely — orientation is now owned by segmentation. Update the module docstring: change "Auto-rotates portrait pages to landscape (registers are landscape)." to "Orientation is handled downstream by 1b_segment_records.py." and change `JPEG` mentions to `PNG`.)

- [ ] **Step 3: Smoke-test against the sample PDF**

Run: `python 1_render_pages.py 20260319_053700_KAM_Stlhb.pdf 1 1 && python -c "from PIL import Image; im=Image.open('_output/20260319_053700_KAM_Stlhb_p1.png'); print(im.format, im.size)"`
Expected: `PNG (wwww, hhhh)` where the long side is ~3508 px (A4 at 300 DPI). No rotation applied.

- [ ] **Step 4: Commit**

```bash
git add config.py 1_render_pages.py
git commit -m "feat: render at 300dpi PNG; drop blind rotation; add SEGMENTS_DIR"
```

---

### Task 3: Synthetic-page test fixture + rectification

**Files:**
- Create: `segmentation.py`
- Create: `tests/conftest.py`
- Test: `tests/test_rectify.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces:
  - `segmentation.CANON_W, CANON_H, N_RECORDS, SUBROWS_PER_RECORD, HEADER_FRAC, SNAP_TOL` (module constants)
  - `find_table_quad(gray: np.ndarray) -> np.ndarray | None` — ordered 4×2 float32 (tl, tr, br, bl)
  - `order_corners(pts: np.ndarray) -> np.ndarray`
  - `rectify_page(img: np.ndarray, quad: np.ndarray, size=(CANON_W, CANON_H)) -> np.ndarray`
  - `ensure_upright(rect_gray: np.ndarray) -> tuple[np.ndarray, bool]` — (image, was_flipped)
  - test helpers `tests/conftest.py`: `draw_table(n_records=5) -> (np.ndarray, dict)`, `warp_page(img) -> np.ndarray`, constants `PAGE_W, PAGE_H, TABLE`

- [ ] **Step 1: Write the fixture helpers in `tests/conftest.py`**

```python
import numpy as np
import cv2

PAGE_W, PAGE_H = 1200, 850            # synthetic "photo" size
TABLE = (60, 40, 1140, 810)           # left, top, right, bottom of drawn table
HEADER_FRAC = 0.13
N_RECORDS = 5

def draw_table(n_records=N_RECORDS):
    """Grey page with a black grid: border, dense header band, 15 body rows,
    6 column lines. Returns (image, truth) where truth holds the known
    geometry for assertions."""
    img = np.full((PAGE_H, PAGE_W), 235, np.uint8)
    l, t, r, b = TABLE
    cv2.rectangle(img, (l, t), (r, b), 0, 3)
    header_bottom = t + int((b - t) * HEADER_FRAC)
    cv2.line(img, (l, header_bottom), (r, header_bottom), 0, 2)
    for x in range(l + 6, r - 6, 10):              # dense printed "text"
        cv2.line(img, (x, t + 6), (x, header_bottom - 6), 0, 1)
    ys = np.linspace(header_bottom, b, n_records * 3 + 1).astype(int)
    for y in ys[1:-1]:
        cv2.line(img, (l, int(y)), (r, int(y)), 0, 2)
    col_fracs = (0.08, 0.20, 0.35, 0.50, 0.70, 0.85)
    xs = [l + int((r - l) * fx) for fx in col_fracs]
    for x in xs:
        cv2.line(img, (x, t), (x, b), 0, 2)
    truth = {"table": TABLE, "header_bottom": header_bottom,
             "body_ys": [int(y) for y in ys], "col_xs": xs}
    return img, truth

WARP_DST = np.float32([[90, 70], [1130, 55], [1150, 800], [70, 815]])

def warp_page(img):
    """Apply a known mild perspective warp (simulates a photographed page)."""
    l, t, r, b = TABLE
    src = np.float32([[l, t], [r, t], [r, b], [l, b]])
    H = cv2.getPerspectiveTransform(src, WARP_DST)
    return cv2.warpPerspective(img, H, (PAGE_W, PAGE_H), borderValue=235)
```

- [ ] **Step 2: Write the failing tests in `tests/test_rectify.py`**

```python
import numpy as np
import cv2
from conftest import draw_table, warp_page, WARP_DST
from segmentation import (find_table_quad, rectify_page, ensure_upright,
                          CANON_W, CANON_H)


def test_find_table_quad_on_warped_page():
    img, _ = draw_table()
    warped = warp_page(img)
    quad = find_table_quad(warped)
    assert quad is not None and quad.shape == (4, 2)
    for found, expected in zip(quad, WARP_DST):
        assert np.linalg.norm(found - expected) < 12   # px tolerance


def test_find_table_quad_none_on_blank_page():
    blank = np.full((850, 1200), 235, np.uint8)
    assert find_table_quad(blank) is None


def test_rectify_maps_table_to_canonical_size():
    img, _ = draw_table()
    warped = warp_page(img)
    quad = find_table_quad(warped)
    rect = rectify_page(warped, quad)
    assert rect.shape[:2] == (CANON_H, CANON_W)
    # border line should now hug the edges: dark pixels near x=0 and x=W-1
    assert rect[:, :6].min() < 100 and rect[:, -6:].min() < 100


def test_ensure_upright_flips_header_to_top():
    img, _ = draw_table()
    quad = find_table_quad(img)
    rect = rectify_page(img, quad)
    up, flipped = ensure_upright(rect)
    assert not flipped
    upside_down = cv2.rotate(rect, cv2.ROTATE_180)
    up2, flipped2 = ensure_upright(upside_down)
    assert flipped2
    assert np.array_equal(up2, rect)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_rectify.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'segmentation'`

- [ ] **Step 4: Implement rectification in `segmentation.py`**

```python
"""
segmentation.py — OpenCV page rectification and per-record segmentation
for handwritten OPD register scans.

Pipeline per page:  rectify → clean → detect grid → emit record strips.
See docs/superpowers/specs/2026-08-17-record-segmentation-design.md.
"""

import os
import json

import cv2
import numpy as np

# Canonical rectified table geometry
CANON_W, CANON_H = 2000, 1400
N_RECORDS = 5
SUBROWS_PER_RECORD = 3
HEADER_FRAC = 0.13     # header band as fraction of table height
SNAP_TOL = 0.02        # boundary snap tolerance, fraction of table height


def order_corners(pts):
    """Order 4 points as tl, tr, br, bl (image coordinates)."""
    pts = pts.astype("float32")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype="float32")


def find_table_quad(gray):
    """Locate the printed table's outer border. Returns ordered 4x2 float32
    corners, or None if no plausible table is found."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 51, 10)
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < 0.30 * gray.shape[0] * gray.shape[1]:
        return None
    peri = cv2.arcLength(biggest, True)
    approx = cv2.approxPolyDP(biggest, 0.02 * peri, True)
    if len(approx) != 4:
        approx = cv2.boxPoints(cv2.minAreaRect(biggest))
    return order_corners(np.asarray(approx).reshape(4, 2))


def rectify_page(img, quad, size=(CANON_W, CANON_H)):
    """Perspective-warp *quad* onto the canonical landscape rectangle."""
    w, h = size
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    H = cv2.getPerspectiveTransform(quad.astype("float32"), dst)
    return cv2.warpPerspective(img, H, (w, h))


def _small_marks(gray):
    """Ink mask with long grid lines removed — leaves text and handwriting."""
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 31, 15)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (gray.shape[1] // 4, 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, gray.shape[0] // 4))
    long_lines = cv2.morphologyEx(thr, cv2.MORPH_OPEN, hk) | \
                 cv2.morphologyEx(thr, cv2.MORPH_OPEN, vk)
    return cv2.subtract(thr, long_lines)


def ensure_upright(rect_gray):
    """The dense printed header band must sit at the top. Returns
    (image, was_flipped)."""
    marks = _small_marks(rect_gray)
    band = int(rect_gray.shape[0] * 0.15)
    top, bottom = int(marks[:band].sum()), int(marks[-band:].sum())
    if bottom > top:
        return cv2.rotate(rect_gray, cv2.ROTATE_180), True
    return rect_gray, False
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_rectify.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add segmentation.py tests/conftest.py tests/test_rectify.py
git commit -m "feat: table detection, perspective rectification, 180-flip check"
```

---

### Task 4: Image cleanup (channel choice, illumination flattening, CLAHE)

**Files:**
- Modify: `segmentation.py` (append functions)
- Test: `tests/test_clean.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `best_channel(img: np.ndarray) -> np.ndarray` — grayscale (passes 2-D input through)
  - `flatten_illumination(gray: np.ndarray) -> np.ndarray`
  - `clean_page(img: np.ndarray) -> np.ndarray` — grayscale, shadow-free, CLAHE-boosted

- [ ] **Step 1: Write the failing tests in `tests/test_clean.py`**

```python
import numpy as np
import cv2
from segmentation import best_channel, flatten_illumination, clean_page


def _shaded(gray):
    """Multiply by a horizontal 0.55→1.0 illumination gradient."""
    grad = np.linspace(0.55, 1.0, gray.shape[1], dtype=np.float32)
    return (gray.astype(np.float32) * grad[None, :]).astype(np.uint8)


def test_flatten_removes_gradient():
    base = np.full((400, 600), 220, np.uint8)
    cv2.line(base, (50, 200), (550, 200), 30, 3)      # some "ink"
    shaded = _shaded(base)
    flat = flatten_illumination(shaded)
    bg = flat[:150]                                    # ink-free region
    assert float(bg.std()) < 6.0
    assert float(_shaded(base)[:150].std()) > 20.0     # gradient was real


def test_flatten_preserves_ink():
    base = np.full((400, 600), 220, np.uint8)
    cv2.line(base, (50, 200), (550, 200), 30, 3)
    flat = flatten_illumination(_shaded(base))
    assert flat[198:203, 300].min() < 140              # stroke still dark


def test_best_channel_picks_highest_contrast():
    b = np.full((100, 100), 200, np.uint8)             # flat blue channel
    g = np.full((100, 100), 200, np.uint8)
    r = np.full((100, 100), 230, np.uint8)
    cv2.line(r, (0, 50), (99, 50), 20, 3)              # ink visible in red
    bgr = cv2.merge([b, g, r])
    chosen = best_channel(bgr)
    assert np.array_equal(chosen, r)


def test_clean_page_returns_grayscale_same_size():
    bgr = cv2.merge([np.full((300, 500), v, np.uint8) for v in (180, 200, 230)])
    out = clean_page(bgr)
    assert out.ndim == 2 and out.shape == (300, 500)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_clean.py -v`
Expected: FAIL with `ImportError: cannot import name 'best_channel'`

- [ ] **Step 3: Append the implementation to `segmentation.py`**

```python
def best_channel(img):
    """Pick the color channel with the most ink/paper contrast (highest std).
    On the blue register paper this is typically the red channel."""
    if img.ndim == 2:
        return img
    return max(cv2.split(img), key=lambda c: float(c.std()))


def flatten_illumination(gray):
    """Divide by a median-blurred background estimate — removes shadows and
    lighting gradients from photographed pages without touching strokes."""
    bg = cv2.medianBlur(gray, 61)
    return cv2.divide(gray, bg, scale=255)


def clean_page(img):
    """Full cleanup: channel choice → flatten → CLAHE. Grayscale out."""
    flat = flatten_illumination(best_channel(img))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(flat)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_clean.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add segmentation.py tests/test_clean.py
git commit -m "feat: illumination flattening, channel selection, CLAHE cleanup"
```

---

### Task 5: Grid line detection and record grouping

**Files:**
- Modify: `segmentation.py` (append functions)
- Test: `tests/test_grid.py`

**Interfaces:**
- Consumes: `rectify_page`/fixture geometry from Task 3.
- Produces:
  - `detect_h_lines(gray, min_frac=0.5) -> list[int]` — y-centers of full-width lines
  - `detect_v_lines(gray, min_frac=0.5) -> list[int]` — x-centers of full-height lines
  - `group_records(h_lines: list[int], table_h: int, n_records=N_RECORDS, header_frac=HEADER_FRAC, snap_tol=SNAP_TOL) -> tuple[int, list[tuple[int,int]], list[str]]` — (header_bottom, [(y0, y1)] per record, warnings). A boundary with no detected line within tolerance falls back to its ideal position and appends a warning containing the string `"using ideal"`.

- [ ] **Step 1: Write the failing tests in `tests/test_grid.py`**

```python
import numpy as np
from conftest import draw_table
from segmentation import (find_table_quad, rectify_page, detect_h_lines,
                          detect_v_lines, group_records, CANON_H, N_RECORDS)


def _rectified():
    img, truth = draw_table()
    rect = rectify_page(img, find_table_quad(img))
    return rect, truth


def test_detect_h_lines_finds_all_body_rows():
    rect, truth = _rectified()
    ys = detect_h_lines(rect)
    # border top+bottom, header line, and 14 interior body lines ≈ 17 lines
    assert len(ys) >= N_RECORDS * 3 + 1


def test_detect_v_lines_finds_columns():
    rect, truth = _rectified()
    xs = detect_v_lines(rect)
    assert len(xs) >= len(truth["col_xs"])            # 6 columns + 2 borders


def test_group_records_returns_five_snapped_blocks():
    rect, _ = _rectified()
    h_lines = detect_h_lines(rect)
    header_bottom, records, warnings = group_records(h_lines, CANON_H)
    assert len(records) == N_RECORDS
    assert not [w for w in warnings if "using ideal" in w]
    assert records[0][0] == header_bottom
    for (a0, a1), (b0, b1) in zip(records, records[1:]):
        assert a1 == b0                                # contiguous blocks
    heights = [y1 - y0 for y0, y1 in records]
    assert max(heights) - min(heights) < 0.02 * CANON_H


def test_group_records_warns_without_lines():
    header_bottom, records, warnings = group_records([], CANON_H)
    assert len(records) == N_RECORDS
    assert any("using ideal" in w for w in warnings)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_grid.py -v`
Expected: FAIL with `ImportError: cannot import name 'detect_h_lines'`

- [ ] **Step 3: Append the implementation to `segmentation.py`**

```python
def _cluster(indices, gap=6):
    """Group consecutive pixel indices into line-center coordinates."""
    groups = []
    for i in indices:
        if groups and i - groups[-1][-1] <= gap:
            groups[-1].append(i)
        else:
            groups.append([i])
    return [int(round(np.mean(g))) for g in groups]


def _long_line_mask(gray, axis):
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 31, 15)
    if axis == "h":
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (gray.shape[1] // 6, 1))
    else:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, gray.shape[0] // 6))
    return cv2.morphologyEx(thr, cv2.MORPH_OPEN, k)


def detect_h_lines(gray, min_frac=0.5):
    """y-coordinates of horizontal lines spanning ≥ min_frac of the width."""
    mask = _long_line_mask(gray, "h")
    counts = (mask > 0).sum(axis=1)
    return _cluster(list(np.where(counts > min_frac * gray.shape[1])[0]))


def detect_v_lines(gray, min_frac=0.5):
    """x-coordinates of vertical lines spanning ≥ min_frac of the height."""
    mask = _long_line_mask(gray, "v")
    counts = (mask > 0).sum(axis=0)
    return _cluster(list(np.where(counts > min_frac * gray.shape[0])[0]))


def group_records(h_lines, table_h, n_records=N_RECORDS,
                  header_frac=HEADER_FRAC, snap_tol=SNAP_TOL):
    """Split the table body into record blocks.

    Ideal equal-split boundaries are snapped to the nearest detected line
    within tolerance; unsnappable boundaries keep the ideal position and
    produce a warning (the caller downgrades the page to needs_review).
    Returns (header_bottom, [(y0, y1), ...], warnings).
    """
    warnings = []
    tol = snap_tol * table_h

    def snap(ideal):
        nearest = min(h_lines, key=lambda y: abs(y - ideal)) if h_lines else None
        if nearest is not None and abs(nearest - ideal) <= tol:
            return int(nearest)
        warnings.append(
            f"no grid line within {tol:.0f}px of y={ideal:.0f}; using ideal")
        return int(round(ideal))

    header_bottom = snap(table_h * header_frac)
    body_bottom = snap(table_h - 1)
    bounds = np.linspace(header_bottom, body_bottom, n_records + 1)
    ys = [header_bottom] + [snap(b) for b in bounds[1:-1]] + [body_bottom]
    records = [(ys[i], ys[i + 1]) for i in range(n_records)]
    return header_bottom, records, warnings
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_grid.py -v`
Expected: 4 passed

- [ ] **Step 5: Run the whole suite to catch regressions**

Run: `python -m pytest -v`
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add segmentation.py tests/test_grid.py
git commit -m "feat: grid line detection and record-block grouping with snap"
```

---

### Task 6: Strip emission, debug overlay, manifest, `segment_page`

**Files:**
- Modify: `segmentation.py` (append functions)
- Test: `tests/test_segment_page.py`

**Interfaces:**
- Consumes: everything from Tasks 3–5.
- Produces:
  - `emit_record_strips(rect_gray, header_bottom: int, records, out_dir: str, stem: str) -> list[dict]` — writes `<stem>_rec<K>.png`, returns manifest record entries `{"index", "y0", "y1", "strip"}`
  - `save_debug_overlay(rect_gray, header_bottom, records, col_x, path)`
  - `segment_page(image_path: str, out_dir: str) -> dict` — full per-page pipeline; writes `<stem>.json`, `<stem>_debug.jpg`, and on success `<stem>_rec1..N.png` + `<stem>_full.png`. Manifest keys: `source_image, stem, status ("ok"|"needs_review"), canonical_size, header_band, records, col_x, warnings`.

- [ ] **Step 1: Write the failing tests in `tests/test_segment_page.py`**

```python
import os
import json
import cv2
import numpy as np
from conftest import draw_table, warp_page
from segmentation import segment_page, N_RECORDS, CANON_W


def _page_on_disk(tmp_path, transform=None):
    img, _ = draw_table()
    if transform:
        img = transform(img)
    path = str(tmp_path / "reg_p1.png")
    cv2.imwrite(path, img)
    return path


def test_segment_page_ok(tmp_path):
    path = _page_on_disk(tmp_path, warp_page)
    out = str(tmp_path / "seg")
    m = segment_page(path, out)
    assert m["status"] == "ok"
    assert len(m["records"]) == N_RECORDS
    for rec in m["records"]:
        strip = cv2.imread(os.path.join(out, rec["strip"]),
                           cv2.IMREAD_GRAYSCALE)
        assert strip is not None
        assert strip.shape[1] == CANON_W
        # strip = header band + record block
        assert strip.shape[0] == m["header_band"][1] + (rec["y1"] - rec["y0"])
    assert os.path.isfile(os.path.join(out, "reg_p1_full.png"))
    assert os.path.isfile(os.path.join(out, "reg_p1_debug.jpg"))
    with open(os.path.join(out, "reg_p1.json")) as f:
        assert json.load(f)["stem"] == "reg_p1"
    assert len(m["col_x"]) >= 6


def test_segment_page_handles_portrait_input(tmp_path):
    rot = lambda im: cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
    m = segment_page(_page_on_disk(tmp_path, rot), str(tmp_path / "seg"))
    assert m["status"] == "ok"
    assert len(m["records"]) == N_RECORDS


def test_segment_page_needs_review_on_blank(tmp_path):
    blank = str(tmp_path / "blank_p1.png")
    cv2.imwrite(blank, np.full((850, 1200), 235, np.uint8))
    out = str(tmp_path / "seg")
    m = segment_page(blank, out)
    assert m["status"] == "needs_review"
    assert m["records"] == []
    assert not [f for f in os.listdir(out) if "_rec" in f]   # no strips
    assert os.path.isfile(os.path.join(out, "blank_p1.json"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_segment_page.py -v`
Expected: FAIL with `ImportError: cannot import name 'segment_page'`

- [ ] **Step 3: Append the implementation to `segmentation.py`**

```python
def emit_record_strips(rect_gray, header_bottom, records, out_dir, stem):
    """Write one PNG per record: printed header band + the record's rows."""
    header = rect_gray[0:header_bottom]
    entries = []
    for i, (y0, y1) in enumerate(records, start=1):
        strip = np.vstack([header, rect_gray[y0:y1]])
        name = f"{stem}_rec{i}.png"
        cv2.imwrite(os.path.join(out_dir, name), strip)
        entries.append({"index": i, "y0": int(y0), "y1": int(y1),
                        "strip": name})
    return entries


def save_debug_overlay(rect_gray, header_bottom, records, col_x, path):
    vis = cv2.cvtColor(rect_gray, cv2.COLOR_GRAY2BGR)
    w = vis.shape[1]
    cv2.line(vis, (0, header_bottom), (w, header_bottom), (255, 0, 0), 3)
    for y0, y1 in records:
        cv2.rectangle(vis, (2, y0), (w - 3, y1), (0, 0, 255), 2)
    for x in col_x:
        cv2.line(vis, (x, 0), (x, vis.shape[0]), (0, 180, 0), 1)
    cv2.imwrite(path, vis)


def _write_manifest(manifest, out_dir):
    with open(os.path.join(out_dir, manifest["stem"] + ".json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def segment_page(image_path, out_dir):
    """Rectify, clean, and segment one rendered page image.

    Writes strips + debug overlay + manifest into out_dir. On any failure
    the page gets status "needs_review" and NO strips are emitted.
    """
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    manifest = {"source_image": image_path, "stem": stem,
                "status": "needs_review",
                "canonical_size": [CANON_W, CANON_H],
                "header_band": None, "records": [], "col_x": [],
                "warnings": []}

    img = cv2.imread(image_path)
    if img is None:
        manifest["warnings"].append("could not read image")
        return _write_manifest(manifest, out_dir)
    if img.shape[0] > img.shape[1]:            # portrait → landscape
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    quad = find_table_quad(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if quad is None:
        manifest["warnings"].append("table border not found")
        cv2.imwrite(os.path.join(out_dir, f"{stem}_debug.jpg"), img)
        return _write_manifest(manifest, out_dir)

    rect_gray = clean_page(rectify_page(img, quad))
    rect_gray, flipped = ensure_upright(rect_gray)
    if flipped:
        manifest["warnings"].append("rotated 180 (header was at bottom)")

    h_lines = detect_h_lines(rect_gray)
    col_x = detect_v_lines(rect_gray)
    header_bottom, records, grp_warnings = group_records(h_lines, CANON_H)
    manifest["warnings"] += grp_warnings
    manifest["header_band"] = [0, int(header_bottom)]
    manifest["col_x"] = [int(x) for x in col_x]

    if not any("using ideal" in w for w in grp_warnings):
        manifest["status"] = "ok"
        manifest["records"] = emit_record_strips(
            rect_gray, header_bottom, records, out_dir, stem)
        cv2.imwrite(os.path.join(out_dir, f"{stem}_full.png"), rect_gray)

    save_debug_overlay(rect_gray, header_bottom, records, col_x,
                       os.path.join(out_dir, f"{stem}_debug.jpg"))
    return _write_manifest(manifest, out_dir)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_segment_page.py -v`
Expected: 3 passed

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest -v`
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add segmentation.py tests/test_segment_page.py
git commit -m "feat: segment_page orchestration, strip emission, manifest, debug overlay"
```

---

### Task 7: CLI `1b_segment_records.py` with contact sheet

**Files:**
- Create: `1b_segment_records.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `segmentation.segment_page`, `config.SEGMENTS_DIR`.
- Produces: CLI — `python 1b_segment_records.py <img> [...] [--glob PATTERN] [--out DIR] [--contact-sheet]`. Per page output dir: `<out>/<stem>/`. Also `build_contact_sheet(manifests: list[dict], base_dir: str, out_path: str)`.

- [ ] **Step 1: Write the failing test in `tests/test_cli.py`**

```python
import os
import subprocess
import sys
import cv2
from conftest import draw_table, warp_page

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_cli_segments_and_builds_contact_sheet(tmp_path):
    img, _ = draw_table()
    page = str(tmp_path / "reg_p1.png")
    cv2.imwrite(page, warp_page(img))
    out = str(tmp_path / "segments")

    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "1b_segment_records.py"),
         page, "--out", out, "--contact-sheet"],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout
    assert os.path.isfile(os.path.join(out, "reg_p1", "reg_p1_rec1.png"))
    assert os.path.isfile(os.path.join(out, "contact_sheet.jpg"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli.py -v`
Expected: FAIL (script does not exist → non-zero returncode)

- [ ] **Step 3: Write `1b_segment_records.py`**

```python
#!/usr/bin/env python3
"""
1b_segment_records.py — Rectify rendered page images and crop per-record
strips for the extraction step.

Usage:
    python 1b_segment_records.py _output/foo_p1.png [_output/foo_p2.png ...]
    python 1b_segment_records.py --glob "_output/foo_p*.png" --contact-sheet
    python 1b_segment_records.py page.png --out /tmp/segments

Each page writes to  <out>/<stem>/ :
    <stem>_rec1..5.png   record strips (header band + 3 sub-rows)
    <stem>_full.png      rectified cleaned page
    <stem>_debug.jpg     detection overlay for QA
    <stem>.json          manifest (record y-ranges, column x-boundaries)

Pages that fail detection get status "needs_review" and NO strips.
"""

import os
import sys
import glob as globmod
import argparse

import cv2
import numpy as np

from config import SEGMENTS_DIR
from segmentation import segment_page


def build_contact_sheet(manifests, base_dir, out_path, cols=4, thumb_w=500):
    """Tile every page's debug overlay into one JPEG, bordered green (ok)
    or red (needs_review)."""
    thumbs = []
    for m in manifests:
        p = os.path.join(base_dir, m["stem"], m["stem"] + "_debug.jpg")
        img = cv2.imread(p)
        if img is None:
            continue
        h = max(1, int(img.shape[0] * thumb_w / img.shape[1]))
        t = cv2.resize(img, (thumb_w, h))
        color = (0, 200, 0) if m["status"] == "ok" else (0, 0, 255)
        cv2.rectangle(t, (0, 0), (thumb_w - 1, h - 1), color, 8)
        cv2.putText(t, m["stem"], (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        thumbs.append(t)
    if not thumbs:
        return
    h = max(t.shape[0] for t in thumbs)
    thumbs = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0,
                                 cv2.BORDER_CONSTANT) for t in thumbs]
    blank = np.zeros_like(thumbs[0])
    while len(thumbs) % cols:
        thumbs.append(blank.copy())
    rows = [np.hstack(thumbs[i:i + cols]) for i in range(0, len(thumbs), cols)]
    cv2.imwrite(out_path, np.vstack(rows))


def main():
    p = argparse.ArgumentParser(
        description="Rectify pages and crop per-record strips.")
    p.add_argument("images", nargs="*", help="Rendered page images")
    p.add_argument("--glob", default=None, help="Glob pattern for pages")
    p.add_argument("--out", default=SEGMENTS_DIR,
                   help=f"Output base dir (default {SEGMENTS_DIR})")
    p.add_argument("--contact-sheet", action="store_true",
                   help="Tile debug overlays into <out>/contact_sheet.jpg")
    args = p.parse_args()

    paths = list(args.images)
    if args.glob:
        paths += sorted(globmod.glob(args.glob))
    if not paths:
        p.error("no input images (pass paths or --glob)")

    manifests = []
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        m = segment_page(path, os.path.join(args.out, stem))
        manifests.append(m)
        note = "; ".join(m["warnings"]) if m["warnings"] else ""
        print(f"  {stem}: {m['status']}  "
              f"({len(m['records'])} records)  {note}")

    ok = sum(1 for m in manifests if m["status"] == "ok")
    bad = len(manifests) - ok
    print(f"\n{ok} ok, {bad} needs_review → {args.out}/")

    if args.contact_sheet:
        sheet = os.path.join(args.out, "contact_sheet.jpg")
        build_contact_sheet(manifests, args.out, sheet)
        print(f"Contact sheet → {sheet}")

    sys.exit(0 if bad == 0 else 2)


if __name__ == "__main__":
    main()
```

Note: exit code 2 signals `needs_review` pages exist — the CLI test's page succeeds, so it expects 0.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cli.py -v`
Expected: 1 passed

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest -v`
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add 1b_segment_records.py tests/test_cli.py
git commit -m "feat: 1b_segment_records CLI with contact-sheet QA output"
```

---

### Task 8: Integration QA on the 23-page sample PDF

**Files:**
- Modify (only if tuning is needed): `segmentation.py` constants (`HEADER_FRAC`, `SNAP_TOL`, thresholds)

**Interfaces:**
- Consumes: Tasks 2–7 complete.
- Produces: a contact sheet verifying real-world behavior; tuned constants committed.

- [ ] **Step 1: Render all 23 sample pages**

Run: `python 1_render_pages.py 20260319_053700_KAM_Stlhb.pdf`
Expected: `Rendered 23 page(s) → _output/`

- [ ] **Step 2: Segment all pages with contact sheet**

Run: `python 1b_segment_records.py --glob "_output/20260319_053700_KAM_Stlhb_p*.png" --contact-sheet`
Expected: per-page lines mostly `ok (5 records)`; contact sheet written to `_segments/contact_sheet.jpg`.

- [ ] **Step 3: Inspect the contact sheet (human/agent eyeball)**

Open `_segments/contact_sheet.jpg`. For every page check: header line (blue) sits under the printed header band; the 5 red boxes each enclose exactly one record (3 sub-rows); green column lines align with printed columns; no page is upside down or mirrored.

- [ ] **Step 4: Tune constants for any failing pages**

For each `needs_review` or visually-wrong page, inspect its `_debug.jpg` and adjust in `segmentation.py`: `HEADER_FRAC` (header line landing inside header text → raise/lower), `SNAP_TOL` (boundaries not snapping on slightly warped pages → raise toward 0.03), `min_frac` in line detection (faint grid lines missed → lower toward 0.4), the `0.30` area floor in `find_table_quad` (table smaller in frame → lower). Re-run Step 2 after each change. **Do not** change emitted-file naming or the manifest schema — Tasks 6–7 tests pin those.

- [ ] **Step 5: Re-run the full unit suite after tuning**

Run: `python -m pytest -v`
Expected: all tests still pass

- [ ] **Step 6: Record the outcome and commit**

Add the final per-page result (N ok / N needs_review, and why for any residual failures) to the bottom of the spec doc under a new `## Integration results (2026-08-17)` heading.

```bash
git add segmentation.py docs/superpowers/specs/2026-08-17-record-segmentation-design.md
git commit -m "feat: tune segmentation constants against 23-page sample; record QA results"
```

---

## Self-review notes

- Spec coverage: rectification (T3), cleanup (T4), grid detection/grouping (T5), strips + manifest + failure behavior (T6), CLI + contact sheet (T7), render changes + SEGMENTS_DIR (T2), git/requirements/PHI gitignore (T1), sample-PDF acceptance (T8). Cell-level crops are intentionally deferred; the manifest's `col_x` (T6) is the enabling contract, per spec.
- Type consistency: `segment_page(image_path, out_dir) -> dict` is used identically in T6 tests, T7 CLI; `group_records` return triple matches between T5 and T6; manifest keys match the spec schema.
- Known judgment call: `ensure_upright` compares small-mark density top vs bottom; if real pages defeat it (heavy handwriting in the last record), the fallback cue is the page-number box in the top-right corner — note this in T8 results if encountered.
