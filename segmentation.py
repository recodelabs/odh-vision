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
