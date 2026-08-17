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
