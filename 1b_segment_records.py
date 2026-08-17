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
