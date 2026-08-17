#!/usr/bin/env python3
"""
1_render_pages.py  —  Convert PDF pages to JPEG images.

Usage:
    python 1_render_pages.py <pdf_path> [page_start] [page_end]
    python 1_render_pages.py "ODHFILESCANS_optimized/Masaka/Kabanda/Apr week 2.pdf"
    python 1_render_pages.py "path/to/file.pdf" 3 7

Steps:
    1. Runs pdftoppm to render each page as JPEG at configured DPI.
    2. Auto-rotates portrait pages to landscape (registers are landscape).
    3. Saves to OUTPUT_DIR as  <stem>_p<N>.jpg

Requires: poppler-utils (pdftoppm, pdfinfo), Pillow.
"""

import sys, os, subprocess, glob
from PIL import Image
from config import RENDER_DPI, OUTPUT_DIR


def render_pdf_pages(pdf_path: str,
                     page_start: int = 1,
                     page_end: int = None,
                     dpi: int = RENDER_DPI,
                     out_dir: str = OUTPUT_DIR) -> list:
    """Render pages to JPEG. Returns list of output file paths."""

    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(pdf_path))[0]

    # Detect total pages if not specified
    if page_end is None:
        info = subprocess.run(
            ["pdfinfo", pdf_path], capture_output=True, text=True)
        for line in info.stdout.splitlines():
            if line.startswith("Pages:"):
                page_end = int(line.split(":")[1].strip())
                break
        if page_end is None:
            page_end = page_start

    output_files = []
    for p in range(page_start, page_end + 1):
        prefix = os.path.join(out_dir, f"{stem}_p{p}")
        subprocess.run(
            ["pdftoppm", "-jpeg", "-r", str(dpi),
             "-f", str(p), "-l", str(p),
             pdf_path, prefix],
            check=True)

        # pdftoppm adds a suffix like -01.jpg — rename cleanly
        matches = sorted(glob.glob(f"{prefix}*.jpg"))
        if not matches:
            print(f"  WARNING: no output for page {p}", file=sys.stderr)
            continue
        raw = matches[0]
        final = f"{prefix}.jpg"
        if raw != final:
            os.rename(raw, final)

        # Auto-rotate portrait → landscape
        im = Image.open(final)
        if im.height > im.width:
            im = im.rotate(90, expand=True)
            im.save(final)

        print(f"  page {p}: {im.size[0]}x{im.size[1]}  → {final}")
        output_files.append(final)

    return output_files


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 1_render_pages.py <pdf> [start] [end]")
        sys.exit(1)
    pdf   = sys.argv[1]
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    end   = int(sys.argv[3]) if len(sys.argv) > 3 else None
    files = render_pdf_pages(pdf, start, end)
    print(f"\nRendered {len(files)} page(s) → {OUTPUT_DIR}/")
