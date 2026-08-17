#!/usr/bin/env python3
"""
6_audit_pick.py  —  Pick random PDF + pages for quality auditing.

Usage:
    python 6_audit_pick.py [--pages N]

Outputs JSON: {"pdf": "...", "sheet": "...", "pages": [1, 2, 3]}

Selection is weighted toward high-confidence PDFs — we want to verify
the ones we trust most, since errors there are the hardest to catch.
"""

import sys, json, time, random, argparse

from config import MASTER_XLSX, HEADERS, FLAG_COLORS
import openpyxl
from extraction_helpers import region_sheets, cell_rgb


def pick_audit_target(wb, n_pages=3):
    groups = {}
    for ws in region_sheets(wb):
        for row in ws.iter_rows(min_row=2):
            pdf = row[2].value if len(row) > 2 else None
            page = row[3].value if len(row) > 3 else None
            if not pdf or page is None:
                continue
            key = (str(pdf), page)
            g = groups.setdefault(key, {"sheet": ws.title, "conf": 0, "tot": 0})
            for ci in range(min(len(row), len(HEADERS))):
                rgb = cell_rgb(row[ci])
                g["tot"] += 1
                if not (rgb and rgb in FLAG_COLORS):
                    g["conf"] += 1

    if not groups:
        return {"pdf": None, "sheet": None, "pages": []}

    pdfs = {}
    for (pdf, page), g in groups.items():
        p = pdfs.setdefault(pdf, {"pages": [], "conf": 0, "tot": 0})
        p["pages"].append(page)
        p["conf"] += g["conf"]
        p["tot"] += g["tot"]

    random.seed(int(time.time()))
    pdf_list = list(pdfs.keys())
    weights = [(pdfs[p]["conf"] / pdfs[p]["tot"] if pdfs[p]["tot"] else 0) + 0.05
               for p in pdf_list]
    chosen_pdf = random.choices(pdf_list, weights=weights, k=1)[0]

    pages = sorted(set(pdfs[chosen_pdf]["pages"]))
    pw = [(groups[(chosen_pdf, pg)]["conf"] / groups[(chosen_pdf, pg)]["tot"]
           if groups[(chosen_pdf, pg)]["tot"] else 0) + 0.05
          for pg in pages]

    k = min(n_pages, len(pages))
    chosen = []
    pool = list(zip(pages, pw))
    for _ in range(k):
        ps, ws_ = zip(*pool)
        pick = random.choices(list(ps), weights=list(ws_), k=1)[0]
        chosen.append(pick)
        pool = [(p, w) for p, w in pool if p != pick]
    chosen.sort()

    sheet = groups[(chosen_pdf, chosen[0])]["sheet"]
    return {"pdf": chosen_pdf, "sheet": sheet, "pages": chosen}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Pick random pages for audit.")
    p.add_argument("--pages", type=int, default=3)
    args = p.parse_args()

    wb = openpyxl.load_workbook(MASTER_XLSX, data_only=True)
    result = pick_audit_target(wb, n_pages=args.pages)
    wb.close()
    print(json.dumps(result, indent=2))
