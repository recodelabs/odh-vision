#!/usr/bin/env python3
"""
4_store_records.py  —  Save extracted records to the master workbook.

Usage:
    python 4_store_records.py records.json --pdf "Apr week 2.pdf" --finished
    python 4_store_records.py records.json --pdf "file.pdf" --last-page 5 --total-pages 12

Steps:
    1. Loads the JSON file (array of record dicts from step 3).
    2. Appends to the correct region sheet(s).
    3. Updates the Progress sheet.
    4. Saves + backs up the workbook.
"""

import sys, os, json, argparse, datetime

from extraction_helpers import master_workbook, append_records, update_progress
from config import RESERVED_SHEETS


def store_records(json_path, pdf_name, finished=False,
                  last_page=None, total_pages=None):
    with open(json_path) as f:
        rows = json.load(f)
    if not rows:
        print("No records to store.")
        return

    date_str = datetime.date.today().isoformat()

    with master_workbook() as wb:
        n = append_records(wb, rows)

        existing = sum(
            1 for sn in wb.sheetnames if sn not in RESERVED_SHEETS
            for r in wb[sn].iter_rows(min_row=2, values_only=True)
            if len(r) > 2 and r[2] == pdf_name)

        if finished:
            note = f"Completed {date_str}."
            ext = "yes"
        elif last_page and total_pages:
            note = f"Partial through page {last_page}/{total_pages} {date_str}."
            ext = "partial"
        else:
            note = f"Updated {date_str}."
            ext = "partial"

        update_progress(wb, source_pdf=pdf_name,
                        extracted=ext, records_count=existing, notes=note)

    print(f"Stored {n} records for '{pdf_name}'.  Total: {existing}.  Status: {ext}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Store records into master xlsx.")
    p.add_argument("json_file")
    p.add_argument("--pdf", required=True)
    p.add_argument("--finished", action="store_true")
    p.add_argument("--last-page", type=int, default=None)
    p.add_argument("--total-pages", type=int, default=None)
    args = p.parse_args()

    if not os.path.isfile(args.json_file):
        sys.exit(f"ERROR: {args.json_file} not found")

    store_records(args.json_file, args.pdf,
                  finished=args.finished,
                  last_page=args.last_page,
                  total_pages=args.total_pages)
