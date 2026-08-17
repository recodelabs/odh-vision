#!/usr/bin/env python3
"""
7_audit_compare.py  —  Log stored-vs-reread comparisons and report accuracy.

Usage:
    python 7_audit_compare.py comparisons.json [--pdf "Apr week 2.pdf"]

comparisons.json format:
[
  {"pdf": "Apr week 2.pdf", "page": 1, "record_no": 1,
   "field": "Patient name", "stored": "SAM WANJI",
   "reread": "SAM WANJI", "result": "MATCH"},
  ...
]

result must be:  MATCH | MISMATCH | UNCLEAR
"""

import sys, os, csv, json, argparse, datetime
from config import AUDIT_LOG


def log_comparisons(comparisons, pdf_name=None):
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    header = ["timestamp", "source_pdf", "page", "record_no",
              "field", "stored_value", "reread_value", "result"]

    need_header = not os.path.exists(AUDIT_LOG) or os.path.getsize(AUDIT_LOG) == 0

    with open(AUDIT_LOG, "a", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        if need_header:
            w.writerow(header)
        for c in comparisons:
            w.writerow([
                ts,
                c.get("pdf", pdf_name or ""),
                c["page"], c["record_no"], c["field"],
                c.get("stored", ""), c.get("reread", ""),
                c["result"],
            ])
    return len(comparisons)


def run_stats(comparisons):
    m  = sum(1 for c in comparisons if c["result"] == "MATCH")
    mm = sum(1 for c in comparisons if c["result"] == "MISMATCH")
    un = sum(1 for c in comparisons if c["result"] == "UNCLEAR")
    judged = m + mm
    return {"match": m, "mismatch": mm, "unclear": un,
            "accuracy": round(m / judged * 100, 1) if judged else 0}


def cumulative_stats():
    if not os.path.exists(AUDIT_LOG):
        return {"rows": 0, "judged": 0, "accuracy": 0}
    cm = cmm = total = 0
    with open(AUDIT_LOG, newline="") as f:
        for row in csv.DictReader(f):
            total += 1
            if row["result"] == "MATCH":    cm += 1
            elif row["result"] == "MISMATCH": cmm += 1
    judged = cm + cmm
    return {"rows": total, "judged": judged,
            "accuracy": round(cm / judged * 100, 1) if judged else 0}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Log audit comparisons.")
    p.add_argument("json_file")
    p.add_argument("--pdf", default=None)
    args = p.parse_args()

    if not os.path.isfile(args.json_file):
        sys.exit(f"ERROR: {args.json_file} not found")

    with open(args.json_file) as f:
        comparisons = json.load(f)

    log_comparisons(comparisons, pdf_name=args.pdf)

    rs = run_stats(comparisons)
    print(f"THIS RUN:   MATCH={rs['match']}  MISMATCH={rs['mismatch']}  "
          f"UNCLEAR={rs['unclear']}  accuracy={rs['accuracy']}%")

    cs = cumulative_stats()
    print(f"CUMULATIVE: rows={cs['rows']}  judged={cs['judged']}  "
          f"accuracy={cs['accuracy']}%")
