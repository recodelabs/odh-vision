#!/usr/bin/env python3
"""
1d_compare_models.py — Field-level agreement between two models' extractions.

Usage:
    python 1d_compare_models.py --models gemini-3.5-flash-lite gemini-3.7-flash

Local-only (no API). Prints agreement stats (worst fields first) and writes
a disagreements CSV for human adjudication.
"""

import argparse
import csv
import glob as globmod
import json
import os
import sys

from config import EXTRACTIONS_DIR
from extraction import FIELD_NAMES


def norm(s: str) -> str:
    return " ".join(str(s).split()).casefold()


def compare_extractions(dir_a, dir_b):
    per_field = {n: {"agree": 0, "total": 0} for n in FIELD_NAMES}
    disagreements = []
    n_records = both_empty = 0

    for path_a in sorted(globmod.glob(os.path.join(dir_a, "*.json"))):
        stem = os.path.splitext(os.path.basename(path_a))[0]
        if stem.startswith("compare_"):
            continue
        path_b = os.path.join(dir_b, f"{stem}.json")
        if not os.path.isfile(path_b):
            continue
        recs_a = json.load(open(path_a))["records"]
        recs_b = json.load(open(path_b))["records"]
        for key in sorted(set(recs_a) & set(recs_b)):
            n_records += 1
            fa, fb = recs_a[key]["fields"], recs_b[key]["fields"]
            for name in FIELD_NAMES:
                va, vb = fa[name]["value"], fb[name]["value"]
                if va == "" and vb == "":
                    both_empty += 1
                    continue
                per_field[name]["total"] += 1
                if norm(va) == norm(vb):
                    per_field[name]["agree"] += 1
                else:
                    disagreements.append({
                        "stem": stem, "record": key, "field": name,
                        "a_value": va, "a_conf": fa[name]["confidence"],
                        "b_value": vb, "b_conf": fb[name]["confidence"]})

    for stats in per_field.values():
        stats["rate"] = (round(stats["agree"] / stats["total"], 3)
                         if stats["total"] else 1.0)
    compared = sum(s["total"] for s in per_field.values())
    agreed = sum(s["agree"] for s in per_field.values())
    return {"n_records": n_records, "n_compared_fields": compared,
            "agreement_rate": round(agreed / compared, 3) if compared else 1.0,
            "both_empty": both_empty, "per_field": per_field,
            "disagreements": disagreements}


def write_csv(disagreements, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "record", "field",
                                          "a_value", "a_conf",
                                          "b_value", "b_conf"])
        w.writeheader()
        w.writerows(disagreements)


def main(argv=None):
    p = argparse.ArgumentParser(description="Compare two models' extractions.")
    p.add_argument("--models", nargs=2, required=True, metavar=("A", "B"))
    p.add_argument("--base", default=EXTRACTIONS_DIR)
    p.add_argument("--out-csv", default=None)
    args = p.parse_args(argv)

    a, b = args.models
    result = compare_extractions(os.path.join(args.base, a),
                                 os.path.join(args.base, b))
    print(f"Compared {result['n_records']} records, "
          f"{result['n_compared_fields']} non-empty field pairs "
          f"(+{result['both_empty']} both-empty)")
    print(f"Overall agreement: {result['agreement_rate']:.1%}\n")
    print(f"{'field':<22} {'agree':>6} {'total':>6} {'rate':>7}")
    ranked = sorted(result["per_field"].items(), key=lambda kv: kv[1]["rate"])
    for name, s in ranked:
        if s["total"]:
            print(f"{name:<22} {s['agree']:>6} {s['total']:>6} {s['rate']:>6.1%}")

    out_csv = args.out_csv or os.path.join(args.base, f"compare_{a}__{b}.csv")
    write_csv(result["disagreements"], out_csv)
    print(f"\n{len(result['disagreements'])} disagreements → {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
