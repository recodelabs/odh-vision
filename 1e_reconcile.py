#!/usr/bin/env python3
"""
1e_reconcile.py — Per-strip extractions → per-patient records.

Usage:
    python 1e_reconcile.py --all                       # gemini-3.7-flash
    python 1e_reconcile.py reg_p1 reg_p2 --no-repair
    python 1e_reconcile.py --all --force --center "Kameno" --year 2026

Boundary-repair re-reads call Gemini (credentials from .env) and are built
lazily — pages needing no repair never touch the network. --no-repair
disables re-reads entirely (clip signatures become review flags).
Output: _reconciled/<model>/<stem>.json. Exit 0 clean, 2 review queue
non-empty, 1 errors.
"""

import argparse
import glob as globmod
import os
import sys

from config import SEGMENTS_DIR, EXTRACTIONS_DIR, RECONCILED_DIR
from reconciliation import reconcile_page


def discover_stems(extractions_dir, model):
    d = os.path.join(extractions_dir, model)
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in globmod.glob(os.path.join(d, "*.json"))
                  if not os.path.basename(p).startswith("compare_"))


def _make_extract_fn(model, context_unused):
    """Lazy Gemini-backed repair re-reader; client built on first call."""
    state = {"client": None}

    def extract_fn(image_path, record_index, context):
        if state["client"] is None:
            from extraction import make_client
            state["client"] = make_client()
        from extraction import extract_strip
        return extract_strip(state["client"], model, image_path,
                             record_index, context=context)

    return extract_fn


def main(argv=None):
    p = argparse.ArgumentParser(description="Reconcile strips into patients.")
    p.add_argument("stems", nargs="*")
    p.add_argument("--all", action="store_true")
    p.add_argument("--model", default="gemini-3.7-flash")
    p.add_argument("--no-repair", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--center", default="")
    p.add_argument("--year", default="")
    p.add_argument("--segments-dir", default=SEGMENTS_DIR)
    p.add_argument("--extractions-dir", default=EXTRACTIONS_DIR)
    p.add_argument("--out", default=RECONCILED_DIR)
    args = p.parse_args(argv)

    stems = list(args.stems)
    if args.all:
        stems += [s for s in discover_stems(args.extractions_dir, args.model)
                  if s not in stems]
    if not stems:
        p.error("no stems given (pass stems or --all)")

    context = ", ".join(x for x in
                        ([f"Center: {args.center}"] if args.center else []) +
                        ([f"Year: {args.year}"] if args.year else []))
    extract_fn = None if args.no_repair else _make_extract_fn(args.model,
                                                              context)

    total_patients = reviews = errors = repairs = 0
    rcost = 0.0
    for stem in stems:
        try:
            r = reconcile_page(stem, args.model,
                               segments_dir=args.segments_dir,
                               extractions_dir=args.extractions_dir,
                               out_base=args.out, extract_fn=extract_fn,
                               force=args.force, context=context)
        except Exception as err:
            errors += 1
            print(f"  {stem}: ERROR {err}")
            continue
        if "refused" in r:
            print(f"  {stem}: refused ({r['refused']})")
            continue
        n_rev = sum(1 for pt in r["patients"] if pt["review"])
        if r["page_checks"].get("review"):
            n_rev += 1
        if r.get("skipped"):
            reviews += n_rev
            total_patients += len(r["patients"])
            print(f"  {stem}: skipped (existing; use --force) - "
                  f"{len(r['patients'])} patients, {n_rev} review")
            continue
        n_merged = sum(1 for pt in r["patients"] if pt["merged_from"])
        reviews += n_rev
        repairs += r["repair_usage"]["calls"]
        rcost += r["repair_usage"].get("est_cost_usd", 0.0)
        total_patients += len(r["patients"])
        print(f"  {stem}: {len(r['patients'])} patients, {n_merged} merged, "
              f"{r['repair_usage']['calls']} repaired, {n_rev} review, "
              f"seq={r['page_checks']['record_no_sequence']}")

    print(f"\n[{args.model}] {total_patients} patients, {repairs} repair "
          f"re-reads (est ${rcost:.4f}), {reviews} review, {errors} errors")
    if errors:
        return 1
    return 2 if reviews else 0


if __name__ == "__main__":
    sys.exit(main())
