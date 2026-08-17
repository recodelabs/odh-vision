#!/usr/bin/env python3
"""
1c_extract_strips.py — Extract segmented record strips to JSON via Gemini.

Usage:
    python 1c_extract_strips.py --all --dry-run
    python 1c_extract_strips.py reg_p1 reg_p2 --model gemini-3.7-flash
    python 1c_extract_strips.py --all --limit 10 --center "Kameno" --year 2026

Credentials (Vertex AI service account or API key) come from .env — see
docs/superpowers/specs/2026-08-17-gemini-extraction-design.md. --dry-run
needs no credentials. Output: _extractions/<model>/<stem>.json.
"""

import argparse
import glob as globmod
import json
import os
import sys

from config import SEGMENTS_DIR, EXTRACTIONS_DIR
from extraction import (DEFAULT_MODEL, estimate_cost, extract_page,
                        make_client)

# Planning figures for --dry-run (tokens per strip round trip).
EST_IN_PER_STRIP, EST_OUT_PER_STRIP = 1000, 500


def discover_stems(segments_dir):
    stems = []
    for mpath in sorted(globmod.glob(os.path.join(segments_dir, "*", "*.json"))):
        stem = os.path.splitext(os.path.basename(mpath))[0]
        if os.path.basename(os.path.dirname(mpath)) == stem:
            stems.append(stem)
    return stems


def _pending(stem, model, segments_dir, out_base, force):
    """(status, n_pending) for one stem without calling any API."""
    with open(os.path.join(segments_dir, stem, f"{stem}.json")) as f:
        manifest = json.load(f)
    if manifest.get("status") != "ok":
        return manifest.get("status", "unknown"), 0
    done = set()
    out_path = os.path.join(out_base, model, f"{stem}.json")
    if os.path.isfile(out_path) and not force:
        with open(out_path) as f:
            done = set(json.load(f).get("records", {}))
    n = sum(1 for r in manifest["records"] if str(r["index"]) not in done)
    return "ok", n


def main(argv=None):
    p = argparse.ArgumentParser(description="Extract record strips via Gemini.")
    p.add_argument("stems", nargs="*", help="Page stems (e.g. reg_p1)")
    p.add_argument("--all", action="store_true",
                   help="All segmented pages under the segments dir")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--force", action="store_true")
    p.add_argument("--limit", type=int, default=None,
                   help="Max NEW strips this run (cheap trials)")
    p.add_argument("--center", default="")
    p.add_argument("--year", default="")
    p.add_argument("--dry-run", action="store_true",
                   help="Count pending strips + estimate cost; no API calls")
    p.add_argument("--segments-dir", default=SEGMENTS_DIR)
    p.add_argument("--out", default=EXTRACTIONS_DIR)
    args = p.parse_args(argv)

    stems = list(args.stems)
    if args.all:
        stems += [s for s in discover_stems(args.segments_dir)
                  if s not in stems]
    if not stems:
        p.error("no stems given (pass stems or --all)")

    context = ", ".join(x for x in
                        ([f"Center: {args.center}"] if args.center else []) +
                        ([f"Year: {args.year}"] if args.year else []))

    if args.dry_run:
        total = 0
        refused = 0
        for stem in stems:
            status, n = _pending(stem, args.model, args.segments_dir,
                                 args.out, args.force)
            if status != "ok":
                refused += 1
                print(f"  {stem}: refused ({status})")
            else:
                total += n
                print(f"  {stem}: {n} pending")
        if args.limit is not None:
            total = min(total, args.limit)
        cost = estimate_cost(args.model, total * EST_IN_PER_STRIP,
                             total * EST_OUT_PER_STRIP)
        print(f"\nDRY RUN [{args.model}]: {total} strips pending, "
              f"{refused} pages refused, estimated ~${cost:.2f}")
        return 0

    client = make_client()
    tin = tout = new = skipped = errors = 0
    remaining = args.limit
    for stem in stems:
        if remaining is not None and remaining <= 0:
            break
        try:
            r = extract_page(client, args.model, stem,
                             segments_dir=args.segments_dir,
                             out_base=args.out, context=context,
                             force=args.force, limit=remaining)
        except Exception as err:
            errors += 1
            print(f"  {stem}: ERROR {err}")
            continue
        if "refused" in r:
            print(f"  {stem}: refused ({r['refused']})")
            continue
        n_new = len(r["records"]) - r["skipped_existing"] \
            if not args.force else len(r["records"])
        if remaining is not None:
            remaining -= n_new
        new += n_new
        skipped += r["skipped_existing"]
        tin += r["totals"].get("input_tokens", 0)
        tout += r["totals"].get("output_tokens", 0)
        print(f"  {stem}: {n_new} extracted, {r['skipped_existing']} skipped")

    cost = estimate_cost(args.model, tin, tout)
    print(f"\n[{args.model}] {new} strips extracted, {skipped} skipped, "
          f"{errors} errors | {tin} in / {tout} out tokens | est ${cost:.4f}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
