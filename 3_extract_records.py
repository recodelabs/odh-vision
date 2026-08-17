#!/usr/bin/env python3
"""
3_extract_records.py  —  Extract structured patient records from scan images.

This is the CORE extraction script. It:
    - Builds the Claude prompt for a given page image
    - Parses Claude's JSON response into structured record dicts
    - Tags confidence based on [?] markers and empty critical fields
    - Outputs clean JSON ready for step 4 (store)

In practice, Claude reads the image directly in a Cowork session.
For standalone / batch use, you can feed a saved response:

    python 3_extract_records.py page.jpg --region Masaka --center Kabanda \\
           --pdf "Apr week 2.pdf" --page 3 --response-file resp.json
"""

import sys, os, json, argparse
from typing import List, Dict

from config import HEADERS, HEADER_INDEX, AUDIT_FIELDS


# ═══════════════════════════════════════════════════════════════════════════════
#  Record builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_record(region: str, center: str, pdf: str, page: int,
                 raw: Dict) -> Dict:
    """Wrap a raw extraction dict with the standard metadata fields."""
    rec = {"Region": region, "Center": center,
           "Source PDF": pdf, "Page": page}
    for key in HEADERS:
        if key in raw:
            rec[key] = raw[key]
    return rec


def build_records_from_page(region, center, pdf, page, raw_rows):
    return [build_record(region, center, pdf, page, r) for r in raw_rows]


# ═══════════════════════════════════════════════════════════════════════════════
#  Prompt template  (send this to Claude alongside the page image)
# ═══════════════════════════════════════════════════════════════════════════════

EXTRACTION_PROMPT = """\
You are reading a handwritten Ugandan health-facility OPD register.
This is page {page} of "{pdf}" from {center} ({region}).

For EVERY patient row visible on this page, extract these fields:
  Record No., Date, Time in, Patient name, Village,
  1st time at ODH, Last care code, Travel time code, 1st voucher use,
  Age (yrs), Sex, Tests, Malaria result, Sev. malaria,
  Diagnosis, Treatment, Tab no., Cost (UGX)

Rules:
- Read the handwriting carefully. Ugandan names often include: Nakato,
  Ssemwuda, Nabanema, Katusiime, Tumukunde, Akello, Adongo, etc.
- If a value is unclear, append [?] to your best guess.
- If a value is very unclear / messy, append [??].
- If a cell is empty, use "".
- For Diagnosis, use the canonical ODH name where recognisable
  (Malaria, URTI, Pneumonia, Gastritis, PUD, UTI, S. Malaria, etc.).
- Treatment: write exactly what's on the register, abbreviations and all.
- Cost: numeric only (e.g. 5000).
- If a row spans two physical rows, merge into one record.

Return ONLY a JSON array of objects, no commentary.
Example:
[
  {{
    "Record No.": 1,
    "Date": "07/02/2026",
    "Patient name": "Kabusingye",
    "Village": "Kiryanongo",
    "Age (yrs)": "75",
    "Sex": "M",
    "Diagnosis": "Malaria",
    "Cost (UGX)": 7000,
    ...
  }}
]
"""


def make_prompt(region, center, pdf, page):
    return EXTRACTION_PROMPT.format(
        page=page, pdf=pdf, center=center, region=region)


# ═══════════════════════════════════════════════════════════════════════════════
#  Parse Claude's response
# ═══════════════════════════════════════════════════════════════════════════════

def parse_extraction_response(text: str) -> List[Dict]:
    """Parse Claude's JSON array from a response (handles code fences, etc.)."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("No JSON array found in response")
    return json.loads(text[start:end + 1])


# ═══════════════════════════════════════════════════════════════════════════════
#  Confidence tagging
# ═══════════════════════════════════════════════════════════════════════════════

def tag_confidence(records: List[Dict]) -> List[Dict]:
    """Add Overall confidence and Notes based on [?] markers / empty fields."""
    for rec in records:
        uncertain = []
        empty = []
        for f in AUDIT_FIELDS:
            val = str(rec.get(f, ""))
            if "[?]" in val or "[??]" in val:
                uncertain.append(f)
            if not val.strip():
                empty.append(f)

        n = len(uncertain) + len(empty)
        if n == 0:
            rec.setdefault("Overall confidence", "high")
        elif n <= 2:
            rec.setdefault("Overall confidence", "medium")
        else:
            rec.setdefault("Overall confidence", "low")

        parts = []
        if uncertain:
            parts.append("Uncertain: " + ", ".join(uncertain))
        if empty:
            parts.append("Empty: " + ", ".join(empty))
        if parts:
            existing = rec.get("Notes / uncertainties", "")
            new = "; ".join(parts)
            rec["Notes / uncertainties"] = (
                f"{existing}; {new}" if existing else new)

    return records


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extract records from a page scan.")
    p.add_argument("image")
    p.add_argument("--region", required=True)
    p.add_argument("--center", required=True)
    p.add_argument("--pdf", required=True)
    p.add_argument("--page", type=int, required=True)
    p.add_argument("--response-file", default=None,
                   help="File with Claude's JSON response (offline mode)")
    args = p.parse_args()

    if args.response_file:
        with open(args.response_file) as f:
            raw_rows = parse_extraction_response(f.read())
        records = build_records_from_page(
            args.region, args.center, args.pdf, args.page, raw_rows)
        records = tag_confidence(records)
        print(json.dumps(records, indent=2, default=str))
        print(f"\n{len(records)} records extracted.")
    else:
        prompt = make_prompt(args.region, args.center, args.pdf, args.page)
        print("═══ PROMPT — send to Claude with the image ═══")
        print(prompt)
        print("═══ END ═══")
        print("\nSave Claude's response to a file, then re-run with "
              "--response-file <path>")
