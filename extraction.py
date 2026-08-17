"""
extraction.py — Gemini-based extraction of segmented record strips.

Phase 2 of the odh-vision pipeline: consumes _segments/<stem>/ strips +
manifests (phase 1) and produces _extractions/<model>/<stem>.json.
See docs/superpowers/specs/2026-08-17-gemini-extraction-design.md.
"""

import json
import os
import time
from typing import Literal, Optional

from pydantic import BaseModel, model_validator

from config import PROJECT_ROOT, SEGMENTS_DIR, EXTRACTIONS_DIR

PROMPT_VERSION = "1"
DEFAULT_MODEL = "gemini-3.5-flash-lite"

# USD per 1M tokens (input, output), standard tier, as of 2026-08-17.
PRICING = {
    "gemini-3.5-flash-lite": (0.25, 1.50),
    "gemini-3.7-flash": (0.75, 3.75),
}


# ─── Environment & auth ──────────────────────────────────────────────────────

def load_env(path=None):
    """Load KEY=VALUE lines from .env into os.environ without overwriting."""
    path = path or os.path.join(PROJECT_ROOT, ".env")
    if not os.path.isfile(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def resolve_auth(env_path=None):
    """Determine credentials. Returns {"mode": "api_key", ...} or
    {"mode": "vertex", ...}; raises RuntimeError if nothing usable is set."""
    load_env(env_path)
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        return {"mode": "api_key", "api_key": api_key}

    creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if creds and not os.path.isabs(creds):
        creds = os.path.join(PROJECT_ROOT, creds)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project and creds and os.path.isfile(creds):
        with open(creds) as f:
            project = json.load(f).get("project_id")
    if project:
        return {"mode": "vertex", "project": project,
                "location": os.environ.get("GOOGLE_CLOUD_LOCATION", "global")}

    raise RuntimeError(
        "No credentials found. Put GOOGLE_APPLICATION_CREDENTIALS=<sa.json> "
        "(Vertex) or GEMINI_API_KEY=<key> in .env — see the spec.")


def make_client():
    """Build a google-genai client from resolved credentials."""
    from google import genai
    auth = resolve_auth()
    if auth["mode"] == "api_key":
        return genai.Client(api_key=auth["api_key"])
    return genai.Client(vertexai=True, project=auth["project"],
                        location=auth["location"])


# ─── Record schema ───────────────────────────────────────────────────────────

Confidence = Literal["high", "medium", "low", "illegible"]


class Reading(BaseModel):
    """One transcribed cell: verbatim value + confidence. Illegible ⇒ empty."""
    value: str = ""
    confidence: Confidence = "high"

    @model_validator(mode="after")
    def _illegible_is_empty(self):
        if self.confidence == "illegible" and self.value != "":
            raise ValueError("illegible readings must have an empty value")
        return self


class RecordExtraction(BaseModel):
    """One patient record (3 sub-rows) transcribed from a strip, form order."""
    record_no: Reading
    day: Reading
    month: Reading
    time_hh: Reading
    time_mm: Reading
    am_pm: Reading              # "AM" | "PM" | ""
    voucher_na: Reading         # NA checkbox: "Y" | ""
    voucher_color: Reading
    voucher_id: Reading
    patient_name: Reading
    village: Reading
    village_code: Reading
    first_time_odh: Reading     # "Y" | "N" | ""
    first_voucher_use: Reading  # "Y" | "N" | ""
    sex: Reading                # "M" | "F" | ""
    hh_owns_phone: Reading      # "Y" | "N" | ""
    hh_owns_toilet: Reading     # "Y" | "N" | ""
    last_care: Reading          # code 1-5
    group_appt: Reading         # "Y" | ""
    age_yrs: Reading
    hoh_education: Reading
    tests: Reading
    result_pn: Reading          # "P" | "N" | ""
    malaria: Reading            # Mal checkbox: "Y" | ""
    sev_malaria: Reading        # Sev mal checkbox: "Y" | ""
    weight_kg: Reading
    diagnosis: Reading
    art_dose: Reading           # ArT dose # boxes: "1" | "2" | "3" | ""
    treatment_line1: Reading
    treatment_line2: Reading
    treatment_line3: Reading
    tab_no: Reading
    full_cost: Reading
    balance: Reading
    cost_after_discount: Reading
    row_notes: str = ""


FIELD_NAMES = [n for n, f in RecordExtraction.model_fields.items()
               if f.annotation is Reading]


# ─── Prompt ──────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """\
You are transcribing ONE patient record from a handwritten Ugandan health\
-facility OPD register. The image is a horizontal strip: the PRINTED COLUMN \
HEADER BAND is stitched on top, and below it is record #{record_index} of the \
page — exactly one record spanning 3 physical sub-rows.

Column layout, left to right (use the printed header band to locate each):
No. (record number, left margin) | Row ID (pre-printed sub-row numbers — do not transcribe) | Day/HH/AM | Month/MM/PM | \
voucher (NA checkbox, color, ID) | Name / Village / Village code | 1st time \
in ODH (Y/N) + Last care code | 1st voucher use (Y/N) + Group appt. | Sex \
(M/F) + Age (yrs) | HH owns ≥1 phone (Y/N) | HH owns ≥1 toilet (Y/N) + HoH \
education | Tests | Result (P/N) | Malaria checkboxes (Mal, Sev mal) + \
Weight (kg) | Diagnosis | ArT dose # (1/2/3 boxes) | Treatment (up to 3 \
handwritten lines) | Tab # | costs (Full cost / Balance / Cost after \
discount, top to bottom).

Rules:
- Transcribe VERBATIM: keep abbreviations, dose notation, and spelling \
exactly as written. Do not normalize or expand.
- Checkboxes: a marked box (tick/X/scribble) = its printed label ("Y", "N", \
"M", "F", "P"...). Unmarked = "". If both marks or ambiguous, pick the \
clearer one and lower the confidence.
- Confidence per field: "high" = certain; "medium" = probably right; \
"low" = a guess you can defend; "illegible" = unreadable — then value MUST \
be "" (never guess an illegible cell).
- Empty cells: value "" with confidence "high".
- Treatment lines map top sub-row → treatment_line1, middle → \
treatment_line2, bottom → treatment_line3; leave unused lines "".
- Costs: digits only where possible (e.g. 26000).
- Anything anomalous (crossed-out text, merged rows, arrows, marginalia) \
goes in row_notes.
{context_block}\
Return ONLY the JSON object matching the provided schema."""


def build_prompt(record_index: int, context: str = "") -> str:
    context_block = f"Page context: {context}\n" if context else ""
    return PROMPT_TEMPLATE.format(record_index=record_index,
                                  context_block=context_block)


# ─── Cost accounting ─────────────────────────────────────────────────────────

def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """USD cost at standard-tier rates; 0.0 for models not in PRICING."""
    if model not in PRICING:
        return 0.0
    in_rate, out_rate = PRICING[model]
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000
