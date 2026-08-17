"""
extraction.py — Gemini-based extraction of segmented record strips.

Phase 2 of the odh-vision pipeline: consumes _segments/<stem>/ strips +
manifests (phase 1) and produces _extractions/<model>/<stem>.json.
See docs/superpowers/specs/2026-08-17-gemini-extraction-design.md.
"""

import json
import os
import re
import time
from typing import Literal, Optional

from pydantic import BaseModel, model_validator

from config import PROJECT_ROOT, SEGMENTS_DIR, EXTRACTIONS_DIR

PROMPT_VERSION = "3"
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
- The record you are transcribing lies BETWEEN the two red horizontal \
lines. Rows fully outside the red lines belong to ADJACENT records -- do \
not transcribe values that clearly originate in an adjacent record's \
cells. However, handwriting that OVERLAPS or CROSSES a red line belongs to \
whichever record's cell it originates in: if a mark or entry starts inside \
this record's rows (including checkbox marks and text partly covered by \
the red line itself), transcribe it as this record's value even though it \
extends past the line.
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


# ─── Per-strip extraction ────────────────────────────────────────────────────

_RETRYABLE_CODES = {429, 500, 501, 502, 503, 504}
# Word-bounded so a message containing e.g. "15000" doesn't false-match "500".
_RETRYABLE_CODE_RE = re.compile(r"\b(429|50[0-4])\b")
_RETRYABLE_NAMED_MARKERS = ("RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE_EXCEEDED")


def _is_retryable(err: Exception) -> bool:
    # Prefer a numeric status code attribute when the exception carries one.
    code = getattr(err, "code", None)
    if code is not None:
        return code in _RETRYABLE_CODES
    msg = str(err)
    return bool(_RETRYABLE_CODE_RE.search(msg)) or any(
        m in msg for m in _RETRYABLE_NAMED_MARKERS)


def extract_strip(client, model, strip_path, record_index, context="",
                  thinking_budget=0, max_attempts=4, sleep=time.sleep):
    """Extract one record strip. Returns (RecordExtraction, usage dict)."""
    from google.genai import types

    with open(strip_path, "rb") as f:
        png = f.read()
    prompt = build_prompt(record_index, context)

    cfg = {"response_mime_type": "application/json",
           "response_schema": RecordExtraction,
           "temperature": 0.0}
    if thinking_budget is not None:
        cfg["thinking_config"] = types.ThinkingConfig(
            thinking_budget=thinking_budget)

    last_err = None
    attempt = 0
    while attempt < max_attempts:
        try:
            t0 = time.time()
            resp = client.models.generate_content(
                model=model,
                contents=[types.Part.from_bytes(data=png, mime_type="image/png"),
                          prompt],
                config=types.GenerateContentConfig(**cfg))
            rec = resp.parsed
            if not isinstance(rec, RecordExtraction):
                rec = RecordExtraction.model_validate_json(resp.text)
            u = resp.usage_metadata
            usage = {
                "input_tokens": getattr(u, "prompt_token_count", 0) or 0,
                "output_tokens": (getattr(u, "candidates_token_count", 0) or 0)
                                 + (getattr(u, "thoughts_token_count", 0) or 0),
                "latency_s": round(time.time() - t0, 2),
            }
            return rec, usage
        except Exception as err:
            # Model may reject the thinking config — drop it once and retry.
            if "thinking" in str(err).lower() and "thinking_config" in cfg:
                cfg.pop("thinking_config")
                last_err = err
                continue
            if _is_retryable(err):
                last_err = err
                sleep(2 ** attempt)
                attempt += 1
                continue
            raise
    raise RuntimeError(
        f"extract_strip({os.path.basename(strip_path)}) failed after "
        f"{max_attempts} attempts: {last_err}")


# ─── Per-page orchestration ──────────────────────────────────────────────────

def _atomic_write_json(path, payload):
    tmp = path + ".wtmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def extract_page(client, model, stem, segments_dir=SEGMENTS_DIR,
                 out_base=EXTRACTIONS_DIR, context="", force=False,
                 limit=None, thinking_budget=0, sleep=time.sleep):
    """Extract every strip of one segmented page. Resume-safe and atomic."""
    manifest_path = os.path.join(segments_dir, stem, f"{stem}.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    if manifest.get("status") != "ok":
        return {"stem": stem, "refused": manifest.get("status", "unknown")}

    out_dir = os.path.join(out_base, model)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{stem}.json")

    page = {"stem": stem, "model": model, "prompt_version": PROMPT_VERSION,
            "records": {}, "totals": {}}
    if os.path.isfile(out_path):
        with open(out_path) as f:
            page = json.load(f)
        # Re-stamp provenance so a re-extraction never carries a stale
        # prompt_version/model from a prior run. Note: with mixed old/new
        # records after a partial force run, the page-level stamp reflects
        # the LATEST run only -- a documented, acceptable tradeoff.
        page["prompt_version"] = PROMPT_VERSION
        page["model"] = model

    skipped = 0
    done = 0
    run_in = 0
    run_out = 0
    strip_errors = []
    for entry in manifest["records"]:
        key = str(entry["index"])
        if key in page["records"] and not force:
            skipped += 1
            continue
        if limit is not None and done >= limit:
            break
        strip_path = os.path.join(segments_dir, stem, entry["strip"])
        try:
            rec, usage = extract_strip(client, model, strip_path, entry["index"],
                                       context=context,
                                       thinking_budget=thinking_budget,
                                       sleep=sleep)
        except Exception as err:
            # Leave any prior record for this key untouched so it stays
            # re-tryable on resume (never persist errors into page["records"]).
            strip_errors.append({"index": entry["index"], "error": str(err)})
            continue
        page["records"][key] = {"fields": rec.model_dump(), "usage": usage}
        run_in += usage.get("input_tokens", 0)
        run_out += usage.get("output_tokens", 0)
        done += 1
        _totalize(page, model)
        _atomic_write_json(out_path, page)     # crash-safe after every record

    if not os.path.isfile(out_path):
        _totalize(page, model)
        _atomic_write_json(out_path, page)

    result = dict(page)
    result["skipped_existing"] = skipped
    result["extracted_this_run"] = done
    result["usage_this_run"] = {"input_tokens": run_in, "output_tokens": run_out}
    result["strip_errors"] = strip_errors
    return result


def _totalize(page, model):
    tin = sum(r["usage"]["input_tokens"] for r in page["records"].values())
    tout = sum(r["usage"]["output_tokens"] for r in page["records"].values())
    page["totals"] = {"input_tokens": tin, "output_tokens": tout,
                      "est_cost_usd": round(estimate_cost(model, tin, tout), 6)}
