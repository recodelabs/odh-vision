# Gemini Strip Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scripted, resumable, cost-tracked extraction of segmented record strips into schema-validated JSON via Gemini on Vertex AI, plus a Flash-Lite vs Flash A/B comparison harness.

**Architecture:** Library `extraction.py` (env/auth, pydantic schema, prompt, per-strip call with retries, per-page orchestration with atomic resume-safe output) driven by CLI `1c_extract_strips.py`; local-only `1d_compare_models.py` computes field-level agreement between two models' outputs. All unit tests run offline against a stub client; one live smoke test is credential-gated.

**Tech Stack:** Python 3, `google-genai` (Vertex AI mode), `pydantic`, pytest. Consumes phase-1 outputs in `_segments/`.

**Spec:** `docs/superpowers/specs/2026-08-17-gemini-extraction-design.md`

## Global Constraints

- Auth order: `GEMINI_API_KEY` if set, else Vertex (`GOOGLE_APPLICATION_CREDENTIALS`, project from `GOOGLE_CLOUD_PROJECT` or the SA JSON's `project_id`, location `GOOGLE_CLOUD_LOCATION` default `"global"`). Secrets come only from env or the gitignored `.env`; never committed, never printed/logged.
- `illegible` confidence ⇒ empty `value` — enforced by a pydantic validator, never guessed.
- Output path contract: `_extractions/<model>/<stem>.json`, structure exactly as in the spec; atomic writes (tmp + `os.replace`) after every record.
- `needs_review` pages are refused; resume skips already-extracted records unless `--force`.
- Pricing table (per 1M tokens, standard tier, as of 2026-08-17): `gemini-3.5-flash-lite` = $0.25 in / $1.50 out; `gemini-3.7-flash` = $0.75 in / $3.75 out.
- `PROMPT_VERSION = "1"`. Default model `gemini-3.5-flash-lite`. Temperature 0.0. Thinking budget 0 (drop the config and retry once if the API rejects it).
- Run tests as `python -m pytest` from the repo root. All tests except `tests/test_live_extraction.py` must pass with no network and no credentials.
- Never commit `.env`, `.gcp-sa.json`, `_extractions/`, or anything under `_segments/`/`_output/`.

---

### Task 1: Dependencies, env plumbing, client factory

**Files:**
- Modify: `requirements.txt` (append)
- Modify: `.gitignore` (append)
- Modify: `config.py` (add `EXTRACTIONS_DIR` below `SEGMENTS_DIR`)
- Create: `extraction.py` (module header + env/auth section)
- Test: `tests/test_extraction.py` (env/auth tests)

**Interfaces:**
- Produces: `config.EXTRACTIONS_DIR` (str, dir auto-created); `extraction.load_env(path=None)`; `extraction.resolve_auth() -> dict` (pure, testable: returns `{"mode": "api_key"|"vertex", ...}` or raises `RuntimeError`); `extraction.make_client()` (thin wrapper that builds the SDK client from `resolve_auth()`).

- [ ] **Step 1: Append dependencies and ignores**

Append to `requirements.txt`:

```text
google-genai>=1.0
pydantic>=2.6
```

Append to `.gitignore`:

```gitignore

# Extraction phase — secrets and model outputs
.env
.gcp-sa.json
_extractions/
```

Run: `python -m pip install -r requirements.txt`
Expected: installs cleanly; `python -c "from google import genai; import pydantic; print('ok')"` prints `ok`.

- [ ] **Step 2: Add `EXTRACTIONS_DIR` to `config.py`** (directly below the `SEGMENTS_DIR` block)

```python
EXTRACTIONS_DIR = os.path.join(PROJECT_ROOT, "_extractions")
os.makedirs(EXTRACTIONS_DIR, exist_ok=True)
```

- [ ] **Step 3: Write the failing tests** (start `tests/test_extraction.py`)

```python
import json
import os

import pytest

from extraction import load_env, resolve_auth


def test_load_env_sets_without_overwriting(tmp_path, monkeypatch):
    envfile = tmp_path / ".env"
    envfile.write_text('FOO_X=abc\n# comment\nBAR_Y="quoted"\nEXISTING=new\n')
    monkeypatch.setenv("EXISTING", "old")
    monkeypatch.delenv("FOO_X", raising=False)
    monkeypatch.delenv("BAR_Y", raising=False)
    load_env(str(envfile))
    assert os.environ["FOO_X"] == "abc"
    assert os.environ["BAR_Y"] == "quoted"
    assert os.environ["EXISTING"] == "old"          # no overwrite


def test_resolve_auth_prefers_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k-123")
    auth = resolve_auth(env_path="/nonexistent")
    assert auth == {"mode": "api_key", "api_key": "k-123"}


def test_resolve_auth_vertex_from_sa_json(tmp_path, monkeypatch):
    sa = tmp_path / "sa.json"
    sa.write_text(json.dumps({"type": "service_account", "project_id": "proj-9"}))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(sa))
    auth = resolve_auth(env_path="/nonexistent")
    assert auth == {"mode": "vertex", "project": "proj-9", "location": "global"}


def test_resolve_auth_raises_without_credentials(monkeypatch):
    for var in ("GEMINI_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS",
                "GOOGLE_CLOUD_PROJECT"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(RuntimeError, match="credentials"):
        resolve_auth(env_path="/nonexistent")
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `python -m pytest tests/test_extraction.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'extraction'`

- [ ] **Step 5: Create `extraction.py` with the env/auth section**

```python
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
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_extraction.py -v`
Expected: 4 passed. Then full suite: `python -m pytest -v` → 22 passed (18 prior + 4).

- [ ] **Step 7: Commit**

```bash
git add requirements.txt .gitignore config.py extraction.py tests/test_extraction.py
git commit -m "feat: extraction env/auth plumbing, deps, EXTRACTIONS_DIR"
```

---

### Task 2: Record schema, prompt, cost math

**Files:**
- Modify: `extraction.py` (append)
- Test: `tests/test_extraction.py` (append)

**Interfaces:**
- Produces: `Reading` (pydantic: `value: str`, `confidence: Literal[...]`; validator forces `value=""` when `confidence=="illegible"` by raising `ValueError`); `RecordExtraction` (all form fields + `row_notes: str`); `FIELD_NAMES` (ordered list of the Reading-typed field names); `build_prompt(record_index: int, context: str = "") -> str`; `estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float` (USD, returns 0.0 for unknown models).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_extraction.py`)

```python
from pydantic import ValidationError

from extraction import (Reading, RecordExtraction, FIELD_NAMES,
                        build_prompt, estimate_cost)


def test_illegible_requires_empty_value():
    assert Reading(value="", confidence="illegible").value == ""
    with pytest.raises(ValidationError):
        Reading(value="guess", confidence="illegible")


def test_record_schema_roundtrip():
    payload = {name: {"value": "", "confidence": "high"} for name in FIELD_NAMES}
    payload["patient_name"] = {"value": "Aromo K.", "confidence": "medium"}
    payload["row_notes"] = "second treatment line crossed out"
    rec = RecordExtraction.model_validate(payload)
    dumped = rec.model_dump()
    assert dumped["patient_name"]["value"] == "Aromo K."
    assert dumped["row_notes"].startswith("second")
    assert len(FIELD_NAMES) == 35


def test_build_prompt_mentions_layout_and_context():
    p = build_prompt(3, context="Center: Kameno, Year: 2026")
    assert "record 3" in p.lower() or "record #3" in p.lower()
    assert "header" in p.lower()
    assert "Kameno" in p
    assert "illegible" in p.lower()
    assert "Kameno" not in build_prompt(1)


def test_estimate_cost():
    # 1M in + 1M out at flash-lite rates = 0.25 + 1.50
    assert estimate_cost("gemini-3.5-flash-lite", 1_000_000, 1_000_000) == pytest.approx(1.75)
    assert estimate_cost("gemini-3.7-flash", 2_000_000, 0) == pytest.approx(1.50)
    assert estimate_cost("unknown-model", 5, 5) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_extraction.py -v`
Expected: new tests FAIL with ImportError (`Reading` etc. not defined)

- [ ] **Step 3: Append schema, prompt, cost to `extraction.py`**

```python
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
No. (record number, left margin) | Row ID | Day/HH/AM | Month/MM/PM | \
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_extraction.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add extraction.py tests/test_extraction.py
git commit -m "feat: form-faithful record schema, extraction prompt, cost math"
```

---

### Task 3: Per-strip extraction call with retries

**Files:**
- Modify: `extraction.py` (append)
- Test: `tests/test_extraction.py` (append)

**Interfaces:**
- Produces: `extract_strip(client, model, strip_path, record_index, context="", thinking_budget=0, max_attempts=4, sleep=time.sleep) -> tuple[RecordExtraction, dict]` — usage dict `{"input_tokens", "output_tokens", "latency_s"}`; output_tokens includes thinking tokens. Retries on 429/5xx/RESOURCE_EXHAUSTED with exponential backoff (`sleep(2**attempt)`); drops the thinking config once if the API rejects it; raises `RuntimeError` after `max_attempts` retryable failures and re-raises non-retryable errors immediately. The injectable `sleep` keeps tests instant.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_extraction.py`; the stub client mimics the SDK surface)

```python
import extraction as ex


class _Usage:
    prompt_token_count = 800
    candidates_token_count = 500
    thoughts_token_count = 100


class _Resp:
    def __init__(self, parsed):
        self.parsed = parsed
        self.text = parsed.model_dump_json() if parsed else "{}"
        self.usage_metadata = _Usage()


def _valid_record():
    payload = {n: {"value": "", "confidence": "high"} for n in FIELD_NAMES}
    return RecordExtraction.model_validate(payload)


class StubModels:
    def __init__(self, script):
        self.script = list(script)   # each item: _Resp or Exception
        self.calls = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "config": config})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class StubClient:
    def __init__(self, script):
        self.models = StubModels(script)


@pytest.fixture
def strip_file(tmp_path):
    p = tmp_path / "page_p1_rec2.png"
    p.write_bytes(b"\x89PNG fake")
    return str(p)


def test_extract_strip_success(strip_file):
    client = StubClient([_Resp(_valid_record())])
    rec, usage = ex.extract_strip(client, "gemini-3.5-flash-lite", strip_file, 2)
    assert isinstance(rec, RecordExtraction)
    assert usage["input_tokens"] == 800
    assert usage["output_tokens"] == 600      # candidates + thoughts
    assert "latency_s" in usage
    assert len(client.models.calls) == 1


def test_extract_strip_retries_on_429_then_succeeds(strip_file):
    client = StubClient([RuntimeError("429 RESOURCE_EXHAUSTED"),
                         _Resp(_valid_record())])
    naps = []
    rec, _ = ex.extract_strip(client, "m", strip_file, 1, sleep=naps.append)
    assert isinstance(rec, RecordExtraction)
    assert naps == [1]                         # 2**0


def test_extract_strip_gives_up_after_max_attempts(strip_file):
    errs = [RuntimeError("503 unavailable")] * 4
    client = StubClient(errs)
    with pytest.raises(RuntimeError, match="failed after 4 attempts"):
        ex.extract_strip(client, "m", strip_file, 1, sleep=lambda s: None)


def test_extract_strip_nonretryable_raises_immediately(strip_file):
    client = StubClient([ValueError("400 invalid argument: bad schema")])
    with pytest.raises(ValueError):
        ex.extract_strip(client, "m", strip_file, 1, sleep=lambda s: None)
    assert len(client.models.calls) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_extraction.py -v`
Expected: new tests FAIL (`extract_strip` not defined)

- [ ] **Step 3: Append `extract_strip` to `extraction.py`**

```python
# ─── Per-strip extraction ────────────────────────────────────────────────────

_RETRYABLE_MARKERS = ("429", "500", "503", "504", "RESOURCE_EXHAUSTED",
                      "UNAVAILABLE", "DEADLINE_EXCEEDED")


def _is_retryable(err: Exception) -> bool:
    msg = str(err)
    return any(m in msg for m in _RETRYABLE_MARKERS)


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
    for attempt in range(max_attempts):
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
                continue
            raise
    raise RuntimeError(
        f"extract_strip({os.path.basename(strip_path)}) failed after "
        f"{max_attempts} attempts: {last_err}")
```

Note: the stub client's `generate_content` receives `config=types.GenerateContentConfig(...)` — a real SDK object. `google-genai` constructs it locally (no network), so offline tests still work.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_extraction.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add extraction.py tests/test_extraction.py
git commit -m "feat: per-strip Gemini call with retries and usage capture"
```

---

### Task 4: Per-page orchestration — resume, refusal, atomic output

**Files:**
- Modify: `extraction.py` (append)
- Test: `tests/test_extraction.py` (append)

**Interfaces:**
- Produces: `extract_page(client, model, stem, segments_dir=SEGMENTS_DIR, out_base=EXTRACTIONS_DIR, context="", force=False, limit=None, thinking_budget=0, sleep=time.sleep) -> dict` — the page result written to `<out_base>/<model>/<stem>.json` with keys `stem, model, prompt_version, records, totals` (spec format); returns the dict plus a transient `"skipped_existing"` count and, for refused pages, `{"stem", "refused": "<reason>"}` without writing an output file. `limit` caps NEW extractions this call (for `--limit`). Output written atomically after each record.

- [ ] **Step 1: Write the failing tests** (append; fabricate a segments dir)

```python
from extraction import extract_page


def _make_segments(tmp_path, stem="reg_p1", status="ok", n=3):
    d = tmp_path / "segments" / stem
    d.mkdir(parents=True)
    records = []
    for i in range(1, n + 1):
        name = f"{stem}_rec{i}.png"
        (d / name).write_bytes(b"\x89PNG fake")
        records.append({"index": i, "y0": 0, "y1": 10, "strip": name})
    manifest = {"stem": stem, "status": status, "records": records,
                "warnings": [], "header_band": [0, 5], "col_x": [],
                "canonical_size": [2000, 1400], "source_image": "x.png"}
    (d / f"{stem}.json").write_text(json.dumps(manifest))
    return str(tmp_path / "segments")


def test_extract_page_writes_output(tmp_path):
    seg = _make_segments(tmp_path)
    out = str(tmp_path / "ex")
    client = StubClient([_Resp(_valid_record()) for _ in range(3)])
    result = extract_page(client, "gemini-3.5-flash-lite", "reg_p1",
                          segments_dir=seg, out_base=out, sleep=lambda s: None)
    path = os.path.join(out, "gemini-3.5-flash-lite", "reg_p1.json")
    assert os.path.isfile(path)
    saved = json.load(open(path))
    assert saved["prompt_version"] == ex.PROMPT_VERSION
    assert set(saved["records"]) == {"1", "2", "3"}
    assert saved["records"]["2"]["usage"]["input_tokens"] == 800
    assert saved["totals"]["input_tokens"] == 2400
    assert saved["totals"]["est_cost_usd"] == pytest.approx(
        ex.estimate_cost("gemini-3.5-flash-lite", 2400, 1800))
    assert result["stem"] == "reg_p1"


def test_extract_page_refuses_needs_review(tmp_path):
    seg = _make_segments(tmp_path, status="needs_review")
    out = str(tmp_path / "ex")
    client = StubClient([])
    result = extract_page(client, "m", "reg_p1", segments_dir=seg, out_base=out)
    assert result["refused"] == "needs_review"
    assert not os.path.exists(os.path.join(out, "m", "reg_p1.json"))
    assert client.models.calls == []


def test_extract_page_resumes_and_forces(tmp_path):
    seg = _make_segments(tmp_path)
    out = str(tmp_path / "ex")
    c1 = StubClient([_Resp(_valid_record()) for _ in range(3)])
    extract_page(c1, "m", "reg_p1", segments_dir=seg, out_base=out,
                 sleep=lambda s: None)
    # resume: nothing new to do
    c2 = StubClient([])
    r2 = extract_page(c2, "m", "reg_p1", segments_dir=seg, out_base=out,
                      sleep=lambda s: None)
    assert c2.models.calls == [] and r2["skipped_existing"] == 3
    # force: re-extracts all
    c3 = StubClient([_Resp(_valid_record()) for _ in range(3)])
    r3 = extract_page(c3, "m", "reg_p1", segments_dir=seg, out_base=out,
                      force=True, sleep=lambda s: None)
    assert len(c3.models.calls) == 3 and r3["skipped_existing"] == 0


def test_extract_page_limit_caps_new_extractions(tmp_path):
    seg = _make_segments(tmp_path)
    out = str(tmp_path / "ex")
    client = StubClient([_Resp(_valid_record())])
    extract_page(client, "m", "reg_p1", segments_dir=seg, out_base=out,
                 limit=1, sleep=lambda s: None)
    saved = json.load(open(os.path.join(out, "m", "reg_p1.json")))
    assert len(saved["records"]) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_extraction.py -v`
Expected: new tests FAIL (`extract_page` not defined)

- [ ] **Step 3: Append `extract_page` to `extraction.py`**

```python
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
    if os.path.isfile(out_path) and not force:
        with open(out_path) as f:
            page = json.load(f)

    skipped = 0
    done = 0
    for entry in manifest["records"]:
        key = str(entry["index"])
        if key in page["records"] and not force:
            skipped += 1
            continue
        if limit is not None and done >= limit:
            break
        strip_path = os.path.join(segments_dir, stem, entry["strip"])
        rec, usage = extract_strip(client, model, strip_path, entry["index"],
                                   context=context,
                                   thinking_budget=thinking_budget,
                                   sleep=sleep)
        page["records"][key] = {"fields": rec.model_dump(), "usage": usage}
        done += 1
        _totalize(page, model)
        _atomic_write_json(out_path, page)     # crash-safe after every record

    if done == 0 and skipped and not os.path.isfile(out_path):
        _totalize(page, model)
        _atomic_write_json(out_path, page)

    result = dict(page)
    result["skipped_existing"] = skipped
    return result


def _totalize(page, model):
    tin = sum(r["usage"]["input_tokens"] for r in page["records"].values())
    tout = sum(r["usage"]["output_tokens"] for r in page["records"].values())
    page["totals"] = {"input_tokens": tin, "output_tokens": tout,
                      "est_cost_usd": round(estimate_cost(model, tin, tout), 6)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_extraction.py -v`
Expected: 16 passed. Full suite: `python -m pytest -v` → 34 passed.

- [ ] **Step 5: Commit**

```bash
git add extraction.py tests/test_extraction.py
git commit -m "feat: per-page extraction with resume, refusal, atomic writes"
```

---

### Task 5: CLI `1c_extract_strips.py`

**Files:**
- Create: `1c_extract_strips.py`
- Test: `tests/test_extract_cli.py`

**Interfaces:**
- Consumes: `extraction.extract_page`, `extraction.make_client`, `extraction.estimate_cost`, `extraction.PRICING`, `config.SEGMENTS_DIR`.
- Produces: CLI `python 1c_extract_strips.py [stems...] [--all] [--model M] [--force] [--limit N] [--center X] [--year Y] [--dry-run] [--segments-dir D] [--out D]`. `--all` discovers every `<stem>/<stem>.json` under the segments dir. `--dry-run` needs NO credentials: prints pending strip count and cost estimate (uses 1500 tokens/strip planning figure: 1000 in + 500 out) and exits 0. Real runs print a per-page line and a final summary (pages, strips new/skipped/refused, tokens, est cost). Exit 0 on success, 1 if any page errored.
- Also produces `main(argv=None)` importable for tests; `discover_stems(segments_dir) -> list[str]`.

- [ ] **Step 1: Write the failing test** (`tests/test_extract_cli.py`; dry-run path only — no client involved)

```python
import importlib
import json
import sys


def _make_segments(tmp_path, stem, status="ok", n=5):
    d = tmp_path / "segments" / stem
    d.mkdir(parents=True)
    records = [{"index": i, "y0": 0, "y1": 1, "strip": f"{stem}_rec{i}.png"}
               for i in range(1, n + 1)]
    for r in records:
        (d / r["strip"]).write_bytes(b"png")
    (d / f"{stem}.json").write_text(json.dumps(
        {"stem": stem, "status": status, "records": records, "warnings": []}))


def test_dry_run_counts_and_estimates(tmp_path, capsys):
    _make_segments(tmp_path, "reg_p1")
    _make_segments(tmp_path, "reg_p2", status="needs_review")
    cli = importlib.import_module("1c_extract_strips")
    rc = cli.main(["--all", "--dry-run",
                   "--segments-dir", str(tmp_path / "segments"),
                   "--out", str(tmp_path / "ex")])
    outtxt = capsys.readouterr().out
    assert rc == 0
    assert "5 strips" in outtxt          # only the ok page counts
    assert "refused" in outtxt.lower() or "needs_review" in outtxt.lower()
    assert "$" in outtxt                 # cost estimate printed
```

(Import note: the module name starts with a digit, hence `importlib.import_module("1c_extract_strips")` — works because the repo root is on `sys.path` under `python -m pytest`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_extract_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named '1c_extract_strips'`

- [ ] **Step 3: Write `1c_extract_strips.py`**

```python
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
```

- [ ] **Step 4: Run test, then full suite**

Run: `python -m pytest tests/test_extract_cli.py -v` → 1 passed; `python -m pytest -v` → 35 passed.

- [ ] **Step 5: Commit**

```bash
git add 1c_extract_strips.py tests/test_extract_cli.py
git commit -m "feat: 1c_extract_strips CLI with dry-run, resume, cost summary"
```

---

### Task 6: A/B comparison harness `1d_compare_models.py`

**Files:**
- Create: `1d_compare_models.py`
- Test: `tests/test_compare.py`

**Interfaces:**
- Consumes: extraction output files; `extraction.FIELD_NAMES`; `config.EXTRACTIONS_DIR`.
- Produces: `norm(s) -> str` (collapse whitespace + casefold); `compare_extractions(dir_a, dir_b) -> dict` with keys `n_records, n_compared_fields, agreement_rate, both_empty, per_field` (dict field → `{"agree": int, "total": int, "rate": float}`) and `disagreements` (list of dicts `stem, record, field, a_value, a_conf, b_value, b_conf`); CLI `python 1d_compare_models.py --models A B [--out-csv PATH]` printing the report worst-fields-first and writing the CSV (default `_extractions/compare_<A>__<B>.csv`). Agreement counts exclude both-empty pairs (tracked separately).

- [ ] **Step 1: Write the failing tests** (`tests/test_compare.py`)

```python
import csv
import importlib
import json

cmp_mod = None


def setup_module(module):
    global cmp_mod
    cmp_mod = importlib.import_module("1d_compare_models")


def _write_extraction(base, model, stem, values):
    """values: {record_key: {field: (value, conf)}}"""
    d = base / model
    d.mkdir(parents=True, exist_ok=True)
    from extraction import FIELD_NAMES
    records = {}
    for rk, fields in values.items():
        f = {name: {"value": "", "confidence": "high"} for name in FIELD_NAMES}
        for name, (v, c) in fields.items():
            f[name] = {"value": v, "confidence": c}
        records[rk] = {"fields": f, "usage": {"input_tokens": 1,
                                              "output_tokens": 1,
                                              "latency_s": 0.1}}
    (d / f"{stem}.json").write_text(json.dumps(
        {"stem": stem, "model": model, "prompt_version": "1",
         "records": records, "totals": {}}))


def test_norm():
    assert cmp_mod.norm("  Aromo   KETTY ") == "aromo ketty"


def test_compare_and_csv(tmp_path):
    _write_extraction(tmp_path, "m1", "p1", {
        "1": {"patient_name": ("Aromo Ketty", "high"),
              "diagnosis": ("Malaria", "high")},
        "2": {"patient_name": ("Akello", "medium")},
    })
    _write_extraction(tmp_path, "m2", "p1", {
        "1": {"patient_name": ("aromo  ketty", "medium"),      # agrees (norm)
              "diagnosis": ("PID", "low")},                   # disagrees
        "2": {"patient_name": ("Akello", "high")},            # agrees
    })
    result = cmp_mod.compare_extractions(str(tmp_path / "m1"),
                                         str(tmp_path / "m2"))
    assert result["n_records"] == 2
    assert len(result["disagreements"]) == 1
    d = result["disagreements"][0]
    assert (d["field"], d["a_value"], d["b_value"]) == ("diagnosis",
                                                        "Malaria", "PID")
    assert result["per_field"]["diagnosis"]["rate"] == 0.0
    assert result["per_field"]["patient_name"]["rate"] == 1.0
    assert result["both_empty"] > 0

    out_csv = tmp_path / "cmp.csv"
    cmp_mod.write_csv(result["disagreements"], str(out_csv))
    rows = list(csv.DictReader(open(out_csv)))
    assert len(rows) == 1 and rows[0]["field"] == "diagnosis"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_compare.py -v`
Expected: FAIL — module `1d_compare_models` not found

- [ ] **Step 3: Write `1d_compare_models.py`**

```python
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
```

- [ ] **Step 4: Run tests, then full suite**

Run: `python -m pytest tests/test_compare.py -v` → 2 passed; `python -m pytest -v` → 37 passed.

- [ ] **Step 5: Commit**

```bash
git add 1d_compare_models.py tests/test_compare.py
git commit -m "feat: 1d_compare_models A/B agreement harness"
```

---

### Task 7: Live smoke test + mini A/B on sample pages (credential-gated)

**Files:**
- Create: `tests/test_live_extraction.py`
- Modify (results only): `docs/superpowers/specs/2026-08-17-gemini-extraction-design.md`

**Interfaces:**
- Consumes: everything above; real credentials in `.env` (`GOOGLE_APPLICATION_CREDENTIALS` service-account route per spec).

- [ ] **Step 1: Write the credential-gated live test**

```python
"""Live smoke test — auto-skipped unless credentials are configured."""
import glob
import os

import pytest

from config import SEGMENTS_DIR
from extraction import (DEFAULT_MODEL, RecordExtraction, extract_strip,
                        make_client, resolve_auth)


def _have_credentials():
    try:
        resolve_auth()
        return True
    except RuntimeError:
        return False


def _a_strip():
    strips = sorted(glob.glob(os.path.join(SEGMENTS_DIR, "*", "*_rec1.png")))
    return strips[0] if strips else None


@pytest.mark.skipif(not _have_credentials(), reason="no Gemini credentials")
@pytest.mark.skipif(_a_strip() is None, reason="no segmented strips on disk")
def test_live_extract_one_strip():
    client = make_client()
    rec, usage = extract_strip(client, DEFAULT_MODEL, _a_strip(), 1)
    assert isinstance(rec, RecordExtraction)
    assert usage["input_tokens"] > 0 and usage["output_tokens"] > 0
    non_empty = sum(1 for n, r in rec.model_dump().items()
                    if isinstance(r, dict) and r.get("value"))
    assert non_empty >= 5      # a real register strip yields real content
```

- [ ] **Step 2: Run the suite** — without credentials the test SKIPS (suite stays green); with credentials it must PASS.

Run: `python -m pytest -v`
Expected: 37 passed + 1 passed-or-skipped (skip reason printed if no credentials).

- [ ] **Step 3: If credentials are present, run the mini A/B** (if not, commit the test, note the pending A/B in the report, and stop here — DONE_WITH_CONCERNS)

```bash
# 3 pages, both models (≈30 strips ≈ $0.10 total)
python 1c_extract_strips.py 20260319_053700_KAM_Stlhb_p1 20260319_053700_KAM_Stlhb_p2 20260319_053700_KAM_Stlhb_p3 --center "Kameno" --year 2026
python 1c_extract_strips.py 20260319_053700_KAM_Stlhb_p1 20260319_053700_KAM_Stlhb_p2 20260319_053700_KAM_Stlhb_p3 --model gemini-3.7-flash --center "Kameno" --year 2026
python 1d_compare_models.py --models gemini-3.5-flash-lite gemini-3.7-flash
```

- [ ] **Step 4: Sanity-check quality with vision.** Read 2–3 strips and their extracted JSON side by side; verify names/diagnoses/costs match what's visible, confidence marks look honest, and no illegible-with-value violations. Note qualitative findings.

- [ ] **Step 5: Record results in the spec.** Append `## Live results (2026-08-17)` to the spec: per-model token/cost actuals vs the planning figures, overall + worst-field agreement, disagreement count, and vision spot-check notes.

- [ ] **Step 6: Commit**

```bash
git add tests/test_live_extraction.py docs/superpowers/specs/2026-08-17-gemini-extraction-design.md
git commit -m "test: live extraction smoke + record mini A/B results"
```

---

## Self-review notes

- Spec coverage: env/auth (T1), schema+prompt+pricing (T2), per-strip call (T3), per-page resume/refusal/atomic (T4), CLI with dry-run (T5), A/B harness (T6), live smoke + mini A/B + spec results (T7). Batch API and ensemble-as-pipeline-stage are spec'd out of scope.
- Type consistency: `extract_page` signature matches T5's CLI usage (`segments_dir`, `out_base`, `context`, `force`, `limit`); `FIELD_NAMES` used by T2 test, T4 stubs, T6 compare; usage dict keys (`input_tokens`, `output_tokens`, `latency_s`) consistent across T3/T4/T6 fabricators; output file schema matches spec keys exactly.
- Known judgment calls: `resolve_auth(env_path=...)` parameter exists so tests can point at a nonexistent .env (isolating them from the developer's real `.env`); test for `test_resolve_auth_prefers_api_key` relies on monkeypatched env taking precedence, which `load_env`'s setdefault semantics guarantee. The stub client passes through `types.GenerateContentConfig` construction — a local SDK object needing no network.
