import json
import os
import time

import pytest
from pydantic import ValidationError

import extraction as ex
from extraction import (load_env, resolve_auth, Reading, RecordExtraction,
                        FIELD_NAMES, build_prompt, estimate_cost)


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


# ─── Per-strip extraction tests (stub client, offline) ──────────────────────────


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


def test_extract_strip_retries_on_502(strip_file):
    """502 Bad Gateway is retryable and should eventually succeed."""
    client = StubClient([RuntimeError("502 Bad Gateway"),
                         _Resp(_valid_record())])
    naps = []
    rec, _ = ex.extract_strip(client, "m", strip_file, 1, sleep=naps.append)
    assert isinstance(rec, RecordExtraction)
    assert naps == [1]                         # 2**0


def test_is_retryable_word_boundary_no_false_positive_substring():
    """A message containing '15000' (not a standalone retryable code) must
    not be matched via substring containment on '500'."""
    err = RuntimeError("quota exceeded: capacity 15000 requests")
    assert ex._is_retryable(err) is False


def test_is_retryable_numeric_code_attribute():
    class CodedError(Exception):
        code = 503
    assert ex._is_retryable(CodedError("service unavailable")) is True


def test_is_retryable_numeric_code_attribute_non_retryable():
    class CodedError(Exception):
        code = 400
    assert ex._is_retryable(CodedError("bad request")) is False


def test_is_retryable_502_string_still_matches():
    assert ex._is_retryable(RuntimeError("502 Bad Gateway")) is True


def test_extract_strip_15000_in_message_not_retried(strip_file):
    """End-to-end: a non-retryable error whose message merely contains the
    substring '15000' must raise immediately, not be retried as a 500."""
    client = StubClient([RuntimeError("quota exceeded: capacity 15000 requests")])
    with pytest.raises(RuntimeError, match="15000"):
        ex.extract_strip(client, "m", strip_file, 1, sleep=lambda s: None)
    assert len(client.models.calls) == 1


def test_extract_strip_retries_on_numeric_code_attribute(strip_file):
    """End-to-end: an exception exposing .code == 503 is retried and succeeds."""
    class CodedError(Exception):
        code = 503

    client = StubClient([CodedError("service unavailable"), _Resp(_valid_record())])
    naps = []
    rec, _ = ex.extract_strip(client, "m", strip_file, 1, sleep=naps.append)
    assert isinstance(rec, RecordExtraction)
    assert naps == [1]                         # 2**0


def test_extract_strip_thinking_config_drop_does_not_consume_attempt(strip_file):
    """Thinking config rejection should retry without consuming an attempt slot."""
    client = StubClient([
        RuntimeError("thinking_config not supported"),
        RuntimeError("503 unavailable"),
        RuntimeError("503 unavailable"),
        RuntimeError("503 unavailable"),
        _Resp(_valid_record())
    ])
    naps = []
    rec, _ = ex.extract_strip(client, "m", strip_file, 1, sleep=naps.append, max_attempts=4)
    assert isinstance(rec, RecordExtraction)
    # Should succeed because thinking config drop doesn't consume attempt:
    # attempt 0: thinking config rejected (no sleep, no increment)
    # attempt 0: 503 (sleep 1, increment to 1)
    # attempt 1: 503 (sleep 2, increment to 2)
    # attempt 2: 503 (sleep 4, increment to 3)
    # attempt 3: success
    assert naps == [1, 2, 4]


# ─── Per-page extraction tests (stub client, offline) ────────────────────────


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
    result = ex.extract_page(client, "gemini-3.5-flash-lite", "reg_p1",
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
    result = ex.extract_page(client, "m", "reg_p1", segments_dir=seg, out_base=out)
    assert result["refused"] == "needs_review"
    assert not os.path.exists(os.path.join(out, "m", "reg_p1.json"))
    assert client.models.calls == []


def test_extract_page_resumes_and_forces(tmp_path):
    seg = _make_segments(tmp_path)
    out = str(tmp_path / "ex")
    c1 = StubClient([_Resp(_valid_record()) for _ in range(3)])
    ex.extract_page(c1, "m", "reg_p1", segments_dir=seg, out_base=out,
                    sleep=lambda s: None)
    # resume: nothing new to do
    c2 = StubClient([])
    r2 = ex.extract_page(c2, "m", "reg_p1", segments_dir=seg, out_base=out,
                         sleep=lambda s: None)
    assert c2.models.calls == [] and r2["skipped_existing"] == 3
    # force: re-extracts all
    c3 = StubClient([_Resp(_valid_record()) for _ in range(3)])
    r3 = ex.extract_page(c3, "m", "reg_p1", segments_dir=seg, out_base=out,
                         force=True, sleep=lambda s: None)
    assert len(c3.models.calls) == 3 and r3["skipped_existing"] == 0


def test_extract_page_limit_caps_new_extractions(tmp_path):
    seg = _make_segments(tmp_path)
    out = str(tmp_path / "ex")
    client = StubClient([_Resp(_valid_record())])
    ex.extract_page(client, "m", "reg_p1", segments_dir=seg, out_base=out,
                    limit=1, sleep=lambda s: None)
    saved = json.load(open(os.path.join(out, "m", "reg_p1.json")))
    assert len(saved["records"]) == 1


def test_extract_page_force_failure_preserves_prior_record_and_continues(tmp_path):
    """Force re-extraction with a non-retryable failure on one strip: that
    strip's prior value is preserved (not overwritten, stays re-tryable on
    resume) and later strips in the page still get processed."""
    seg = _make_segments(tmp_path, n=3)
    out = str(tmp_path / "ex")
    # First extraction: all 3 records succeed
    c1 = StubClient([_Resp(_valid_record()) for _ in range(3)])
    ex.extract_page(c1, "m", "reg_p1", segments_dir=seg, out_base=out,
                    sleep=lambda s: None)
    path = os.path.join(out, "m", "reg_p1.json")
    saved1 = json.load(open(path))
    assert len(saved1["records"]) == 3
    # Force re-extraction; 2nd strip fails non-retryably, 1st and 3rd succeed.
    c2 = StubClient([_Resp(_valid_record()), ValueError("400 bad request"),
                     _Resp(_valid_record())])
    r2 = ex.extract_page(c2, "m", "reg_p1", segments_dir=seg, out_base=out,
                         force=True, sleep=lambda s: None)
    # Failure is contained: reported transiently, not raised, and page keeps going.
    assert r2["strip_errors"] == [{"index": 2, "error": "400 bad request"}]
    assert r2["extracted_this_run"] == 2       # strips 1 and 3
    # File still has all 3 record keys (1 and 3 refreshed, 2 kept from prior).
    saved2 = json.load(open(path))
    assert set(saved2["records"].keys()) == {"1", "2", "3"}
    # JSON is valid (no partial writes or truncation)
    assert "totals" in saved2
    assert saved2["stem"] == "reg_p1"


def test_extract_page_strip_errors_default_empty_on_success(tmp_path):
    seg = _make_segments(tmp_path)
    out = str(tmp_path / "ex")
    client = StubClient([_Resp(_valid_record()) for _ in range(3)])
    result = ex.extract_page(client, "m", "reg_p1", segments_dir=seg,
                             out_base=out, sleep=lambda s: None)
    assert result["strip_errors"] == []
    # Transient key, never persisted to disk.
    saved = json.load(open(os.path.join(out, "m", "reg_p1.json")))
    assert "strip_errors" not in saved


def test_extract_page_failed_strip_stays_retryable_on_resume(tmp_path):
    """A strip that failed is absent from page["records"], so a resume run
    (without force) retries it instead of skipping it forever."""
    seg = _make_segments(tmp_path, n=2)
    out = str(tmp_path / "ex")
    c1 = StubClient([ValueError("400 bad request"), _Resp(_valid_record())])
    r1 = ex.extract_page(c1, "m", "reg_p1", segments_dir=seg, out_base=out,
                         sleep=lambda s: None)
    assert r1["strip_errors"] == [{"index": 1, "error": "400 bad request"}]
    saved1 = json.load(open(os.path.join(out, "m", "reg_p1.json")))
    assert set(saved1["records"].keys()) == {"2"}
    # Resume: strip 1 (previously failed) is retried, strip 2 is skipped.
    c2 = StubClient([_Resp(_valid_record())])
    r2 = ex.extract_page(c2, "m", "reg_p1", segments_dir=seg, out_base=out,
                         sleep=lambda s: None)
    assert r2["strip_errors"] == []
    assert r2["extracted_this_run"] == 1
    assert r2["skipped_existing"] == 1
    saved2 = json.load(open(os.path.join(out, "m", "reg_p1.json")))
    assert set(saved2["records"].keys()) == {"1", "2"}


def test_extract_page_empty_records_writes_output(tmp_path):
    """An ok page with no records in manifest still produces an output file."""
    seg = _make_segments(tmp_path, n=0)  # 0 records
    out = str(tmp_path / "ex")
    client = StubClient([])
    result = ex.extract_page(client, "m", "reg_p1", segments_dir=seg, out_base=out)
    path = os.path.join(out, "m", "reg_p1.json")
    assert os.path.isfile(path)
    saved = json.load(open(path))
    assert saved["records"] == {}
    assert saved["totals"]["input_tokens"] == 0
    assert result["skipped_existing"] == 0


def test_extract_page_resume_returns_zero_extracted_this_run(tmp_path):
    """Resume run (all records exist) returns extracted_this_run == 0 and usage_this_run of zeros."""
    seg = _make_segments(tmp_path, n=3)
    out = str(tmp_path / "ex")
    # First extraction: all 3 records
    c1 = StubClient([_Resp(_valid_record()) for _ in range(3)])
    r1 = ex.extract_page(c1, "m", "reg_p1", segments_dir=seg, out_base=out,
                         sleep=lambda s: None)
    assert r1["extracted_this_run"] == 3
    assert r1["usage_this_run"]["input_tokens"] > 0
    # Resume: no new records
    c2 = StubClient([])
    r2 = ex.extract_page(c2, "m", "reg_p1", segments_dir=seg, out_base=out,
                         sleep=lambda s: None)
    assert r2["extracted_this_run"] == 0
    assert r2["usage_this_run"]["input_tokens"] == 0
    assert r2["usage_this_run"]["output_tokens"] == 0


def test_extract_page_force_with_limit_tracks_extracted_correctly(tmp_path):
    """Force run with limit=1 extracts 1 while file holds all 3."""
    seg = _make_segments(tmp_path, n=3)
    out = str(tmp_path / "ex")
    # First extraction: all 3 records
    c1 = StubClient([_Resp(_valid_record()) for _ in range(3)])
    ex.extract_page(c1, "m", "reg_p1", segments_dir=seg, out_base=out,
                    sleep=lambda s: None)
    path = os.path.join(out, "m", "reg_p1.json")
    saved1 = json.load(open(path))
    assert len(saved1["records"]) == 3
    # Force re-extract with limit=1
    c2 = StubClient([_Resp(_valid_record())])
    r2 = ex.extract_page(c2, "m", "reg_p1", segments_dir=seg, out_base=out,
                         force=True, limit=1, sleep=lambda s: None)
    assert r2["extracted_this_run"] == 1
    # File still has all 3 records (record 1 was refreshed, 2 and 3 from prior)
    saved2 = json.load(open(path))
    assert len(saved2["records"]) == 3
