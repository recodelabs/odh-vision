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
