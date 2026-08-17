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
