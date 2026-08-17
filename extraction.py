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
