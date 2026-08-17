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
