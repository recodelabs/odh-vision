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


def test_real_run_with_accounting_and_limit(tmp_path, capsys, monkeypatch):
    """Real-run test: monkeypatches extract_page and make_client; validates per-run accounting."""
    _make_segments(tmp_path, "reg_p1", n=3)
    _make_segments(tmp_path, "reg_p2", n=2)
    cli = importlib.import_module("1c_extract_strips")

    # Mock client (not used, but required by main())
    class MockClient:
        pass

    # Track calls to extract_page
    extract_calls = []

    def fake_extract_page(client, model, stem, **kwargs):
        """Mock that returns cumulative-looking records but small extracted_this_run/usage_this_run."""
        extract_calls.append(stem)
        # Simulate that we extracted only 1 record per call, but return cumulative
        # records to match the task description (simulating merge behavior)
        if stem == "reg_p1":
            return {
                "stem": stem,
                "model": model,
                "records": {"1": {}, "2": {}},  # cumulative: 2 records
                "skipped_existing": 0,
                "extracted_this_run": 1,  # but only 1 extracted this run
                "usage_this_run": {"input_tokens": 1000, "output_tokens": 500},
                "totals": {"input_tokens": 2000, "output_tokens": 1000},
            }
        elif stem == "reg_p2":
            return {
                "stem": stem,
                "model": model,
                "records": {"1": {}},  # cumulative: 1 record
                "skipped_existing": 0,
                "extracted_this_run": 1,  # only 1 extracted this run
                "usage_this_run": {"input_tokens": 1000, "output_tokens": 500},
                "totals": {"input_tokens": 1000, "output_tokens": 500},
            }

    def fake_make_client():
        return MockClient()

    monkeypatch.setattr(cli, "extract_page", fake_extract_page)
    monkeypatch.setattr(cli, "make_client", fake_make_client)

    # Run with --limit 2 (should extract from both stems but stop early)
    rc = cli.main(["--all", "--limit", "2",
                   "--segments-dir", str(tmp_path / "segments"),
                   "--out", str(tmp_path / "ex")])
    outtxt = capsys.readouterr().out

    # Should process both stems (2 calls total)
    assert len(extract_calls) == 2
    assert "reg_p1" in extract_calls
    assert "reg_p2" in extract_calls

    # Summary should report per-run extracted numbers (1 + 1 = 2), not cumulative
    assert "2 strips extracted" in outtxt
    assert "2000 in" in outtxt  # 1000 + 1000 per-run tokens
    assert "1000 out" in outtxt  # 500 + 500 per-run tokens
    assert "$" in outtxt  # cost estimate
    assert rc == 0


def test_strip_errors_counted_printed_and_fail_exit_code(tmp_path, capsys, monkeypatch):
    """Per-strip errors (contained inside extract_page, not raised) must be
    tallied into the CLI's error count, printed per page, and cause exit 1."""
    _make_segments(tmp_path, "reg_p1", n=2)
    cli = importlib.import_module("1c_extract_strips")

    class MockClient:
        pass

    def fake_extract_page(client, model, stem, **kwargs):
        return {
            "stem": stem,
            "model": model,
            "records": {"1": {}},
            "skipped_existing": 0,
            "extracted_this_run": 1,
            "usage_this_run": {"input_tokens": 100, "output_tokens": 50},
            "totals": {"input_tokens": 100, "output_tokens": 50},
            "strip_errors": [{"index": 2, "error": "400 bad request"}],
        }

    monkeypatch.setattr(cli, "extract_page", fake_extract_page)
    monkeypatch.setattr(cli, "make_client", lambda: MockClient())

    rc = cli.main(["reg_p1", "--segments-dir", str(tmp_path / "segments"),
                   "--out", str(tmp_path / "ex")])
    outtxt = capsys.readouterr().out

    assert "strip 2" in outtxt
    assert "400 bad request" in outtxt
    assert "1 errors" in outtxt
    assert rc == 1
