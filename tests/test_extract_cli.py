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
