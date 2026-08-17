import os
import subprocess
import sys
import cv2
from conftest import draw_table, warp_page

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_cli_segments_and_builds_contact_sheet(tmp_path):
    img, _ = draw_table()
    page = str(tmp_path / "reg_p1.png")
    cv2.imwrite(page, warp_page(img))
    out = str(tmp_path / "segments")

    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "1b_segment_records.py"),
         page, "--out", out, "--contact-sheet"],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "ok" in r.stdout
    assert os.path.isfile(os.path.join(out, "reg_p1", "reg_p1_rec1.png"))
    assert os.path.isfile(os.path.join(out, "contact_sheet.jpg"))
