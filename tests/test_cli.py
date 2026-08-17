import os
import json
import subprocess
import sys
import cv2
import numpy as np
from conftest import draw_table, warp_page

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_cli_segments_and_builds_contact_sheet(tmp_path):
    img, _ = draw_table(margins=True)
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


def test_cli_glob_exits_2_when_any_page_needs_review(tmp_path):
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()

    good_img, _ = draw_table(margins=True)
    cv2.imwrite(str(pages_dir / "good_p1.png"), warp_page(good_img))

    blank = np.full((850, 1200), 235, np.uint8)
    cv2.imwrite(str(pages_dir / "blank_p1.png"), blank)

    out = str(tmp_path / "segments")

    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "1b_segment_records.py"),
         "--glob", str(pages_dir / "*.png"), "--out", out],
        capture_output=True, text=True, cwd=ROOT)

    assert r.returncode == 2, r.stdout + r.stderr

    blank_dir = os.path.join(out, "blank_p1")
    assert not [f for f in os.listdir(blank_dir) if "_rec" in f]

    good_dir = os.path.join(out, "good_p1")
    assert os.path.isfile(os.path.join(good_dir, "good_p1_rec1.png"))
    with open(os.path.join(good_dir, "good_p1.json")) as f:
        assert json.load(f)["status"] == "ok"
