"""Tests for src/report/audit.py (shelf rows, re-use check, blur gap, full run on a synthetic video).

Run from the project root:  python -m pytest -v
"""
import json

import cv2
import numpy as np
import yaml

from src.report import audit as au


def test_shelf_rows_two_shelves():
    # Two shelves of 50 px tall products; the lower shelf is slightly tilted
    top = [(x, 100 + x // 50, x + 40, 150 + x // 50) for x in range(0, 300, 50)]
    bottom = [(x, 300 + x // 30, x + 40, 350 + x // 30) for x in range(0, 300, 50)]
    rows = au.shelf_rows(top + bottom)
    assert rows == [0] * 6 + [1] * 6


def test_shelf_rows_stacked_boxes_stay_on_one_shelf():
    # Shelf 1: three stacked layers of flat boxes touching each other; shelf 2 starts after a 30 px gap
    shelf1 = [(x, y, x + 60, y + 30) for y in (100, 130, 160) for x in range(0, 300, 60)]
    shelf2 = [(x, 220, x + 60, 250) for x in range(0, 300, 60)]
    rows = au.shelf_rows(shelf1 + shelf2)
    assert rows == [0] * 15 + [1] * 5


def test_shelf_rows_empty():
    assert au.shelf_rows([]) == []


def test_longest_blur_gap():
    log = [{"time_sec": t, "status": s} for t, s in [
        (0, "kept"), (1, "dropped_blur"), (2, "dropped_blur"), (3, "dropped_blur"), (4, "kept"),
        (5, "dropped_blur"), (6, "kept")]]
    assert au.longest_blur_gap(log) == 2.0


def test_reuse_check(tmp_path):
    old = tmp_path / "old_video"
    old.mkdir()
    hashes = ["ffff0000ffff0000", "0f0f0f0f0f0f0f0f", "123456789abcdef0"]
    (old / "audit.json").write_text(json.dumps({
        "video": {"file": "old.mp4", "sha256": "abc"}, "keyframes": [{"phash": h} for h in hashes]}))
    # Same pictures (one bit different) -> flagged; different pictures -> not flagged
    near = ["ffff0000ffff0001", "0f0f0f0f0f0f0f0f", "fedcba9876543210"]
    assert au.reuse_check("new.mp4", near, tmp_path, max_distance=4, min_share=0.5)[0]["video"] == "old.mp4"
    assert au.reuse_check("new.mp4", ["fedcba9876543210", "a5a5a5a5a5a5a5a5"], tmp_path, 4, 0.5) == []
    # A video is never compared with its own earlier report
    assert au.reuse_check("old.mp4", hashes, tmp_path, 4, 0.5) == []


def shelf_video(path, seconds=3, fps=10):
    """A static synthetic shelf with three rows of boxes (detector-independent smoke test)."""
    img = np.full((640, 360, 3), 230, np.uint8)
    rng = np.random.default_rng(0)
    for y in (60, 260, 460):
        cv2.rectangle(img, (0, y + 120), (360, y + 130), (90, 90, 90), -1)
        for x in range(10, 330, 45):
            cv2.rectangle(img, (x, y), (x + 38, y + 118), tuple(int(c) for c in rng.integers(0, 200, 3)), -1)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (360, 640))
    for _ in range(seconds * fps):
        writer.write(img)
    writer.release()


def test_audit_end_to_end(tmp_path):
    video = tmp_path / "shelf.mp4"
    shelf_video(video)
    full = yaml.safe_load(open(au.DEFAULT_CONFIG, encoding="utf-8"))
    report, out_dir = au.audit(video, "yolov8n.pt", full, device="cpu", reports_dir=tmp_path / "reports")
    assert (out_dir / "audit.json").exists()
    assert report["video"]["sha256"] == au.sha256(video)
    assert report["summary"]["keyframes"] >= 1              # a static video keeps its first frame
    assert report["summary"]["dropped_duplicate"] >= 1
    for k in report["keyframes"]:
        assert (out_dir / k["evidence"]).exists()
    assert all(v["status"] == "pending" for v in report["kpis"].values())
    # Auditing the same footage under another name is caught as re-used
    copy = tmp_path / "shelf_copy.mp4"
    copy.write_bytes(video.read_bytes())
    report2, _ = au.audit(copy, "yolov8n.pt", full, device="cpu", reports_dir=tmp_path / "reports")
    assert report2["integrity"]["verdict"] == "re-audit"
    assert report2["integrity"]["reuse_matches"][0]["video"] == "shelf.mp4"
