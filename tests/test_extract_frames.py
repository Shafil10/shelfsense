"""Tests for src/extraction/extract_frames.py using synthetic shelf images.

Run from the project root:  python -m pytest -v
"""
import csv
import json

import cv2
import numpy as np
import pytest

from src.extraction import extract_frames as ef

CFG = {"sample_fps": 1, "resize_width": 1280, "blur_max": 0.5, "hash_distance": 6}


def make_shelf(seed, w=640, h=480):
    """Fake shelf: rows of coloured 'bottles' with labels, different for each seed."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 230, np.uint8)
    for row_y in range(20, h - 80, 110):
        cv2.line(img, (0, row_y + 95), (w, row_y + 95), (90, 90, 90), 4)  # shelf edge
        x = 10
        while x < w - 40:
            bw = int(rng.integers(25, 50))
            color = tuple(int(c) for c in rng.integers(0, 255, 3))
            cv2.rectangle(img, (x, row_y), (x + bw, row_y + 90), color, -1)
            cv2.putText(img, str(rng.integers(10, 99)), (x + 3, row_y + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            x += bw + int(rng.integers(2, 8))
    return img


def motion_blur(img, n=15):
    k = np.zeros((n, n))
    k[n // 2, :] = 1.0 / n
    return cv2.filter2D(img, -1, k)


def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def write_video(path, frames, fps=30):
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    assert out.isOpened()
    for f in frames:
        out.write(f)
    out.release()


# ---- blur metric -----------------------------------------------------------

@pytest.mark.parametrize("seed", [1, 2, 3])
def test_blur_effect_separates_sharp_and_blurred(seed):
    sharp = make_shelf(seed)
    assert ef.blur_effect(gray(sharp)) < CFG["blur_max"]
    assert ef.blur_effect(gray(motion_blur(sharp))) > CFG["blur_max"]
    # Synthetic images start perfectly sharp (~0.06), so they need a stronger blur
    # than real phone frames (~0.3) to cross the threshold: sigma 2 gives ~0.49, sigma 3 ~0.66
    assert ef.blur_effect(gray(cv2.GaussianBlur(sharp, (0, 0), 3))) > CFG["blur_max"]


def test_blur_effect_flat_image_counts_as_blurred():
    assert ef.blur_effect(np.full((100, 100), 128, np.uint8)) == 1.0


# ---- resize -----------------------------------------------------------------

def test_resize_downscales_keeping_aspect():
    out = ef.resize_max_width(np.zeros((1080, 1920, 3), np.uint8), 1280)
    assert out.shape[:2] == (720, 1280)


def test_resize_never_upscales():
    frame = np.zeros((1024, 576, 3), np.uint8)
    assert ef.resize_max_width(frame, 1280) is frame


# ---- sampling ---------------------------------------------------------------

def test_sample_frames_by_timestamp(tmp_path):
    video = tmp_path / "v.avi"
    write_video(video, [make_shelf(0, 160, 120)] * 150)  # 5 s at 30 fps
    samples = list(ef.sample_frames(video, sample_fps=1))
    assert [i for i, _, _ in samples] == [0, 30, 60, 90, 120]
    assert [round(t) for _, t, _ in samples] == [0, 1, 2, 3, 4]


# ---- full pipeline ----------------------------------------------------------

def test_extract_video_end_to_end(tmp_path):
    """Three scenes of 3 s each; second 4 of scene B is motion-blurred.

    Sampled at 1 fps (t = 0..8):
      A0 kept, A1 dup, A2 dup, B3 kept, B4 blur, B5 dup, C6 kept, C7 dup, C8 dup
    """
    a, b, c = make_shelf(10), make_shelf(20), make_shelf(30)
    frames = [a] * 90 + [b] * 30 + [motion_blur(b)] * 30 + [b] * 30 + [c] * 90
    video = tmp_path / "walk.avi"
    write_video(video, frames)

    stats = ef.extract_video(video, CFG, out_root=tmp_path / "frames")

    assert (stats["sampled"], stats["kept"]) == (9, 3)
    assert (stats["dropped_blur"], stats["dropped_duplicate"]) == (1, 5)

    out = tmp_path / "frames" / "walk"
    assert len(list(out.glob("*.jpg"))) == 3
    assert json.loads((out / "stats.json").read_text())["kept"] == 3
    with open(out / "frames_log.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["status"] for r in rows] == [
        "kept", "dropped_duplicate", "dropped_duplicate",
        "kept", "dropped_blur", "dropped_duplicate",
        "kept", "dropped_duplicate", "dropped_duplicate",
    ]


def test_rerun_clears_old_frames(tmp_path):
    video = tmp_path / "v.avi"
    write_video(video, [make_shelf(1)] * 60)
    out = tmp_path / "frames"
    (out / "v").mkdir(parents=True)
    (out / "v" / "stale.jpg").write_bytes(b"x")
    ef.extract_video(video, CFG, out_root=out)
    assert not (out / "v" / "stale.jpg").exists()
