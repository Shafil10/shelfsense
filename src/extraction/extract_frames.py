"""Sprint 2: extract sharp, non-duplicate keyframes from a shelf walkthrough video.

Steps for each video (frames are processed one at a time, never all in memory):
  1. Sample frames by timestamp at `sample_fps` (robust to phone variable frame rate).
  2. Downscale to at most `resize_width` pixels wide (never upscale).
  3. Blur filter: drop a frame if its blur effect score (Crete et al. 2007) is above
     `blur_max`. The score is 0 (sharp) to 1 (fully blurred) and compares the frame
     with a re-blurred copy of itself, so it barely depends on scene content.
     Laplacian variance was tried first but depends heavily on content: on our
     fridge videos sharp close-ups scored 49 while motion-blurred frames scored 147.
     Laplacian variance is still logged for comparison.
  4. Duplicate filter: drop a frame if its perceptual hash is within
     `hash_distance` of the last kept frame.

Outputs in data/frames/<video_name>/:
  - kept frames as JPEG
  - frames_log.csv: one row per sampled frame with its scores and why it was kept/dropped
  - stats.json: summary counts for the weekly report

Usage:
  python -m src.extraction.extract_frames --video data/raw_videos/fridge_01.mp4
  python -m src.extraction.extract_frames --video data/raw_videos          # every video in a folder
  python -m src.extraction.extract_frames --video data/raw_videos --fps 4  # override config
"""
import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import cv2
import imagehash
import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"
DEFAULT_OUT = ROOT / "data" / "frames"
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def load_config(path=DEFAULT_CONFIG):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["extraction"]


def sample_frames(video_path, sample_fps):
    """Yield (frame_index, time_sec, frame) at roughly `sample_fps` frames per second.

    Frames are chosen by timestamp, not frame index, because phone videos often
    have a variable frame rate. Skipped frames are only grabbed, not decoded, which is faster.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    step = 1.0 / sample_fps
    next_t = 0.0
    index = -1
    try:
        while cap.grab():
            index += 1
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if t + 1e-6 < next_t:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                continue
            yield index, t, frame
            # Advance by whole steps so a long gap in timestamps does not cause a burst
            while next_t <= t + 1e-6:
                next_t += step
    finally:
        cap.release()


def video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": round(fps, 2),
        "frame_count": n,
        "duration_sec": round(n / fps, 2) if fps else None,
    }
    cap.release()
    return info


def resize_max_width(frame, max_width):
    """Downscale so width <= max_width, keeping aspect ratio. Never upscales."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (max_width, round(h * scale)), interpolation=cv2.INTER_AREA)


def laplacian_var(gray):
    """Variance of the Laplacian (classic sharpness score). Logged only, see module docstring."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def blur_effect(gray, h=11):
    """No-reference blur score from Crete et al. (2007): 0 = sharp, 1 = fully blurred.

    Blur the image again with an h-pixel box filter and measure how much of the
    neighbouring-pixel variation is lost. A sharp image loses a lot, an already
    blurry image loses little. Computed horizontally and vertically; the worse one wins.
    """
    f = gray.astype(np.float64)
    scores = []
    for axis, ksize in ((1, (h, 1)), (0, (1, h))):  # axis 1 = horizontal, 0 = vertical
        d_orig = np.abs(np.diff(f, axis=axis))
        d_blur = np.abs(np.diff(cv2.blur(f, ksize), axis=axis))
        lost = np.maximum(0, d_orig - d_blur).sum()
        total = d_orig.sum()
        scores.append((total - lost) / total if total > 0 else 1.0)
    return float(max(scores))


def phash(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return imagehash.phash(Image.fromarray(rgb))


def extract_video(video_path, cfg, out_root=DEFAULT_OUT):
    """Run the full extraction on one video. Returns the stats dict."""
    video_path = Path(video_path)
    out_dir = Path(out_root) / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):  # clear frames from a previous run
        old.unlink()

    last_hash = None
    rows = []
    counts = {"sampled": 0, "kept": 0, "dropped_blur": 0, "dropped_duplicate": 0}
    start = time.perf_counter()

    for index, t, frame in sample_frames(video_path, cfg["sample_fps"]):
        counts["sampled"] += 1
        frame = resize_max_width(frame, cfg["resize_width"])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = blur_effect(gray)
        is_blurry = score > cfg["blur_max"]
        row = {
            "frame_index": index,
            "time_sec": round(t, 3),
            "blur_effect": round(score, 3),
            "laplacian_var": round(laplacian_var(gray), 1),
            "hash_distance": "",
            "status": "",
            "file": "",
        }

        if is_blurry:
            row["status"] = "dropped_blur"
        else:
            h = phash(frame)
            dist = h - last_hash if last_hash is not None else None
            row["hash_distance"] = dist if dist is not None else ""
            if dist is not None and dist < cfg["hash_distance"]:
                row["status"] = "dropped_duplicate"
            else:
                row["status"] = "kept"
                name = f"frame_{index:06d}_t{t:07.2f}.jpg"
                cv2.imwrite(str(out_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                row["file"] = name
                last_hash = h
        counts[row["status"]] += 1
        rows.append(row)

    with open(out_dir / "frames_log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["frame_index"])
        writer.writeheader()
        writer.writerows(rows)

    scores = [r["blur_effect"] for r in rows]
    kept_shape = None
    if counts["kept"]:
        first_kept = next(r["file"] for r in rows if r["status"] == "kept")
        kept_shape = list(cv2.imread(str(out_dir / first_kept)).shape[:2][::-1])
    stats = {
        "video": video_path.name,
        "source": video_info(video_path),
        "settings": {k: cfg[k] for k in sorted(cfg)},
        **counts,
        "keep_rate": round(counts["kept"] / counts["sampled"], 3) if counts["sampled"] else 0.0,
        "output_size_wh": kept_shape,
        "blur_effect_min": min(scores) if scores else None,
        "blur_effect_median": round(statistics.median(scores), 3) if scores else None,
        "blur_effect_max": max(scores) if scores else None,
        "processing_sec": round(time.perf_counter() - start, 2),
    }
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats


def find_videos(paths):
    videos = []
    for p in map(Path, paths):
        if p.is_dir():
            videos += sorted(v for v in p.iterdir() if v.suffix.lower() in VIDEO_EXTS)
        else:
            videos.append(p)
    return videos


def main():
    parser = argparse.ArgumentParser(description="Extract keyframes from shelf videos.")
    parser.add_argument("--video", nargs="+", required=True, help="video file(s) or folder(s)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output root folder")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--fps", type=float, help="override sample_fps from config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.fps:
        cfg["sample_fps"] = args.fps

    for video in find_videos(args.video):
        s = extract_video(video, cfg, args.out)
        print(
            f"{s['video']}: {s['source']['duration_sec']}s, sampled {s['sampled']}, kept {s['kept']}, "
            f"blurry {s['dropped_blur']}, duplicate {s['dropped_duplicate']} "
            f"({s['processing_sec']}s) -> {Path(args.out) / video.stem}"
        )


if __name__ == "__main__":
    main()
