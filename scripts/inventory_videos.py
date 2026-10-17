"""List every video in a folder with its metadata and SHA-256, and flag exact duplicates.

The SHA-256 is also the evidence-integrity fingerprint: the same file submitted twice
(e.g. a rep re-using an old video) has the same hash.

Output: a CSV with one row per video (default data/results/video_inventory.csv).

Usage:
  python scripts/inventory_videos.py --folder Test_Videos
"""
import argparse
import csv
import hashlib
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def video_meta(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"width": w, "height": h, "fps": round(fps, 2), "frames": frames,
            "duration_s": round(frames / fps, 2) if fps else 0.0,
            "orientation": "portrait" if h > w else "landscape"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", required=True)
    ap.add_argument("--out", default=str(ROOT / "data" / "results" / "video_inventory.csv"))
    args = ap.parse_args()

    videos = sorted(p for p in Path(args.folder).iterdir() if p.suffix.lower() in VIDEO_EXTS)
    rows, first_with_hash = [], {}
    for p in videos:
        digest = sha256(p)
        row = {"file": p.name, "size_mb": round(p.stat().st_size / 1e6, 2), **video_meta(p),
               "sha256": digest, "duplicate_of": first_with_hash.get(digest, "")}
        first_with_hash.setdefault(digest, p.name)
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    total = sum(r["duration_s"] for r in rows)
    dups = sum(1 for r in rows if r["duplicate_of"])
    print(f"{len(rows)} videos, {total:.0f} s total, {dups} exact duplicates -> {out}")


if __name__ == "__main__":
    main()
