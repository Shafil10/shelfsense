"""Sprint 4: pick a small, varied set of keyframes to annotate for YOLO training.

Kept keyframes from one video still overlap a lot, so annotating all of them wastes time.
For each video in the manifest this script splits its kept frames into `--per-video`
equal parts along time and takes the sharpest frame (lowest blur effect) of each part.

Input:
  data/annotations/video_manifest.csv   video, split (train / test / exclude_*), ...
  data/frames/<video>/frames_log.csv    written by src/extraction/extract_frames.py
Output:
  data/annotation_batch/<split>/<video>__<frame>.jpg   ready to upload to Roboflow / CVAT
  data/annotation_batch/selection.csv                   which frame came from where

Usage:
  python scripts/select_annotation_frames.py --per-video 2
"""
import argparse
import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pick_frames(log_rows, n):
    """Sharpest kept frame in each of n equal time parts of the video."""
    kept = [r for r in log_rows if r["status"] == "kept"]
    if len(kept) <= n:
        return kept
    picks = []
    for i in range(n):
        part = kept[i * len(kept) // n:(i + 1) * len(kept) // n]
        picks.append(min(part, key=lambda r: float(r["blur_effect"])))
    return picks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(ROOT / "data" / "annotations" / "video_manifest.csv"))
    ap.add_argument("--frames", default=str(ROOT / "data" / "frames"))
    ap.add_argument("--out", default=str(ROOT / "data" / "annotation_batch"))
    ap.add_argument("--per-video", type=int, default=2)
    args = ap.parse_args()

    out = Path(args.out)
    selection = []
    for m in csv.DictReader(open(args.manifest, encoding="utf-8")):
        if m["split"] not in ("train", "test"):
            continue
        frame_dir = Path(args.frames) / m["video"]
        log = list(csv.DictReader(open(frame_dir / "frames_log.csv", encoding="utf-8")))
        for r in pick_frames(log, args.per_video):
            dest = out / m["split"] / f"{m['video']}__{r['file']}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(frame_dir / r["file"], dest)
            selection.append({"split": m["split"], "video": m["video"], "category": m["category"],
                              "frame": r["file"], "time_sec": r["time_sec"],
                              "blur_effect": r["blur_effect"], "image": dest.name})

    with open(out / "selection.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(selection[0]))
        w.writeheader()
        w.writerows(selection)
    for split in ("train", "test"):
        print(f"{split}: {sum(s['split'] == split for s in selection)} frames -> {out / split}")


if __name__ == "__main__":
    main()
