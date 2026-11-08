"""Can local-feature matching (src/report/reuse.py) catch cropped copies without flagging honest re-filming?

Positives: altered copies of three test videos (scripts/make_reuse_variants.py: re-encoded, 10% crop, half size)
vs the original's keyframes. Negatives: neighbouring clips of the same test blocks (filmed minutes apart, often
the same shelf) vs each other. For each pair, every keyframe of the first video gets its best match among the
second's keyframes; the table shows the medians and how many keyframes look like a copy at the thresholds.

Output: data/results/reuse_eval.json and .csv
Usage (project root, venv active):  python scripts/eval_reuse.py
"""
import csv
import json
import statistics
import sys
import tempfile
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.extraction.extract_frames import extract_video  # noqa: E402
from src.report.reuse import best_matches, prepare, same_picture  # noqa: E402

BLOCKS = [range(2, 6), range(54, 61), range(61, 67), range(89, 95), range(106, 112)]  # the test split's blocks
ORIGINALS = ["VID-20261018-WA0002", "VID-20261018-WA0061", "VID-20261018-WA0094"]


def keyframes(folder):
    return [prepare(cv2.imread(str(p))) for p in sorted(Path(folder).glob("*.jpg"))]


def report_frames(video):
    return ROOT / "data" / "reports" / video / "frames" / video


def summarise(kind, a, b, ms, cfg):
    hits = sum(same_picture(m, cfg) for m in ms)
    med = lambda k: statistics.median([m[k] for m in ms if m[k] is not None]) if any(m[k] is not None for m in ms) else None  # noqa: E731
    return {"kind": kind, "new": a, "old": b, "keyframes": len(ms), "copy_keyframes": hits, "share": round(hits / len(ms), 2) if ms else 0,
            "median_ncc": med("ncc"), "median_reproj_px": med("reproj_px"), "median_inliers": med("inliers"),
            "max_ncc": max((m["ncc"] for m in ms), default=0)}


def main():
    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml", encoding="utf-8"))
    rcfg, rows = cfg["report"], []
    with tempfile.TemporaryDirectory(dir=ROOT / "data") as tmp:  # temporary keyframes stay on E:
        for orig in ORIGINALS:
            old = keyframes(report_frames(orig))
            for variant in ("reencoded", "crop10", "half"):
                video = ROOT / "data" / "reuse_test" / f"{orig}_{variant}.mp4"
                if not video.exists():
                    continue
                extract_video(video, cfg["extraction"], out_root=Path(tmp))
                new = keyframes(Path(tmp) / video.stem)
                rows.append(summarise("copy:" + variant, video.stem, orig, best_matches(new, old), rcfg))
                print(rows[-1])
    for block in BLOCKS:
        names = [f"VID-20261018-WA{n:04d}" for n in block]
        for a, b in zip(names, names[1:]):
            if report_frames(a).exists() and report_frames(b).exists():
                rows.append(summarise("neighbour", a, b, best_matches(keyframes(report_frames(a)), keyframes(report_frames(b))), rcfg))
                print(rows[-1])
    out = ROOT / "data" / "results"
    (out / "reuse_eval.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    with open(out / "reuse_eval.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    copies = [r for r in rows if r["kind"].startswith("copy")]
    neigh = [r for r in rows if r["kind"] == "neighbour"]
    flagged = lambda r: r["share"] >= rcfg["reuse_min_share"]  # noqa: E731
    print(f"copies caught: {sum(map(flagged, copies))}/{len(copies)}; honest neighbours flagged: {sum(map(flagged, neigh))}/{len(neigh)}")


if __name__ == "__main__":
    main()
