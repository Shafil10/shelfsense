"""Sprint 5 (first version): one shelf video -> JSON audit report with evidence.

Connects the parts built so far, all on the CPU:
  1. Integrity: SHA-256, video metadata, and a re-use check (do this video's keyframes match the
     keyframes of an earlier audit of a different video? A rep could re-submit old footage).
  2. Keyframes: the Sprint 2 extractor (sampling, blur filter, duplicate filter).
  3. Products: the Sprint 4 detector on every keyframe, with the confidence threshold that counted
     best on SKU-110K validation images (detection.count_conf). Boxes are grouped into shelves.
  4. Coverage: the longest stretch of the video lost to blur (a section the auditor cannot check).
  5. Evidence: every keyframe saved with its boxes and shelf numbers (R1 = top), so each number in the report
     points to an image and a timestamp a person can check.

KPIs that need parts not trained yet (empty_gap class, brand reading on crops) are listed with
status "pending" and the reason, instead of a guessed value.

Output: data/reports/<video>/audit.json and data/reports/<video>/evidence/*.jpg

Usage (project root, venv active):
  python -m src.report.audit --video Test_Videos/VID-20261018-WA0061.mp4      # detector: detection.weights
"""
import argparse
import csv
import hashlib
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import imagehash
import yaml

from src.extraction.extract_frames import extract_video, phash, video_info

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"
REPORTS = ROOT / "data" / "reports"
ROW_COLOURS = [(214, 120, 42), (52, 104, 235), (122, 175, 27), (0, 161, 237), (164, 123, 232), (72, 73, 227)]


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def shelf_rows(boxes, merge_gap=0.25):
    """Group boxes (x1, y1, x2, y2) into shelves, top shelf = 0. Returns one shelf index per box.

    1. Layers: boxes sorted by vertical centre; a box starts a new layer when its centre is more
       than 0.6 of the median box height below the centre of the layer being built.
    2. Shelves: boxes stacked on one shelf (toothpaste, biscuit packs) form several layers that
       touch, while the next shelf down starts after a visible gap (shelf edge, price strip). So
       neighbouring layers are merged when the gap between them (median top of the lower layer minus
       median bottom of the upper layer) is below `merge_gap` of the median box height.
    """
    if not boxes:
        return []
    med_h = statistics.median(y2 - y1 for _, y1, _, y2 in boxes)
    order = sorted(range(len(boxes)), key=lambda i: (boxes[i][1] + boxes[i][3]) / 2)
    layers, centres = [[]], []
    for i in order:
        c = (boxes[i][1] + boxes[i][3]) / 2
        if centres and c - statistics.mean(centres) > 0.6 * med_h:
            layers.append([])
            centres = []
        centres.append(c)
        layers[-1].append(i)
    rows, shelf = [0] * len(boxes), 0
    for n, layer in enumerate(layers):
        if n:
            gap = (statistics.median(boxes[i][1] for i in layer)
                   - statistics.median(boxes[i][3] for i in layers[n - 1]))
            shelf += gap >= merge_gap * med_h
        for i in layer:
            rows[i] = shelf
    return rows


def reuse_check(video_name, hashes, reports_dir, max_distance, min_share):
    """Compare this video's keyframe pHashes with the keyframes of earlier audits of OTHER videos.

    Returns the earlier audits in which at least `min_share` of this video's keyframes have a
    near-identical keyframe (Hamming distance <= max_distance), most similar first.
    """
    mine = [imagehash.hex_to_hash(h) for h in hashes]
    matches = []
    for report in Path(reports_dir).glob("*/audit.json"):
        old = json.loads(report.read_text(encoding="utf-8"))
        if old["video"]["file"] == video_name or not mine:
            continue
        theirs = [imagehash.hex_to_hash(k["phash"]) for k in old["keyframes"]]
        hit = sum(any(a - b <= max_distance for b in theirs) for a in mine)
        if hit / len(mine) >= min_share:
            matches.append({"video": old["video"]["file"], "sha256": old["video"]["sha256"],
                            "matching_keyframes": f"{hit}/{len(mine)}"})
    return sorted(matches, key=lambda m: -int(m["matching_keyframes"].split("/")[0]))


def longest_blur_gap(log_rows):
    """Longest run of consecutive sampled frames dropped as blurry, in seconds of video."""
    best, start = 0.0, None
    for r in log_rows:
        t = float(r["time_sec"])
        if r["status"] == "dropped_blur":
            start = t if start is None else start
            best = max(best, t - start)
        else:
            start = None
    return round(best, 2)


def draw_evidence(frame, boxes, rows, label):
    out = frame.copy()
    for (x1, y1, x2, y2), row in zip(boxes, rows):
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), ROW_COLOURS[row % len(ROW_COLOURS)], 2)
    for row in sorted(set(rows)):
        ys = [boxes[i][1] for i, r in enumerate(rows) if r == row]
        x = int(min(boxes[i][0] for i, r in enumerate(rows) if r == row))
        cv2.putText(out, f"R{row + 1}", (max(x - 34, 2), int(min(ys)) + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 4)
        cv2.putText(out, f"R{row + 1}", (max(x - 34, 2), int(min(ys)) + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    ROW_COLOURS[row % len(ROW_COLOURS)], 2)
    for colour, width in (((0, 0, 0), 5), ((255, 255, 255), 2)):
        cv2.putText(out, label, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, width)
    return out


def audit(video, weights, full, device="cpu", reports_dir=REPORTS):
    from ultralytics import YOLO

    start = time.perf_counter()
    video = Path(video)
    det, rep = full["detection"], full["report"]
    out_dir = Path(reports_dir) / video.stem
    (out_dir / "evidence").mkdir(parents=True, exist_ok=True)

    stats = extract_video(video, full["extraction"], out_root=out_dir / "frames")
    frame_dir = out_dir / "frames" / video.stem
    log_rows = list(csv.DictReader(open(frame_dir / "frames_log.csv", encoding="utf-8")))

    model = YOLO(weights)
    keyframes = []
    for r in (r for r in log_rows if r["status"] == "kept"):
        frame = cv2.imread(str(frame_dir / r["file"]))
        res = model.predict(frame, imgsz=full["sku110k"]["img_size"], conf=det["count_conf"],
                            max_det=det["max_det"], device=device, verbose=False)[0]
        boxes = [tuple(round(v, 1) for v in b) for b in res.boxes.xyxy.tolist()]
        rows = shelf_rows(boxes)
        t = float(r["time_sec"])
        evidence = f"evidence/{Path(r['file']).stem}.jpg"
        cv2.imwrite(str(out_dir / evidence),
                    draw_evidence(frame, boxes, rows, f"t={t:.1f}s  products: {len(boxes)}"),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
        keyframes.append({
            "time_sec": t, "frame": r["file"], "evidence": evidence, "phash": str(phash(frame)),
            "products": len(boxes),
            "products_per_shelf": [rows.count(i) for i in range(max(rows) + 1)] if rows else [],
            "boxes": [{"xyxy": b, "conf": round(c, 3), "shelf": row + 1, "cls": model.names[int(k)]}
                      for b, c, row, k in zip(boxes, res.boxes.conf.tolist(), rows, res.boxes.cls.tolist())],
        })

    counts = [k["products"] for k in keyframes]
    reuse = reuse_check(video.name, [k["phash"] for k in keyframes], reports_dir,
                        rep["reuse_hash_distance"], rep["reuse_min_share"])
    blur_gap = longest_blur_gap(log_rows)
    flags = []
    if reuse:
        flags.append({"check": "reused_footage", "severity": "critical",
                      "detail": f"keyframes match an earlier audit of {reuse[0]['video']}"})
    if blur_gap > rep["max_blur_gap_sec"]:
        flags.append({"check": "coverage", "severity": "warning",
                      "detail": f"{blur_gap} s of the video is blurred in one stretch"})
    if not keyframes:
        flags.append({"check": "coverage", "severity": "critical", "detail": "no usable keyframe"})
    elif min(counts) == 0:
        flags.append({"check": "coverage", "severity": "warning",
                      "detail": f"{counts.count(0)} keyframe(s) show no products (floor, ceiling or too close)"})

    report = {
        "report_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "video": {"file": video.name, "sha256": sha256(video), "bytes": video.stat().st_size,
                  "modified_utc": datetime.fromtimestamp(video.stat().st_mtime, timezone.utc)
                  .isoformat(timespec="seconds"), **video_info(video)},
        "pipeline": {"detector": str(weights), "classes": list(model.names.values()), "count_conf": det["count_conf"],
                     "extraction": full["extraction"], "device": device},
        "integrity": {"reuse_matches": reuse, "longest_blur_gap_sec": blur_gap, "flags": flags,
                      "verdict": "re-audit" if any(f["severity"] == "critical" for f in flags)
                      else "review" if flags else "ok"},
        "summary": {"sampled_frames": stats["sampled"], "keyframes": len(keyframes),
                    "dropped_blur": stats["dropped_blur"], "dropped_duplicate": stats["dropped_duplicate"],
                    "products_per_keyframe": {"median": statistics.median(counts) if counts else 0,
                                              "max": max(counts, default=0)},
                    "shelves_median": statistics.median([len(k["products_per_shelf"]) for k in keyframes])
                    if keyframes else 0},
        "kpis": {
            "out_of_stock": {"status": "pending", "reason": "needs the empty_gap class (fine-tuning on our frames)"},
            "share_of_shelf": {"status": "pending", "reason": "needs a brand per product crop (VLM on GPU)"},
            "paid_placement": {"status": "pending", "reason": "needs brand per crop and the shop contract"},
            "price_tags": {"status": "pending", "reason": "needs a price-tag class or OCR"},
        },
        "keyframes": keyframes,
        "processing_sec": round(time.perf_counter() - start, 1),
    }
    (out_dir / "audit.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report, out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", nargs="+", required=True)
    ap.add_argument("--weights", help="default: detection.weights in the config")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    full = yaml.safe_load(open(args.config, encoding="utf-8"))
    for v in args.video:
        report, out_dir = audit(v, args.weights or ROOT / full["detection"]["weights"], full, args.device)
        s, i = report["summary"], report["integrity"]
        print(f"{Path(v).name}: {s['keyframes']} keyframes, median {s['products_per_keyframe']['median']} products, "
              f"{s['shelves_median']} shelves, verdict {i['verdict']} {[f['check'] for f in i['flags']]} "
              f"({report['processing_sec']} s) -> {out_dir / 'audit.json'}")


if __name__ == "__main__":
    main()
