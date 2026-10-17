"""Sprint 4: run the SKU-110K detector on our frames.

1. Pre-labels: boxes for the annotation frames (data/annotation_batch/{train,test}, from
   scripts/select_annotation_frames.py), zipped in Roboflow's YOLO import format so annotators only
   correct boxes (delete wrong, add missed, draw empty_gap) instead of drawing every product.
2. --all-frames: product count on every kept keyframe in data/frames/ (input for Sprint 5).

Outputs:
  data/annotation_batch/prelabels_roboflow.zip   train/ and test/ with images + labels, data.yaml
  data/results/prelabel_counts.csv               boxes and mean confidence per annotation frame
  data/results/prelabel_examples.jpg             12 test frames with their boxes (report figure)
  data/results/detections_per_frame.csv          --all-frames only: boxes per keyframe

Usage (project root, venv active):
  python -m src.detection.prelabel --weights runs/detect/sku110k_v8n/weights/best.pt
  python -m src.detection.prelabel --weights runs/detect/sku110k_v8n/weights/best.pt --all-frames
"""
import argparse
import csv
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"
CLASSES = ["product", "empty_gap"]   # the model only predicts product; empty_gap is drawn by hand


def yolo_label_text(result):
    lines = [f"0 {x:.5f} {y:.5f} {w:.5f} {h:.5f}" for x, y, w, h in result.boxes.xywhn.tolist()]
    return "\n".join(lines) + ("\n" if lines else "")


def example_grid(model, frames, kw, n=12, cols=6, tile=(288, 512)):
    picks = frames[:: max(1, len(frames) // n)][:n]
    tiles = []
    for p in picks:
        r = model.predict(str(p), **kw)[0]
        im = cv2.resize(r.plot(labels=False, conf=False, line_width=1), tile)
        cv2.putText(im, f"{p.name.split('__')[0][-6:]}: {len(r.boxes)}", (6, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(im, f"{p.name.split('__')[0][-6:]}: {len(r.boxes)}", (6, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tiles.append(im)
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--batch-dir", default=str(ROOT / "data" / "annotation_batch"))
    ap.add_argument("--all-frames", action="store_true", help="also count products on every kept keyframe")
    args = ap.parse_args()

    full = yaml.safe_load(open(args.config, encoding="utf-8"))
    kw = {"imgsz": full["sku110k"]["img_size"], "conf": full["detection"]["conf_threshold"],
          "max_det": full["detection"]["max_det"], "device": 0 if torch.cuda.is_available() else "cpu",
          "verbose": False}
    model = YOLO(args.weights)
    batch_dir, results = Path(args.batch_dir), ROOT / "data" / "results"

    rows = []
    with tempfile.TemporaryDirectory(dir=batch_dir) as tmp:
        tmp = Path(tmp)
        for split in ("train", "test"):
            (tmp / split / "images").mkdir(parents=True)
            (tmp / split / "labels").mkdir(parents=True)
            for p in sorted((batch_dir / split).glob("*.jpg")):
                r = model.predict(str(p), **kw)[0]
                shutil.copy(p, tmp / split / "images" / p.name)
                (tmp / split / "labels" / f"{p.stem}.txt").write_text(yolo_label_text(r))
                confs = r.boxes.conf.tolist()
                rows.append({"split": split, "image": p.name, "video": p.name.split("__")[0], "boxes": len(confs),
                             "mean_conf": round(sum(confs) / len(confs), 3) if confs else ""})
        (tmp / "data.yaml").write_text(f"train: train/images\nval: test/images\nnc: {len(CLASSES)}\n"
                                       f"names: {CLASSES}\n")
        shutil.make_archive(str(batch_dir / "prelabels_roboflow"), "zip", tmp)

    with open(results / "prelabel_counts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for split in ("train", "test"):
        c = sorted(r["boxes"] for r in rows if r["split"] == split)
        print(f"{split}: {len(c)} frames, boxes per frame min {c[0]} / median {c[len(c) // 2]} / max {c[-1]}")
    cv2.imwrite(str(results / "prelabel_examples.jpg"),
                example_grid(model, sorted((batch_dir / "test").glob("*.jpg")), kw), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"-> {batch_dir / 'prelabels_roboflow.zip'}, {results / 'prelabel_counts.csv'}, prelabel_examples.jpg")

    if args.all_frames:
        out = results / "detections_per_frame.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["video", "frame", "boxes", "mean_conf"])
            for d in sorted(p for p in (ROOT / "data" / "frames").iterdir() if p.is_dir()):
                for p in sorted(d.glob("*.jpg")):
                    confs = model.predict(str(p), **kw)[0].boxes.conf.tolist()
                    w.writerow([d.name, p.name, len(confs), round(sum(confs) / len(confs), 3) if confs else ""])
        print(f"-> {out}")


if __name__ == "__main__":
    main()
