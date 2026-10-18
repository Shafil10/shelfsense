"""Sprint 4 step 2: fine-tune the SKU-110K detector on OUR corrected frames and test it on our test frames.

Needs the human-corrected labels exported from Roboflow ("YOLOv8" format, unzipped). The export has
train/ and test/ (and maybe valid/) folders with images/ and labels/, and a data.yaml whose names
include product and empty_gap.

Steps:
  1. Build datasets/shelfsense_yolo: train frames split BY VIDEO into train / val (val is only used for
     early stopping), our test frames kept apart as the final test set.
  2. Fine-tune from the SKU-110K weights.
  3. Evaluate on our test frames, before (SKU-110K only) and after fine-tuning:
     - mAP for product (both models) and empty_gap (fine-tuned only: the SKU model has no such class)
     - counting error per frame: |predicted products - labelled products| (what the VLM could not do)

Output: data/results/yolo_finetune_eval.json, data/results/yolo_finetune_counts.csv

Usage (project root, venv active):
  python -m src.detection.finetune_ours --export path\\to\\roboflow_export --sku-weights runs/detect/sku110k_v8n/weights/best.pt
"""
import argparse
import csv
import json
import random
import shutil
import time
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


def video_of(image_name):
    """Our frame names start with the video name: VID-..._WA0012__frame_...jpg (Roboflow keeps the prefix)."""
    return image_name.split("__")[0]


def export_class_ids(export):
    names = yaml.safe_load((export / "data.yaml").read_text(encoding="utf-8"))["names"]
    names = list(names.values()) if isinstance(names, dict) else list(names)
    return {n: i for i, n in enumerate(names)}


def copy_split(pairs, out, split, remap, keep=None):
    """Copy (image, label) pairs into out/{images,labels}/split, remapping class ids to ours."""
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    for img, lbl in pairs:
        shutil.copy(img, out / "images" / split / img.name)
        lines = []
        for line in (lbl.read_text().splitlines() if lbl.exists() else []):
            parts = line.split()
            if parts and int(parts[0]) in remap and (keep is None or remap[int(parts[0])] in keep):
                lines.append(" ".join([str(remap[int(parts[0])])] + parts[1:5]))
        (out / "labels" / split / f"{img.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


def pairs_in(folder):
    return [(p, folder / "labels" / f"{p.stem}.txt") for p in sorted((folder / "images").glob("*.jpg"))]


def build_dataset(export, out, val_fraction, seed):
    ids = export_class_ids(export)
    remap = {ids["product"]: 0, ids["empty_gap"]: 1}
    if out.exists():
        shutil.rmtree(out)
    train_pairs = pairs_in(export / "train") + (pairs_in(export / "valid") if (export / "valid").exists() else [])
    videos = sorted({video_of(p.name) for p, _ in train_pairs})
    random.Random(seed).shuffle(videos)
    val_videos = set(videos[:max(1, round(len(videos) * val_fraction))])
    copy_split([x for x in train_pairs if video_of(x[0].name) not in val_videos], out, "train", remap)
    copy_split([x for x in train_pairs if video_of(x[0].name) in val_videos], out, "val", remap)
    copy_split(pairs_in(export / "test"), out, "test", remap)
    copy_split(pairs_in(export / "test"), out, "test_product", remap, keep={0})  # for the 1-class SKU model
    (out / "data.yaml").write_text(f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\n"
                                   "test: images/test\nnames:\n  0: product\n  1: empty_gap\n")
    (out / "data_product.yaml").write_text(f"path: {out.as_posix()}\ntrain: images/train\n"
                                           "val: images/test_product\nnames:\n  0: product\n")
    counts = {s: len(list((out / "images" / s).glob("*.jpg"))) for s in ("train", "val", "test")}
    print(f"dataset: {counts} frames; val videos: {sorted(val_videos)}")
    return counts, sorted(val_videos)


def ap_per_class(m, names):
    """mAP50 / mAP50-95 per class name from an Ultralytics validation result."""
    out = {}
    for i, c in enumerate(m.box.ap_class_index.tolist()):
        out[names[c]] = {"mAP50": round(float(m.box.ap50[i]), 3), "mAP50_95": round(float(m.box.ap[i]), 3)}
    return out


def count_errors(model, test_dir, kw):
    rows = []
    for img in sorted((test_dir / "images" / "test").glob("*.jpg")):
        lbl = test_dir / "labels" / "test" / f"{img.stem}.txt"
        true = sum(1 for line in lbl.read_text().splitlines() if line.split() and line.split()[0] == "0")
        r = model.predict(str(img), **kw)[0]
        pred = sum(1 for c in r.boxes.cls.tolist() if int(c) == 0)
        rows.append({"image": img.name, "video": video_of(img.name), "true_products": true, "pred_products": pred})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", required=True, help="unzipped Roboflow YOLOv8 export")
    ap.add_argument("--sku-weights", required=True)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--epochs", type=int, help="override finetune.epochs")
    ap.add_argument("--name", default="shelfsense_ft")
    args = ap.parse_args()

    full = yaml.safe_load(open(args.config, encoding="utf-8"))
    cfg, sku, det = full["finetune"], full["sku110k"], full["detection"]
    out = ROOT / cfg["yolo_dir"]
    device = 0 if torch.cuda.is_available() else "cpu"
    kw = {"imgsz": sku["img_size"], "conf": det["conf_threshold"], "max_det": det["max_det"],
          "device": device, "verbose": False}
    vkw = {"imgsz": sku["img_size"], "batch": sku["batch"], "max_det": det["max_det"], "device": device,
           "workers": sku["workers"],
           "plots": False, "verbose": False, "exist_ok": True}

    counts, val_videos = build_dataset(Path(args.export), out, cfg["val_fraction"], cfg["seed"])

    t = time.time()
    run_dir = ROOT / "runs" / "detect" / args.name
    YOLO(args.sku_weights).train(
        data=str(out / "data.yaml"), epochs=args.epochs or cfg["epochs"], imgsz=sku["img_size"],
        batch=sku["batch"], max_det=det["max_det"], patience=cfg["patience"], seed=cfg["seed"],
        device=device, workers=sku["workers"], project=str(run_dir.parent), name=args.name,
        exist_ok=True)
    train_minutes = round((time.time() - t) / 60, 1)

    before, after = YOLO(args.sku_weights), YOLO(run_dir / "weights" / "best.pt")
    m_before = before.val(data=str(out / "data_product.yaml"), split="val",
                          project=str(run_dir), name="test_sku110k", **vkw)
    m_after = after.val(data=str(out / "data.yaml"), split="test",
                        project=str(run_dir), name="test_finetuned", **vkw)

    rows_b, rows_a = count_errors(before, out, kw), count_errors(after, out, kw)
    with open(ROOT / "data" / "results" / "yolo_finetune_counts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "video", "true_products", "pred_sku110k", "pred_finetuned"])
        w.writeheader()
        for b, a in zip(rows_b, rows_a):
            w.writerow({"image": b["image"], "video": b["video"], "true_products": b["true_products"],
                        "pred_sku110k": b["pred_products"], "pred_finetuned": a["pred_products"]})

    def mae(rows):
        return round(sum(abs(r["pred_products"] - r["true_products"]) for r in rows) / len(rows), 2)

    result = {
        "dataset": counts, "val_videos": val_videos, "train_minutes": train_minutes,
        "test_ap_sku110k_only": ap_per_class(m_before, {0: "product"}),
        "test_ap_finetuned": ap_per_class(m_after, {0: "product", 1: "empty_gap"}),
        "count_mae_sku110k_only": mae(rows_b), "count_mae_finetuned": mae(rows_a),
        "mean_true_products_per_frame": round(sum(r["true_products"] for r in rows_b) / len(rows_b), 1),
        "weights": str((run_dir / "weights" / "best.pt").relative_to(ROOT)),
    }
    (ROOT / "data" / "results" / "yolo_finetune_eval.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
