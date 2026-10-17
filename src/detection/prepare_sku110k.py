"""Sprint 4: turn a random subset of SKU-110K into a YOLO dataset (one class: product).

SKU-110K's CSVs have no header: image, x1, y1, x2, y2, class, image_width, image_height (pixel corners).
YOLO wants one .txt per image with `class x_center y_center width height`, divided by the image size.
Every box becomes class 0 = product (no brands, as planned). Images that are unreadable, or whose
size does not match the CSV (rotated by EXIF), are skipped rather than guessed.

Input:  datasets/SKU110K_fixed/{images,annotations}   (unpack SKU110K_fixed.tar.gz first, see below)
Output: datasets/sku110k_yolo/{images,labels}/{train,val}, data.yaml, subset.json

Usage (PowerShell, project root):
  tar -xzf datasets\\SKU110K_fixed.tar.gz -C datasets
  python -m src.detection.prepare_sku110k
"""
import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"
COLS = ["image", "x1", "y1", "x2", "y2", "class", "image_width", "image_height"]


def yolo_lines(group, w, h):
    """CSV rows of one image -> YOLO label lines. Boxes are clipped to the image; empty ones dropped."""
    x1, x2 = group["x1"].clip(0, w), group["x2"].clip(0, w)
    y1, y2 = group["y1"].clip(0, h), group["y2"].clip(0, h)
    ok = (x2 > x1) & (y2 > y1)
    return [f"0 {(a + c) / 2 / w:.5f} {(b + d) / 2 / h:.5f} {(c - a) / w:.5f} {(d - b) / h:.5f}"
            for a, b, c, d in zip(x1[ok], y1[ok], x2[ok], y2[ok])]


def convert(raw_dir, csv_name, out_dir, split, n, max_side, seed):
    df = pd.read_csv(Path(raw_dir) / "annotations" / csv_name, names=COLS)
    names = sorted(df["image"].unique())
    random.Random(seed).shuffle(names)
    groups = df.groupby("image")
    img_dir, lbl_dir = Path(out_dir) / "images" / split, Path(out_dir) / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    done, skipped, n_boxes, used = 0, 0, 0, []
    for name in names:
        if done == n:
            break
        im = cv2.imread(str(Path(raw_dir) / "images" / name))
        g = groups.get_group(name)
        w, h = int(g["image_width"].iloc[0]), int(g["image_height"].iloc[0])
        if im is None or (im.shape[1], im.shape[0]) != (w, h):
            skipped += 1
            continue
        lines = yolo_lines(g, w, h)
        scale = max_side / max(w, h)
        if scale < 1:  # labels are relative to the image size, so shrinking does not change them
            im = cv2.resize(im, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(img_dir / name), im, [cv2.IMWRITE_JPEG_QUALITY, 90])
        (lbl_dir / name).with_suffix(".txt").write_text("\n".join(lines) + "\n")
        done += 1
        n_boxes += len(lines)
        used.append(name)
    print(f"{split}: {done} images, {n_boxes} boxes ({n_boxes / max(done, 1):.0f} per image), {skipped} skipped")
    return {"images": done, "boxes": n_boxes, "skipped": skipped, "files": used}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))["sku110k"]
    raw, out = ROOT / cfg["raw_dir"], ROOT / cfg["yolo_dir"]
    if out.exists():
        shutil.rmtree(out)
    subset = {
        "train": convert(raw, "annotations_train.csv", out, "train", cfg["n_train"], cfg["max_side"], cfg["seed"]),
        "val": convert(raw, "annotations_val.csv", out, "val", cfg["n_val"], cfg["max_side"], cfg["seed"]),
    }
    (out / "data.yaml").write_text(f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\n"
                                   "names:\n  0: product\n")
    (out / "subset.json").write_text(json.dumps(subset, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
