"""Zoomed SKU-110K images, so the detector also learns big close-up products.

SKU-110K photos show whole shelves from a distance, so every product is small. In our own videos a
merchandiser often films close (milk cartons: median 6 boxes per frame vs 53 for canned food), and the
SKU-110K model misses those big products. Each zoomed image is a random window of 25-50 % of an original
photo's width and height (cut from the full-resolution original, so it stays sharp), saved at most 1024 px:
products become 2-4x larger than in the normal training images. A box cut by the window edge is kept
(clipped) when at least 60 % of it is inside; any other box touching the window would be an unlabelled
partial product, so windows are retried until none is cut badly (or the box is dropped as a last resort).

Writes datasets/sku110k_zoom/{images,labels}/{train,val} plus:
  data.yaml           train = full SKU-110K train + zoomed train, val = the usual 500 val photos
  data_zoomval.yaml   val = zoomed versions of the same 500 val photos (close-up test)

Usage:
  python -m src.detection.prepare_zoom --n-train 4000
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"
COLS = ["image", "x1", "y1", "x2", "y2", "class", "image_width", "image_height"]


def zoom_window(boxes, w, h, rng, tries=20, min_frac=0.25, max_frac=0.5, keep=0.6):
    """Pick a crop window; return (x0, y0, x1, y1, kept boxes in window coordinates) or None."""
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    best = None
    for _ in range(tries):
        f = rng.uniform(min_frac, max_frac)
        cw, ch = int(w * f), int(h * f)
        x0, y0 = rng.randint(0, w - cw), rng.randint(0, h - ch)
        ix1 = np.clip(boxes[:, 0], x0, x0 + cw)
        iy1 = np.clip(boxes[:, 1], y0, y0 + ch)
        ix2 = np.clip(boxes[:, 2], x0, x0 + cw)
        iy2 = np.clip(boxes[:, 3], y0, y0 + ch)
        inside = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0) / np.maximum(area, 1)
        kept = inside >= keep
        bad = int(((inside > 0.05) & ~kept).sum())     # partial products we would leave unlabelled
        if kept.sum() == 0:
            continue
        cand = (bad, (x0, y0, x0 + cw, y0 + ch,
                      np.stack([ix1[kept] - x0, iy1[kept] - y0, ix2[kept] - x0, iy2[kept] - y0], 1)))
        if best is None or cand[0] < best[0]:
            best = cand
        if bad <= 1:
            break
    return best


def build(raw_dir, csv_name, names, out_dir, split, max_side, seed):
    df = pd.read_csv(Path(raw_dir) / "annotations" / csv_name, names=COLS)
    groups = df.groupby("image")
    img_dir, lbl_dir = out_dir / "images" / split, out_dir / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    done, n_boxes, bad_total, skipped = 0, 0, 0, 0
    for name in names:
        im = cv2.imread(str(Path(raw_dir) / "images" / name))
        g = groups.get_group(name)
        w, h = int(g["image_width"].iloc[0]), int(g["image_height"].iloc[0])
        if im is None or (im.shape[1], im.shape[0]) != (w, h):
            skipped += 1
            continue
        boxes = g[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
        pick = zoom_window(boxes, w, h, rng)
        if pick is None:
            skipped += 1
            continue
        bad, (x0, y0, x1, y1, kept) = pick
        crop = im[y0:y1, x0:x1]
        cw, ch = x1 - x0, y1 - y0
        scale = max_side / max(cw, ch)
        if scale < 1:
            crop = cv2.resize(crop, (round(cw * scale), round(ch * scale)), interpolation=cv2.INTER_AREA)
        stem = Path(name).stem + "_zoom"
        cv2.imwrite(str(img_dir / f"{stem}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        lines = [f"0 {(a + c) / 2 / cw:.5f} {(b + d) / 2 / ch:.5f} {(c - a) / cw:.5f} {(d - b) / ch:.5f}"
                 for a, b, c, d in kept if c > a and d > b]
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        done += 1
        n_boxes += len(lines)
        bad_total += bad
    print(f"{split}: {done} zoomed images, {n_boxes} boxes ({n_boxes / max(done, 1):.0f} per image), "
          f"{bad_total} partial products left unlabelled ({bad_total / max(done, 1):.1f} per image), {skipped} skipped")
    return {"images": done, "boxes": n_boxes, "unlabelled_partials": bad_total, "skipped": skipped}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--base", default="datasets/sku110k_yolo_full", help="dataset whose photos are zoomed")
    ap.add_argument("--out", default="datasets/sku110k_zoom")
    ap.add_argument("--max-side", type=int, default=1024)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))["sku110k"]
    base, out = ROOT / args.base, ROOT / args.out
    subset = json.loads((base / "subset.json").read_text())
    train_names = list(subset["train"]["files"])
    random.Random(cfg["seed"] + 1).shuffle(train_names)
    info = {
        "train": build(cfg["raw_dir"] and ROOT / cfg["raw_dir"], "annotations_train.csv",
                       train_names[:args.n_train], out, "train", args.max_side, cfg["seed"]),
        "val": build(ROOT / cfg["raw_dir"], "annotations_val.csv", subset["val"]["files"], out, "val",
                     args.max_side, cfg["seed"]),
    }
    (out / "zoom.json").write_text(json.dumps(info, indent=2))
    names = {0: "product"}
    yaml.safe_dump({"train": [str(base / "images" / "train"), str(out / "images" / "train")],
                    "val": str(base / "images" / "val"), "names": names},
                   open(out / "data.yaml", "w"))
    yaml.safe_dump({"train": str(out / "images" / "train"), "val": str(out / "images" / "val"), "names": names},
                   open(out / "data_zoomval.yaml", "w"))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
