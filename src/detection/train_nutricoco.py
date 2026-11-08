"""Sprint 6: train the NutriCoco detector (cups, cartons, other products, empty gaps) from NutriCoco's photos.

Input: the labelled photos exported from Roboflow in "YOLOv8" format (unzipped: train/ valid/ test/ with
images/ and labels/, plus data.yaml). Class names are normalised ("NutriCoco cup" -> nutricoco_cup); the
agreement check (src/rules/agreement.py) needs at least nutricoco_cup.

Photos of one fridge on one visit look almost the same, so a random split would put near-copies in train and
test and make the test look better than it is. --regroup therefore splits again by GROUP: the part of the file
name before "__" (the photo guide asks for <shop>_<date>__<n>.jpg), 70 / 15 / 15, fixed seed.

Steps: build datasets/nutricoco_yolo -> fine-tune from the shelf detector (it already finds products; it only has
to learn which ones are NutriCoco's) -> test: mAP per class and the cup-counting error per photo.
Output: runs/detect/nutricoco/weights/best.pt (the server uses it automatically: detection.nutricoco_weights)
and data/results/nutricoco_eval.json.

Usage (project root, venv active; GPU PC: batch 2 at 1024 px fits in 4 GB):
  python -m src.detection.train_nutricoco --export path\\to\\roboflow_export --regroup
"""
import argparse
import json
import random
import re
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def canonical(name):
    """'NutriCoco Cup' / 'nutricoco-cup' -> 'nutricoco_cup'."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def group_of(stem):
    """Photos of one visit share the part before '__'; Roboflow adds '_jpg.rf.<hash>' after the original name."""
    stem = re.sub(r"_(jpe?g|png)\.rf\.[0-9a-f]+$", "", stem, flags=re.I)
    return stem.split("__")[0] if "__" in stem else stem


def read_export(export):
    """[(image, label file or None, split)] and the export's class names."""
    names = yaml.safe_load((export / "data.yaml").read_text(encoding="utf-8"))["names"]
    names = list(names.values()) if isinstance(names, dict) else list(names)
    pairs = []
    for split in ("train", "valid", "test"):
        for img in sorted((export / split / "images").glob("*")) if (export / split / "images").exists() else []:
            if img.suffix.lower() in IMAGE_EXT:
                lab = export / split / "labels" / (img.stem + ".txt")
                pairs.append((img, lab if lab.exists() else None, split))
    return pairs, [canonical(n) for n in names]


def regroup(pairs, seed, fractions=(0.7, 0.15)):
    groups = sorted({group_of(p[0].stem) for p in pairs})
    random.Random(seed).shuffle(groups)
    n_train = max(1, round(len(groups) * fractions[0]))
    n_val = max(1, round(len(groups) * fractions[1])) if len(groups) > 2 else 0
    split_of = {g: "train" if i < n_train else "valid" if i < n_train + n_val else "test" for i, g in enumerate(groups)}
    return [(img, lab, split_of[group_of(img.stem)]) for img, lab, _ in pairs]


def build(pairs, names, out):
    if out.exists():
        shutil.rmtree(out)
    counts = {}
    for img, lab, split in pairs:
        for sub in ("images", "labels"):
            (out / split / sub).mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, out / split / "images" / img.name)
        (out / split / "labels" / (img.stem + ".txt")).write_text(lab.read_text(encoding="utf-8") if lab else "", encoding="utf-8")
        counts[split] = counts.get(split, 0) + 1
    data = {"path": str(out), "train": "train/images", "val": "valid/images", "test": "test/images", "names": dict(enumerate(names))}
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return counts


def count_error(model, test_images, cup_id, kw):
    """Mean |predicted cups - labelled cups| per test photo (the number the agreement check relies on)."""
    errors = []
    for img in sorted(test_images.glob("*")):
        if img.suffix.lower() not in IMAGE_EXT:
            continue
        lab = test_images.parent / "labels" / (img.stem + ".txt")
        truth = sum(1 for line in lab.read_text(encoding="utf-8").splitlines() if line.split() and int(line.split()[0]) == cup_id)
        pred = sum(int(c) == cup_id for c in model.predict(str(img), verbose=False, **kw)[0].boxes.cls.tolist())
        errors.append(abs(pred - truth))
    return round(sum(errors) / len(errors), 2) if errors else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", required=True, help="unzipped Roboflow YOLOv8 export")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--regroup", action="store_true", help="split again by photo group (recommended)")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--device", default=None, help="0 for the GPU, cpu")
    ap.add_argument("--out", default="datasets/nutricoco_yolo")
    ap.add_argument("--name", default="nutricoco")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    nc = cfg["nutricoco_train"]
    from ultralytics import YOLO

    pairs, names = read_export(Path(args.export))
    if "nutricoco_cup" not in names:
        raise SystemExit(f"the export has no 'nutricoco_cup' class (classes: {names}); name the cup class 'NutriCoco cup' in Roboflow")
    if args.regroup:
        pairs = regroup(pairs, nc["seed"])
    out = ROOT / args.out
    counts = build(pairs, names, out)
    print("photos per split:", counts, "| classes:", names)
    if not counts.get("test"):
        raise SystemExit("no test photos: export with a test split or use --regroup")

    model = YOLO(str(ROOT / cfg["detection"]["weights"]))  # start from the shelf detector
    device = args.device if args.device is not None else (0 if __import__("torch").cuda.is_available() else "cpu")
    model.train(data=str(out / "data.yaml"), epochs=args.epochs or nc["epochs"], imgsz=args.imgsz or nc["img_size"],
                batch=args.batch or nc["batch"], workers=nc["workers"], patience=nc["patience"], device=device,
                project=str(ROOT / "runs" / "detect"), name=args.name, exist_ok=True, seed=nc["seed"], plots=True)
    best = ROOT / "runs" / "detect" / args.name / "weights" / "best.pt"
    tuned = YOLO(str(best))
    kw = {"imgsz": args.imgsz or nc["img_size"], "conf": nc["count_conf"], "max_det": cfg["detection"]["max_det"], "device": device}
    m = tuned.val(data=str(out / "data.yaml"), split="test", imgsz=kw["imgsz"], batch=1, device=device, plots=False,
                  project=str(ROOT / "runs" / "detect"), name=args.name + "_test", exist_ok=True)
    per_class = {names[c]: round(float(m.box.maps[c]), 3) for c in range(len(names))}
    result = {"photos": counts, "classes": names, "mAP50": round(float(m.box.map50), 3), "mAP50_95": round(float(m.box.map), 3),
              "mAP50_95_per_class": per_class, "cup_count_mae": count_error(tuned, out / "test" / "images", names.index("nutricoco_cup"), kw),
              "count_conf": nc["count_conf"], "weights": str(best.relative_to(ROOT))}
    (ROOT / "data" / "results").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "results" / "nutricoco_eval.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))
    if best == ROOT / cfg["detection"]["nutricoco_weights"]:
        print(f"The server uses {best} automatically after a restart (detection.nutricoco_weights).")
    else:
        print(f"Trained under another name: set detection.nutricoco_weights to {best.relative_to(ROOT)} to use it.")


if __name__ == "__main__":
    main()
