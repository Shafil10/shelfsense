"""Sprint 4: train YOLOv8n on the SKU-110K subset and measure mAP on its validation images.

Starts from the COCO-pretrained nano model (configs/config.yaml: detection.model). Training curves
and weights go to runs/detect/sku110k_v8n/. After training the best weights are validated a few
ways so the report can show what the settings matter:
  - img size 1024 (training size) vs 640 (Ultralytics default)
  - max_det 1000 vs the Ultralytics default 300 (packed shelves have more products than 300)

Output: data/results/yolo_<name>_eval.json

Usage (project root, venv active; needs src.detection.prepare_sku110k first):
  python -m src.detection.train_sku110k
  python -m src.detection.train_sku110k --epochs 1 --name smoke     # quick check that it runs
  python -m src.detection.train_sku110k --eval-only                 # re-validate existing weights
"""
import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


def box_metrics(m):
    return {"mAP50": round(float(m.box.map50), 3), "mAP50_95": round(float(m.box.map), 3),
            "precision": round(float(m.box.mp), 3), "recall": round(float(m.box.mr), 3)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--epochs", type=int, help="override sku110k.epochs")
    ap.add_argument("--name", default="sku110k_v8n")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    full = yaml.safe_load(open(args.config, encoding="utf-8"))
    cfg, det = full["sku110k"], full["detection"]
    data = ROOT / cfg["yolo_dir"] / "data.yaml"
    run_dir = ROOT / "runs" / "detect" / args.name
    device = 0 if torch.cuda.is_available() else "cpu"
    epochs = args.epochs or cfg["epochs"]

    train_minutes = None
    if not args.eval_only:
        t = time.time()
        YOLO(det["model"]).train(
            data=str(data), epochs=epochs, imgsz=cfg["img_size"], batch=cfg["batch"],
            max_det=det["max_det"], patience=cfg["patience"], seed=cfg["seed"], device=device,
            workers=cfg["workers"], cache="disk", project=str(run_dir.parent), name=args.name,
            exist_ok=True)
        train_minutes = round((time.time() - t) / 60, 1)

    model = YOLO(run_dir / "weights" / "best.pt")
    evals = {}
    for imgsz, max_det in ((cfg["img_size"], det["max_det"]), (cfg["img_size"], 300), (640, det["max_det"])):
        m = model.val(data=str(data), imgsz=imgsz, batch=cfg["batch"], max_det=max_det, device=device,
                      project=str(run_dir), name="val", exist_ok=True,
                      plots=False, verbose=False)
        evals[f"imgsz{imgsz}_maxdet{max_det}"] = box_metrics(m)
        print(f"imgsz {imgsz}, max_det {max_det}: {evals[f'imgsz{imgsz}_maxdet{max_det}']}")

    subset = json.loads((ROOT / cfg["yolo_dir"] / "subset.json").read_text())
    result = {
        "model": f"{det['model']} (COCO) fine-tuned on SKU-110K subset",
        "weights": str((run_dir / "weights" / "best.pt").relative_to(ROOT)),
        "settings": {k: cfg[k] for k in ("n_train", "n_val", "max_side", "img_size", "batch", "patience", "seed")}
                    | {"epochs": epochs, "max_det": det["max_det"]},
        "subset": {split: {k: v for k, v in s.items() if k != "files"} for split, s in subset.items()},
        "train_minutes": train_minutes,
        "device": torch.cuda.get_device_name(0) if device == 0 else "cpu",
        "sku110k_val": evals,
    }
    out = ROOT / "data" / "results" / f"yolo_{args.name}_eval.json"
    if args.eval_only and out.exists():  # keep the training time from the original run
        result["train_minutes"] = json.loads(out.read_text()).get("train_minutes")
    out.write_text(json.dumps(result, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
