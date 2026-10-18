"""Sprint 4: train YOLOv8n on the SKU-110K subset and measure mAP on its validation images.

Starts from the COCO-pretrained nano model (configs/config.yaml: detection.model). Training curves
and weights go to runs/detect/sku110k_v8n/. After training the best weights are validated a few
ways so the report can show what the settings matter:
  - img size 1024 (training size) vs 640 (Ultralytics default)
  - max_det 1000 vs 299, just under the Ultralytics default (packed shelves have more products than 300)

Output: data/results/yolo_<name>_eval.json

Usage (project root, venv active; needs src.detection.prepare_sku110k first):
  python -m src.detection.train_sku110k
  python -m src.detection.train_sku110k --epochs 1 --name smoke     # quick check that it runs
  python -m src.detection.train_sku110k --eval-only                 # re-validate existing weights
  python -m src.detection.train_sku110k --resume                    # continue an interrupted run
"""
import argparse
import json
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
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run from last.pt")
    args = ap.parse_args()

    full = yaml.safe_load(open(args.config, encoding="utf-8"))
    cfg, det = full["sku110k"], full["detection"]
    data = ROOT / cfg["yolo_dir"] / "data.yaml"
    run_dir = ROOT / "runs" / "detect" / args.name
    device = 0 if torch.cuda.is_available() else "cpu"
    epochs = args.epochs or cfg["epochs"]

    train_minutes = None
    if args.resume:
        # Continue an interrupted run from its last epoch. Ultralytics restores the saved training
        # settings, so the current worker count is written into the checkpoint first (fewer workers
        # = less Windows commit memory; each worker holds its own copy of torch + CUDA).
        last = run_dir / "weights" / "last.pt"
        ckpt = torch.load(last, weights_only=False)
        ckpt["train_args"]["workers"] = cfg["workers"]
        torch.save(ckpt, last)
        YOLO(last).train(resume=True)
    elif not args.eval_only:
        YOLO(det["model"]).train(
            data=str(data), epochs=epochs, imgsz=cfg["img_size"], batch=cfg["batch"],
            max_det=det["max_det"], patience=cfg["patience"], seed=cfg["seed"], device=device,
            workers=cfg["workers"], cache="disk", project=str(run_dir.parent), name=args.name,
            exist_ok=True)

    model = YOLO(run_dir / "weights" / "best.pt")
    evals = {}
    # 299, not 300: Ultralytics 8.4 silently raises max_det when it equals the default 300 and the
    # dataset has more objects per image, which would make this comparison meaningless
    for imgsz, max_det in ((cfg["img_size"], det["max_det"]), (cfg["img_size"], 299), (640, det["max_det"])):
        m = model.val(data=str(data), imgsz=imgsz, batch=cfg["batch"], max_det=max_det, device=device,
                      workers=cfg["workers"],
                      project=str(run_dir), name="val", exist_ok=True,
                      plots=False, verbose=False)
        evals[f"imgsz{imgsz}_maxdet{max_det}"] = box_metrics(m)
        print(f"imgsz {imgsz}, max_det {max_det}: {evals[f'imgsz{imgsz}_maxdet{max_det}']}")

    if (run_dir / "results.csv").exists():
        # Ultralytics logs cumulative seconds per epoch; the clock restarts after a resume, so add up the pieces
        times = [float(line.split(",")[1]) for line in (run_dir / "results.csv").read_text().strip().splitlines()[1:]]
        train_minutes = round((sum(t for t, nxt in zip(times, times[1:]) if nxt < t) + times[-1]) / 60, 1)

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
    out.write_text(json.dumps(result, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
