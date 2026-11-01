"""Sprint 4: who counts products better, the detector or the VLM? (Report 3 target 3)

Takes 20 random SKU-110K TEST images (never used for training or validation). The true product count
of an image is its number of annotated boxes. Then:
  tune    choose the confidence threshold that counts best on the 500 SKU-110K VAL images (not the
          test images), per detector -> data/results/count_conf_tuning.json
  detect  each detector (--weights, one or more) counts its boxes on every image, at the default
          threshold (detection.conf_threshold) and at the tuned one
  ask     Qwen2-VL-2B (fp16, zero-shot, same loading as the crop test) answers a counting prompt per image
          -> data/results/count_vlm_answers.jsonl (resumable)
  score   MAE, MAPE, share within 10% of the truth and share undercounted for every method
          -> data/results/count_compare.json, count_compare.csv (one row per image),
             data/results/figures/count_compare.png (true vs predicted count per image)

The images are copied, shrunk to sku110k.max_side like the training data, to datasets/sku110k_count20/
(SKU-110K is academic-only: never upload them).

Usage (project root, venv active):
  python -m src.detection.count_compare tune --weights runs/detect/sku110k_v8n/weights/best.pt
  python -m src.detection.count_compare detect --weights runs/detect/sku110k_v8n/weights/best.pt
  python -m src.detection.count_compare ask --device 0       # GPU, float16 (or --device cpu: bfloat16)
  python -m src.detection.count_compare score
"""
import argparse
import csv
import json
import random
import re
import statistics
import time
from pathlib import Path

import cv2
import pandas as pd
import yaml

from src.vlm.evaluate import parse_answer

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"
IMAGES = ROOT / "datasets" / "sku110k_count20"
DETECTIONS = ROOT / "data" / "results" / "count_detector.json"
TUNING = ROOT / "data" / "results" / "count_conf_tuning.json"
THRESHOLDS = [round(0.05 * i, 2) for i in range(2, 15)]     # 0.10 ... 0.70
ANSWERS = ROOT / "data" / "results" / "count_vlm_answers.jsonl"
COLS = ["image", "x1", "y1", "x2", "y2", "class", "image_width", "image_height"]

# Same wording as the counting prompt of Report 3 (Table XI)
PROMPT = ("Count every individual product facing (each visible pack, box or bottle) on all shelves. "
          'Answer only with JSON: {"product_count": <number>}')


def corrupt(path):
    """True if libjpeg reports corrupt data while decoding. Some SKU-110K files are damaged: OpenCV
    still returns an image, with the lost part filled in flat colour, while the ground truth keeps
    its boxes there. The warning goes to the C-level stderr (not catchable in-process on Windows),
    so decode in a short subprocess and read its stderr."""
    import subprocess
    import sys
    err = subprocess.run([sys.executable, "-c", "import cv2, sys; cv2.imread(sys.argv[1])", str(path)],
                         capture_output=True).stderr
    return b"Corrupt JPEG" in err


def pick_images(cfg, n, seed):
    """n random test images with their true counts; copies them (shrunk) on first use."""
    truth_file = IMAGES / "truth.csv"
    if truth_file.exists():
        return {r["image"]: int(r["true_count"]) for r in csv.DictReader(open(truth_file, encoding="utf-8"))}
    raw = ROOT / cfg["raw_dir"]
    df = pd.read_csv(raw / "annotations" / "annotations_test.csv", names=COLS)
    names = sorted(df["image"].unique())
    random.Random(seed).shuffle(names)
    IMAGES.mkdir(parents=True, exist_ok=True)
    truth, skipped = {}, []
    for name in names:
        if len(truth) == n:
            break
        im = cv2.imread(str(raw / "images" / name))
        g = df[df["image"] == name]
        if (im is None or (im.shape[1], im.shape[0]) != (int(g["image_width"].iloc[0]), int(g["image_height"].iloc[0]))
                or corrupt(raw / "images" / name)):
            skipped.append(name)
            continue   # unreadable, rotated by EXIF (as in prepare_sku110k) or corrupt
        scale = cfg["max_side"] / max(im.shape[:2])
        if scale < 1:
            im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(IMAGES / name), im, [cv2.IMWRITE_JPEG_QUALITY, 90])
        truth[name] = len(g)
    with open(truth_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "true_count"])
        w.writerows(truth.items())
    print(f"picked {len(truth)} test images, skipped {len(skipped)}: {skipped}")
    return truth


def confidences(model, paths, full, device):
    """Box confidences per image at a low threshold, so every higher threshold can be counted from them."""
    det, imgsz = full["detection"], full["sku110k"]["img_size"]
    return {p.name: model.predict(str(p), imgsz=imgsz, conf=min(THRESHOLDS), max_det=det["max_det"],
                                  device=device, verbose=False)[0].boxes.conf.tolist() for p in paths}


def run_name(weights):
    return Path(weights).parts[-3]     # e.g. runs/detect/sku110k_v8n/weights/best.pt -> sku110k_v8n


def tune(args, full, truth):
    from ultralytics import YOLO

    val = ROOT / full["sku110k"]["yolo_dir"]
    paths = sorted((val / "images" / "val").glob("*.jpg"))
    val_truth = {p.name: len((val / "labels" / "val" / p.with_suffix(".txt").name).read_text().splitlines())
                 for p in paths}
    tuning = json.loads(TUNING.read_text()) if TUNING.exists() else {}
    for weights in args.weights:
        conf = confidences(YOLO(weights), paths, full, args.device)
        mae = {t: round(statistics.mean(abs(sum(c >= t for c in conf[k]) - val_truth[k]) for k in val_truth), 1)
               for t in THRESHOLDS}
        best = min(mae, key=mae.get)
        tuning[run_name(weights)] = {"val_images": len(paths), "best_conf": best, "val_MAE": mae}
        print(f"{weights}: best conf {best} (val MAE {mae[best]}; at default "
              f"{full['detection']['conf_threshold']}: {mae[full['detection']['conf_threshold']]})")
    TUNING.write_text(json.dumps(tuning, indent=1))


def detect(args, full, truth):
    from ultralytics import YOLO

    default = full["detection"]["conf_threshold"]
    tuning = json.loads(TUNING.read_text()) if TUNING.exists() else {}
    results = json.loads(DETECTIONS.read_text()) if DETECTIONS.exists() else {}
    for weights in args.weights:
        name = run_name(weights)
        conf = confidences(YOLO(weights), [IMAGES / k for k in truth], full, args.device)
        for t in sorted({default, tuning.get(name, {}).get("best_conf", default)}):
            results[f"{name}@conf{t}"] = {k: sum(c >= t for c in v) for k, v in conf.items()}
            print(f"{weights} conf {t}: {sum(results[f'{name}@conf{t}'].values())} boxes on {len(truth)} images")
    DETECTIONS.write_text(json.dumps(results, indent=1))


def ask(args, full, truth):
    import torch
    from qwen_vl_utils import process_vision_info

    from src.vlm.crop_brands import load_vlm

    cfg = full["vlm"]
    if args.device == "cpu":
        # While the GPU trains: bfloat16 on the CPU (float32 needs ~9 GB RAM), fewer threads so training's
        # data loader keeps up. Same weights and prompt; only speed and number rounding differ.
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        torch.set_num_threads(args.threads)
        model = Qwen2VLForConditionalGeneration.from_pretrained(cfg["model"], torch_dtype=torch.bfloat16).eval()
        processor = AutoProcessor.from_pretrained(cfg["model"], min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
    else:
        model, processor = load_vlm(cfg)
    done = set()
    if ANSWERS.exists():
        done = {json.loads(line)["image"] for line in ANSWERS.read_text(encoding="utf-8").splitlines()}
    for name in truth:
        if name in done:
            continue
        messages = [{"role": "user", "content": [{"type": "image", "image": f"file://{(IMAGES / name).as_posix()}"},
                                                 {"type": "text", "text": PROMPT}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = processor(text=[text], images=images, videos=videos, padding=True,
                           return_tensors="pt").to(model.device)
        start = time.perf_counter()
        with torch.no_grad():
            # A count answer is ~12 tokens; a smaller limit than vlm.max_new_tokens only cuts off
            # repetition loops, which cost minutes per image on the 4 GB GPU
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        answer = processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
        sec = round(time.perf_counter() - start, 2)
        with open(ANSWERS, "a", encoding="utf-8") as f:
            f.write(json.dumps({"image": name, "answer": answer, "seconds": sec, "model": cfg["model"],
                                "device": "cpu-bf16" if args.device == "cpu" else "gpu-fp16",
                                "max_new_tokens": args.max_new_tokens}) + "\n")
        print(f"{name} ({sec}s): {answer[:80]}")


def vlm_count(answer):
    """(count, parsed): the JSON product_count, else the first number in the answer (marked not parsed)."""
    parsed = parse_answer(answer)
    if parsed is not None and isinstance(parsed.get("product_count"), (int, float)):
        return int(parsed["product_count"]), True
    m = re.search(r"\d+", answer)
    return (int(m.group()) if m else None), False


def metrics(pred, truth):
    pairs = [(pred[k], truth[k]) for k in truth if pred.get(k) is not None]
    err = [abs(p - t) for p, t in pairs]
    return {"images": len(pairs), "MAE": round(statistics.mean(err), 1),
            "MAPE_pct": round(100 * statistics.mean(e / t for e, (_, t) in zip(err, pairs)), 1),
            "within_10pct": round(sum(e <= 0.1 * t for e, (_, t) in zip(err, pairs)) / len(pairs), 2),
            "undercounted": round(sum(p < t for p, t in pairs) / len(pairs), 2)}


# Categorical slots 1-4 of the dataviz reference palette, in fixed order; marker shape is a second cue
SERIES_STYLE = [("#2a78d6", "o"), ("#eb6834", "^"), ("#1baf7a", "s"), ("#eda100", "D")]


def plot(methods, truth, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo, hi = 0, max([*truth.values()] + [v for m in methods.values() for v in m.values() if v is not None]) * 1.08
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=200)
    xs = [0, hi]
    ax.fill_between(xs, [0.9 * x for x in xs], [1.1 * x for x in xs], color="#e8e7e3", lw=0, label="within 10%")
    ax.plot(xs, xs, color="#8a8984", lw=1, ls="--", label="perfect count")
    for (name, counts), (color, marker) in zip(methods.items(), SERIES_STYLE):
        pts = [(truth[k], v) for k, v in counts.items() if v is not None]
        ax.scatter(*zip(*pts), s=36, color=color, marker=marker, edgecolor="white", linewidth=1.2, label=name, zorder=3)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="True product count (SKU-110K boxes)", ylabel="Predicted count")
    ax.grid(color="#ecebe7", lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4ae")
    ax.tick_params(colors="#52514e")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def score(args, full, truth):
    methods = json.loads(DETECTIONS.read_text()) if DETECTIONS.exists() else {}
    result = {"images": len(truth), "true_count": {"mean": round(statistics.mean(truth.values()), 1),
                                                   "min": min(truth.values()), "max": max(truth.values())},
              "methods": {}}
    for name, counts in methods.items():
        result["methods"][f"detector:{name}"] = metrics(counts, truth)
    if ANSWERS.exists():
        rows = [json.loads(line) for line in ANSWERS.read_text(encoding="utf-8").splitlines()]
        vlm = {r["image"]: vlm_count(r["answer"]) for r in rows}
        methods["vlm"] = {k: c for k, (c, _) in vlm.items()}
        result["methods"]["vlm:qwen2-vl-2b"] = metrics(methods["vlm"], truth) | {
            "parsed": sum(p for _, p in vlm.values()), "no_number": sum(c is None for c, _ in vlm.values()),
            "distinct_answers": sorted({c for c, _ in vlm.values() if c is not None}),
            "mean_seconds": round(statistics.mean(r["seconds"] for r in rows), 1)}
    out = ROOT / "data" / "results" / "count_compare.json"
    out.write_text(json.dumps(result, indent=2))
    with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "true_count"] + list(methods))
        for k, t in truth.items():
            w.writerow([k, t] + [methods[m].get(k) for m in methods])
    labels = {k: ("Qwen2-VL-2B" if k == "vlm" else "YOLOv8n " + k.replace("sku110k_v8n_4k", "4k").replace(
        "sku110k_v8n", "2k").replace("@conf", ", conf ")) for k in methods}
    plot({labels[k]: v for k, v in methods.items()}, truth, ROOT / "data" / "results" / "figures" / "count_compare.png")
    print(json.dumps(result, indent=2))
    print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["tune", "detect", "ask", "score"])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--weights", nargs="+", default=[], help="tune/detect: one or more detector weights")
    ap.add_argument("--device", default="cpu", help="cpu (default) or 0; ask on cpu runs the VLM in bfloat16")
    ap.add_argument("--threads", type=int, default=4, help="ask on cpu: torch threads")
    ap.add_argument("--max-new-tokens", type=int, default=64, help="ask: answer length limit")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    full = yaml.safe_load(open(args.config, encoding="utf-8"))
    truth = pick_images(full["sku110k"], args.n, args.seed)
    {"tune": tune, "detect": detect, "ask": ask, "score": score}[args.step](args, full, truth)


if __name__ == "__main__":
    main()
