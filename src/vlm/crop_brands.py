"""Sprint 4 follow-up: does the VLM read brands better on single-product crops than on whole frames?

Sprint 3 found that Qwen2-VL-2B invents brands on whole-fridge images. The planned fix is the hybrid
pipeline: YOLO finds each product, then the VLM is asked about ONE product crop at a time. This script
tests that on our beverage shelf test videos, with the same model (fp16, zero-shot) as Sprint 3.

Steps (run in order):
  crops  detector boxes (conf >= 0.5) on the annotation test frames of the chosen videos -> up to
         --per-frame random crops per frame in data/crops/, crops.csv, and numbered contact sheets
         for labelling
  ask    VLM answers: one "which brand is this?" per crop, one "list the brands" per whole frame
         -> data/results/crop_vlm_answers.jsonl (resumable)
  score  compare with data/annotations/crop_ground_truth.csv (crop, brand; empty = unreadable) and
         data/annotations/frame_brands.csv (frame, brands separated by ';')
         -> data/results/crop_vlm_eval.json, crop_vlm_failures.csv

Usage (project root, venv active):
  python -m src.vlm.crop_brands crops --weights runs/detect/sku110k_v8n/weights/best.pt
  python -m src.vlm.crop_brands ask
  python -m src.vlm.crop_brands score
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.vlm.evaluate import normalize_brand, parse_answer

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"
CROPS = ROOT / "data" / "crops"
ANSWERS = ROOT / "data" / "results" / "crop_vlm_answers.jsonl"
VIDEOS = [f"VID-20261018-WA{n:04d}" for n in range(61, 67)]   # juice + soft drinks, all in the test split

CROP_PROMPT = (
    "This image shows one product from a supermarket shelf. Read the brand name printed on the product. "
    "If you cannot read a brand name, answer null; do not guess. "
    'Answer only with JSON: {"brand": "<brand>"} or {"brand": null}'
)
# Sprint 3 v2 "brands" prompt, reworded from fridge to open shelf; no "at most 8" (v2 treated it as a target)
FRAME_PROMPT = (
    "Look at the products on the shelves in this image. Read the labels on the bottles, cans and cartons. "
    "Include a brand only if you can read its name or see its logo on a product in this image. "
    "Do not guess brands that are commonly sold. List each brand once. "
    'Answer only with JSON: {"brands": ["<brand>", ...]}'
)


def make_crops(args):
    from ultralytics import YOLO

    import torch
    model = YOLO(args.weights)
    device = 0 if torch.cuda.is_available() else "cpu"
    frames = sorted(p for p in (ROOT / "data" / "annotation_batch" / "test").glob("*.jpg")
                    if p.name.split("__")[0] in VIDEOS)
    CROPS.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    rows = []
    for frame in frames:
        im = cv2.imread(str(frame))
        h, w = im.shape[:2]
        r = model.predict(str(frame), imgsz=1024, conf=0.5, max_det=1000, device=device, verbose=False)[0]
        boxes = [b for b in r.boxes.xyxy.tolist() if min(b[2] - b[0], b[3] - b[1]) >= args.min_side]
        confs = r.boxes.conf.tolist()
        picks = sorted(rng.sample(range(len(boxes)), min(args.per_frame, len(boxes))))
        for i in picks:
            x1, y1, x2, y2 = boxes[i]
            px, py = 0.1 * (x2 - x1), 0.1 * (y2 - y1)   # 10% margin so the label edge is not cut off
            x1, y1, x2, y2 = int(max(0, x1 - px)), int(max(0, y1 - py)), int(min(w, x2 + px)), int(min(h, y2 + py))
            name = f"c{len(rows):03d}.jpg"
            cv2.imwrite(str(CROPS / name), im[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows.append({"crop": name, "frame": frame.name, "video": frame.name.split("__")[0],
                         "x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": round(confs[i], 3)})
        print(f"{frame.name}: {len(boxes)} boxes >= {args.min_side}px, {len(picks)} cropped")
    with open(CROPS / "crops.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    # Numbered contact sheets so a person can label every crop
    tiles = []
    for r in rows:
        im = cv2.imread(str(CROPS / r["crop"]))
        s = min(150 / im.shape[1], 200 / im.shape[0])
        im = cv2.resize(im, (max(1, int(im.shape[1] * s)), max(1, int(im.shape[0] * s))))
        t = np.full((225, 150, 3), 255, np.uint8)
        t[:im.shape[0], :im.shape[1]] = im
        cv2.putText(t, r["crop"][:-4], (4, 218), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        tiles.append(t)
    for k in range(0, len(tiles), 40):
        part = tiles[k:k + 40]
        while len(part) % 10:
            part.append(np.full_like(tiles[0], 255))
        sheet = np.vstack([np.hstack(part[i:i + 10]) for i in range(0, len(part), 10)])
        cv2.imwrite(str(CROPS / f"sheet_{k // 40}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"{len(rows)} crops from {len(frames)} frames -> {CROPS}")


def load_vlm(cfg):
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    # fp16 like Sprint 3. The 2B model (4.4 GB) does not fit a 4 GB GPU, so accelerate keeps some
    # layers in CPU RAM and moves them in when needed: slower, but the same weights and precision.
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        cfg["model"], torch_dtype=torch.float16, device_map="auto",
        max_memory={0: "3GiB", "cpu": "12GiB"}).eval()
    processor = AutoProcessor.from_pretrained(cfg["model"], min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
    return model, processor


def ask_all(args):
    import torch
    from qwen_vl_utils import process_vision_info

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))["vlm"]
    model, processor = load_vlm(cfg)
    crops = list(csv.DictReader(open(CROPS / "crops.csv", encoding="utf-8")))
    done = set()
    if ANSWERS.exists():
        done = {(r["mode"], r["item"]) for r in map(json.loads, ANSWERS.read_text(encoding="utf-8").splitlines())}

    def ask(image, prompt):
        messages = [{"role": "user", "content": [{"type": "image", "image": f"file://{image}"},
                                                 {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = processor(text=[text], images=images, videos=videos, padding=True,
                           return_tensors="pt").to(model.device)
        start = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=cfg["max_new_tokens"], do_sample=False)
        answer = processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        return answer.strip(), round(time.perf_counter() - start, 2)

    jobs = [("crop", c["crop"], CROPS / c["crop"], CROP_PROMPT) for c in crops]
    jobs += [("frame", f, ROOT / "data" / "annotation_batch" / "test" / f, FRAME_PROMPT)
             for f in sorted({c["frame"] for c in crops})]
    for mode, item, path, prompt in jobs:
        if (mode, item) in done:
            continue
        answer, sec = ask(path.as_posix(), prompt)
        with open(ANSWERS, "a", encoding="utf-8") as f:
            f.write(json.dumps({"mode": mode, "item": item, "answer": answer, "seconds": sec,
                                "model": cfg["model"]}) + "\n")
        print(f"[{mode}] {item} ({sec}s): {answer[:80]}")


def brand_of(answer):
    parsed = parse_answer(answer)
    return None if parsed is None else normalize_brand(parsed.get("brand"))


def brands_of(answer):
    parsed = parse_answer(answer)
    items = parsed.get("brands") if parsed else None
    if not isinstance(items, list):
        return set()
    return {b for b in (normalize_brand(i if isinstance(i, str) else None) for i in items) if b}


def score(args):
    answers = [json.loads(line) for line in ANSWERS.read_text(encoding="utf-8").splitlines()]
    crop_ans = {a["item"]: a for a in answers if a["mode"] == "crop"}
    frame_ans = {a["item"]: a for a in answers if a["mode"] == "frame"}
    crops = {c["crop"]: c for c in csv.DictReader(open(CROPS / "crops.csv", encoding="utf-8"))}
    truth = {r["crop"]: normalize_brand(r["brand"]) for r in
             csv.DictReader(open(ROOT / "data" / "annotations" / "crop_ground_truth.csv", encoding="utf-8"))}
    frame_truth = {r["frame"]: {normalize_brand(b) for b in r["brands"].split(";") if normalize_brand(b)}
                   for r in csv.DictReader(open(ROOT / "data" / "annotations" / "frame_brands.csv", encoding="utf-8"))}

    # Crop level: each crop is either readable (has a true brand) or not
    n = {"crops": 0, "readable": 0, "correct": 0, "wrong_brand": 0, "abstained_readable": 0,
         "unreadable": 0, "abstained_unreadable": 0, "invented_on_unreadable": 0}
    failures = []
    for name, true in truth.items():
        got = brand_of(crop_ans[name]["answer"])
        n["crops"] += 1
        if true:
            n["readable"] += 1
            key = "correct" if got == true else ("abstained_readable" if got is None else "wrong_brand")
        else:
            n["unreadable"] += 1
            key = "abstained_unreadable" if got is None else "invented_on_unreadable"
        n[key] += 1
        if key not in ("correct", "abstained_unreadable"):
            failures.append({"level": "crop", "item": name, "frame": crops[name]["frame"], "expected": true or "",
                             "got": got or "", "raw_answer": crop_ans[name]["answer"][:200]})

    # Frame level: brand set from the whole frame vs the union of crop answers in that frame
    def prf(pred, gold):
        hit = len(pred & gold)
        return hit / len(pred) if pred else None, hit / len(gold) if gold else None

    per_frame = []
    for frame, gold in frame_truth.items():
        whole = brands_of(frame_ans[frame]["answer"])
        from_crops = {b for c, a in crop_ans.items() if crops[c]["frame"] == frame for b in [brand_of(a["answer"])] if b}
        pw, rw = prf(whole, gold)
        pc, rc = prf(from_crops, gold)
        per_frame.append({"frame": frame, "true": sorted(gold), "whole_frame": sorted(whole),
                          "from_crops": sorted(from_crops), "whole_precision": pw, "whole_recall": rw,
                          "crops_precision": pc, "crops_recall": rc})
        for b in sorted(whole - gold):
            failures.append({"level": "frame", "item": frame, "frame": frame, "expected": ";".join(sorted(gold)),
                             "got": b, "raw_answer": frame_ans[frame]["answer"][:200]})

    def mean(key):
        vals = [f[key] for f in per_frame if f[key] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    readable = max(n["readable"], 1)
    result = {
        "model": next(iter(crop_ans.values()))["model"], "videos": VIDEOS, "crop_counts": n,
        "crop_accuracy_on_readable": round(n["correct"] / readable, 3),
        "crop_wrong_brand_rate_on_readable": round(n["wrong_brand"] / readable, 3),
        "crop_invented_rate_on_unreadable": round(n["invented_on_unreadable"] / max(n["unreadable"], 1), 3),
        "frames": len(per_frame),
        "whole_frame_brand_precision": mean("whole_precision"), "whole_frame_brand_recall": mean("whole_recall"),
        "crops_brand_precision": mean("crops_precision"), "crops_brand_recall": mean("crops_recall"),
        "mean_seconds_crop": round(sum(a["seconds"] for a in crop_ans.values()) / len(crop_ans), 2),
        "mean_seconds_frame": round(sum(a["seconds"] for a in frame_ans.values()) / len(frame_ans), 2),
        "per_frame": per_frame,
    }
    results = ROOT / "data" / "results"
    (results / "crop_vlm_eval.json").write_text(json.dumps(result, indent=2))
    with open(results / "crop_vlm_failures.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["level", "item", "frame", "expected", "got", "raw_answer"])
        wr.writeheader()
        wr.writerows(failures)
    print(json.dumps({k: v for k, v in result.items() if k != "per_frame"}, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["crops", "ask", "score"])
    ap.add_argument("--weights", help="detector weights (crops step)")
    ap.add_argument("--per-frame", type=int, default=12, help="crops per frame (crops step)")
    ap.add_argument("--min-side", type=int, default=40, help="skip boxes smaller than this in pixels")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = ap.parse_args()
    {"crops": make_crops, "ask": ask_all, "score": score}[args.step](args)


if __name__ == "__main__":
    main()
