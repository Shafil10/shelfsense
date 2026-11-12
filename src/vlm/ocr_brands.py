"""Brand recognition on product crops with text reading (OCR) + a brand catalogue, compared with the VLM.

The detector finds products without knowing brands; a brand is then read from the text printed on each
crop. Nothing is trained on brands, so a new brand only needs a line in configs/brand_catalogue.txt.
Text that does not match a catalogue brand closely enough is dropped ("unknown"): in an audit a wrong
brand is worse than no answer (precision over recall).

Steps:
  read   EasyOCR (English + Bengali, and English + Thai: imported juices use Thai script) on every crop
         in data/crops/ -> data/results/crop_ocr.jsonl (cached; models in E:/tools/easyocr)
  score  OCR alone, VLM alone and both combined, every answer snapped to the catalogue, against
         data/annotations/crop_ground_truth.csv -> data/results/crop_ocr_eval.json

Usage (project root, venv active):
  python -m src.vlm.ocr_brands read
  python -m src.vlm.ocr_brands score
"""
import argparse
import csv
import difflib
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CROPS = ROOT / "data" / "crops"
CATALOGUE = ROOT / "configs" / "brand_catalogue.txt"
OCR_OUT = ROOT / "data" / "results" / "crop_ocr.jsonl"
VLM_ANSWERS = ROOT / "data" / "results" / "crop_vlm_answers.jsonl"
MODEL_DIR = Path("E:/tools/easyocr")          # never the default ~/.EasyOCR on C:
UPSCALE_TO = 320                               # crops are small; OCR reads small print better when enlarged


def load_catalogue(path=CATALOGUE):
    """{spelling: brand}; aliases map to their brand."""
    names = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip().lower()
        if not line:
            continue
        if "=" in line:
            alias, brand = (s.strip() for s in line.split("=", 1))
            names[alias] = brand
        else:
            names[line] = line
    return names


def norm(text):
    return re.sub(r"[^a-z0-9&+' ]+", " ", text.lower()).split()


def match_brand(text, catalogue, cutoff=0.8):
    """Best catalogue brand found in free text, or None.

    Every run of consecutive words in the text is compared with every catalogue spelling (with and
    without spaces). Short names (<= 4 letters: "mug", "jet", "a&w") must match a whole word exactly,
    since a fuzzy 3-letter match fires on almost any text.
    """
    words = norm(text or "")
    if not words:
        return None, 0.0
    best, best_score = None, 0.0
    for spelling, brand in catalogue.items():
        target = spelling.replace(" ", "")
        n = len(spelling.split())
        for size in {max(1, n - 1), n, n + 1}:
            for i in range(len(words) - size + 1):
                piece = "".join(words[i:i + size])
                if len(target) <= 4:
                    score = 1.0 if piece == target else 0.0
                else:
                    if abs(len(piece) - len(target)) > max(2, len(target) // 3):
                        continue
                    score = difflib.SequenceMatcher(None, piece, target).ratio()
                if score > best_score or (score == best_score and len(target) > len(best or "")):
                    best, best_score = brand, score
    return (best, best_score) if best_score >= cutoff else (None, best_score)


def read(args):
    import cv2
    import easyocr

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    readers = [easyocr.Reader(langs, gpu=args.gpu, model_storage_directory=str(MODEL_DIR / "model"),
                              user_network_directory=str(MODEL_DIR / "user_network"), verbose=False)
               for langs in (["en", "bn"], ["en", "th"])]
    done = set()
    if OCR_OUT.exists():
        done = {json.loads(line)["crop"] for line in OCR_OUT.read_text(encoding="utf-8").splitlines()}
    crops = sorted(p for p in CROPS.glob("c*.jpg") if p.name not in done)
    with open(OCR_OUT, "a", encoding="utf-8") as out:
        for p in crops:
            img = cv2.imread(str(p))
            h, w = img.shape[:2]
            scale = UPSCALE_TO / max(h, w)
            if scale > 1:
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            t = time.time()
            texts = []
            # products stand upright but brand names are often printed sideways on cans and bottles
            for rot in (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
                im = img if rot is None else cv2.rotate(img, rot)
                for reader in readers:
                    for _, text, conf in reader.readtext(im):
                        texts.append({"text": text, "conf": round(float(conf), 3), "rot": rot or 0})
            rec = {"crop": p.name, "texts": texts, "seconds": round(time.time() - t, 2)}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            print(p.name, rec["seconds"], "s", " | ".join(x["text"] for x in texts if x["conf"] >= args.min_conf)[:100])


def vlm_brand(answer):
    from src.vlm.evaluate import parse_answer
    parsed = parse_answer(answer)
    b = parsed.get("brand") if isinstance(parsed, dict) else None
    return b if isinstance(b, str) else None


def score(args):
    from src.vlm.evaluate import normalize_brand

    catalogue = load_catalogue()
    brands = set(catalogue.values())
    truth = {r["crop"]: normalize_brand(r["brand"]) for r in
             csv.DictReader(open(ROOT / "data" / "annotations" / "crop_ground_truth.csv", encoding="utf-8"))}
    ocr = {r["crop"]: r for r in map(json.loads, OCR_OUT.read_text(encoding="utf-8").splitlines())}
    vlm = {}
    if VLM_ANSWERS.exists():
        for a in map(json.loads, VLM_ANSWERS.read_text(encoding="utf-8").splitlines()):
            if a["mode"] == "crop":
                vlm[a["item"]] = vlm_brand(a["answer"])

    def same(a, b):
        return a is not None and b is not None and normalize_brand(a) == normalize_brand(b)

    preds = {"ocr": {}, "vlm": {}, "ocr_then_vlm": {}, "both_agree": {}}
    detail = []
    for crop in truth:
        text = " ".join(x["text"] for x in ocr.get(crop, {}).get("texts", []) if x["conf"] >= args.min_conf)
        o, o_score = match_brand(text, catalogue, args.cutoff)
        v, v_score = match_brand(vlm.get(crop) or "", catalogue, args.cutoff)
        preds["ocr"][crop] = o
        preds["vlm"][crop] = v
        preds["ocr_then_vlm"][crop] = o or v
        # strict: OCR must find the brand and the VLM must not name a different one
        preds["both_agree"][crop] = o if (o and (v is None or o == v)) else None
        detail.append({"crop": crop, "true": truth[crop] or "", "ocr_text": text[:120], "ocr": o or "",
                       "ocr_score": round(o_score, 2), "vlm_raw": vlm.get(crop) or "", "vlm": v or ""})

    def counts(pred):
        n = {"readable": 0, "correct": 0, "wrong_brand": 0, "abstained_readable": 0,
             "unreadable": 0, "abstained_unreadable": 0, "named_on_unreadable": 0}
        for crop, true in truth.items():
            got = pred[crop]
            if true:
                n["readable"] += 1
                n["correct" if same(got, true) else ("abstained_readable" if got is None else "wrong_brand")] += 1
            else:
                n["unreadable"] += 1
                n["abstained_unreadable" if got is None else "named_on_unreadable"] += 1
        named = n["correct"] + n["wrong_brand"]
        n["accuracy_on_readable"] = round(n["correct"] / max(n["readable"], 1), 3)
        n["precision_of_named_on_readable"] = round(n["correct"] / named, 3) if named else None
        return n

    missing = sorted({t for t in truth.values() if t} - {normalize_brand(b) for b in brands})
    result = {
        "catalogue_brands": len(brands), "test_brands_missing_from_catalogue": missing,
        "note": "catalogue = brands expected in Bangladeshi shops incl. the imported juices seen in the test "
                "videos; the old crop_vlm_eval catalogue held only the 24 test brands (upper bound)",
        "min_ocr_conf": args.min_conf, "cutoff": args.cutoff,
        "mean_ocr_seconds_per_crop": round(sum(r["seconds"] for r in ocr.values()) / max(len(ocr), 1), 2),
        "methods": {k: counts(p) for k, p in preds.items()},
    }
    out = ROOT / "data" / "results"
    (out / "crop_ocr_eval.json").write_text(json.dumps(result, indent=2))
    with open(out / "crop_ocr_detail.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(detail[0]))
        wr.writeheader()
        wr.writerows(detail)
    print(json.dumps(result, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["read", "score"])
    ap.add_argument("--gpu", action="store_true", help="run OCR on the GPU (default CPU: the GPU is often training)")
    ap.add_argument("--min-conf", type=float, default=0.2, help="ignore OCR text below this confidence")
    ap.add_argument("--cutoff", type=float, default=0.8, help="minimum similarity to a catalogue spelling")
    args = ap.parse_args()
    {"read": read, "score": score}[args.step](args)


if __name__ == "__main__":
    main()
