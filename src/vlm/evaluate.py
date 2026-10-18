"""Sprint 3: score zero-shot VLM answers (from notebooks/02_baseline_vlm.ipynb) against ground truth.

Inputs:
  data/results/vlm_results_<version>.jsonl  downloaded from the Kaggle notebook output (v1, v2, ...)
  data/annotations/vlm_ground_truth.json    filled in by hand, one entry per video

Image-mode answers (one per keyframe) are combined per video: brand lists are merged,
the fridge brand is a majority vote, and yes/no answers count as yes if any frame says yes.

Outputs in data/results/:
  vlm_eval.csv       one row per (mode, video) with the scores
  vlm_failures.csv   every wrong answer with the raw model text, for the failure-case analysis

Every vlm_results_*.jsonl in data/results/ is scored and the versions are printed side by side.
Both CSVs have a `version` column so they can be filtered in Excel.

Usage:
  python -m src.vlm.evaluate
  python -m src.vlm.evaluate --results data/results/vlm_results_v2.jsonl
"""
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"

# Spelling variants -> one canonical brand name. Matched as whole words, longest first.
BRAND_ALIASES = {
    "coca cola": "coca-cola", "cocacola": "coca-cola", "coke": "coca-cola",
    "sprite": "sprite", "fanta": "fanta", "kinley": "kinley", "schweppes": "schweppes",
    "pepsi": "pepsi", "7 up": "7up", "7up": "7up", "seven up": "7up",
    "mountain dew": "mountain dew", "mtn dew": "mountain dew", "dew": "mountain dew",
    "mirinda": "mirinda", "aquafina": "aquafina", "sting": "sting",
    "mojo": "mojo", "clemon": "clemon", "speed": "speed", "frutika": "frutika",
    "pran": "pran", "frooto": "frooto", "red bull": "red bull", "redbull": "red bull",
    # supermarket drinks shelves (crop test, src/vlm/crop_brands.py)
    "a w": "a&w", "a and w": "a&w", "canada dry": "canada dry", "dr pepper": "dr pepper",
    "arizona": "arizona", "gatorade": "gatorade", "prime": "prime",
    "dunkin": "dunkin", "amwel": "amwel", "teazle": "teazle", "malee": "malee", "chabaa": "chabaa",
    "stute": "stute", "latina": "latina", "cyprina": "cyprina", "doi kham": "doi kham",
    "thai coco": "thai coco", "dewlands": "dewlands",
}

# Which company owns each brand: used for cooler purity (competitor products in a branded fridge)
BRAND_OWNER = {
    "coca-cola": "coca-cola company", "sprite": "coca-cola company", "fanta": "coca-cola company",
    "kinley": "coca-cola company", "schweppes": "coca-cola company",
    "pepsi": "pepsico", "7up": "pepsico", "mountain dew": "pepsico", "mirinda": "pepsico",
    "aquafina": "pepsico", "sting": "pepsico",
    "mojo": "akij", "clemon": "akij", "speed": "akij", "frutika": "akij",
    "pran": "pran", "frooto": "pran", "red bull": "red bull",
}

QUESTIONS = ["fridge_brand", "brands", "counts", "empty_space", "price_tags"]


def parse_answer(text):
    """Pull the JSON object out of a model answer. Returns a dict, or None if unparseable.

    Small VLMs often wrap JSON in ```json fences, add a sentence before it, or use
    Python-style True/False/None, so all of those are tolerated.
    """
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    chunk = text[start:end + 1]
    for candidate in (chunk, re.sub(r"\bTrue\b", "true", re.sub(r"\bFalse\b", "false",
                                    re.sub(r"\bNone\b", "null", chunk)))):
        try:
            value = json.loads(candidate)
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            continue
    return None


def normalize_brand(name):
    """Map a free-text brand name to its canonical form ('Coca Cola Zero' -> 'coca-cola')."""
    if not isinstance(name, str):
        return None
    clean = " " + re.sub(r"[^a-z0-9 ]", " ", name.lower().replace("-", " ")) + " "
    clean = re.sub(r"\s+", " ", clean)
    if clean.strip() in {"", "none", "null", "unknown", "n a", "no brand"}:
        return None
    for alias in sorted(BRAND_ALIASES, key=len, reverse=True):
        if f" {alias} " in clean:
            return BRAND_ALIASES[alias]
    return clean.strip()


def as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return {"true": True, "yes": True, "false": False, "no": False}.get(value.strip().lower())
    return None


def extract(prompt_id, parsed):
    """Turn one parsed answer into a plain Python value for that question (None = no usable answer)."""
    if parsed is None:
        return None
    if prompt_id == "fridge_brand":
        return normalize_brand(parsed.get("fridge_brand"))
    if prompt_id == "brands":
        items = parsed.get("brands")
        if not isinstance(items, list):
            return None
        names = [i.get("brand", i.get("name")) if isinstance(i, dict) else i for i in items]
        return {b for b in map(normalize_brand, names) if b}
    if prompt_id == "counts":
        counts = parsed.get("counts")
        if not isinstance(counts, dict):
            return None
        out = {}
        for brand, n in counts.items():
            b = normalize_brand(brand)
            if b and isinstance(n, (int, float)):
                out[b] = out.get(b, 0) + int(n)
        return out
    if prompt_id == "empty_space":
        return as_bool(parsed.get("empty_space"))
    if prompt_id == "price_tags":
        return as_bool(parsed.get("price_tags_visible"))
    return None


def combine_frames(prompt_id, values):
    """Combine per-keyframe answers into one per-video answer (image mode)."""
    values = [v for v in values if v is not None]
    if not values:
        return None
    if prompt_id == "fridge_brand":
        return Counter(values).most_common(1)[0][0]
    if prompt_id == "brands":
        return set().union(*values)
    if prompt_id == "counts":  # largest count seen for each brand across frames
        out = {}
        for counts in values:
            for b, n in counts.items():
                out[b] = max(out.get(b, 0), n)
        return out
    return any(values)  # empty_space, price_tags


def has_competitor(fridge_brand, brands):
    """True if any product belongs to a different company than the fridge. None if unknown."""
    owner = BRAND_OWNER.get(fridge_brand)
    if owner is None or brands is None:
        return None
    owners = [BRAND_OWNER.get(b) for b in brands]
    return any(o is not None and o != owner for o in owners)


def load_results(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def predictions(rows):
    """Return {(mode, video): {prompt_id: value}}, plus parse-failure counts and timings."""
    grouped = defaultdict(lambda: defaultdict(list))
    parse_fail = Counter()
    seconds = defaultdict(list)
    for r in rows:
        key = (r["mode"], r["video"])
        parsed = parse_answer(r["answer"])
        if parsed is None:
            parse_fail[key] += 1
        grouped[key][r["prompt_id"]].append((extract(r["prompt_id"], parsed), r))
        seconds[key].append(r.get("seconds", 0))
    preds = {}
    for key, by_prompt in grouped.items():
        preds[key] = {pid: combine_frames(pid, [v for v, _ in items]) for pid, items in by_prompt.items()}
        preds[key]["_raw"] = {pid: [r["answer"] for _, r in items] for pid, items in by_prompt.items()}
    return preds, parse_fail, seconds


def score(pred, truth):
    """Compare one video's prediction with its ground truth. Returns (scores, failures)."""
    s, failures = {}, []

    def check(name, got, expected):
        if expected is None:  # not labelled in ground truth: skip
            return
        ok = got == expected
        s[f"{name}_correct"] = int(ok)
        if not ok:
            failures.append((name, expected, got))

    gt_fridge = normalize_brand(truth.get("fridge_brand"))
    check("fridge_brand", pred.get("fridge_brand"), gt_fridge)

    gt_brands = truth.get("brands")
    if gt_brands is not None:
        gt_set = {normalize_brand(b) for b in gt_brands} - {None}
        got = pred.get("brands") or set()
        hits = got & gt_set
        s["brand_precision"] = round(len(hits) / len(got), 3) if got else 0.0
        s["brand_recall"] = round(len(hits) / len(gt_set), 3) if gt_set else 1.0
        s["hallucinated_brands"] = ";".join(sorted(got - gt_set))
        s["missed_brands"] = ";".join(sorted(gt_set - got))
        if got != gt_set:
            failures.append(("brands", sorted(gt_set), sorted(got)))

    gt_counts = truth.get("counts")
    if gt_counts:
        gt_counts = {normalize_brand(b): n for b, n in gt_counts.items()}
        got = pred.get("counts") or {}
        errors = [abs(got.get(b, 0) - n) for b, n in gt_counts.items()]
        s["count_mae"] = round(sum(errors) / len(errors), 2)
        s["count_total_true"] = sum(gt_counts.values())
        s["count_total_pred"] = sum(got.get(b, 0) for b in gt_counts)
        if any(errors):
            failures.append(("counts", gt_counts, got))

    check("empty_space", pred.get("empty_space"), truth.get("empty_space"))
    check("price_tags", pred.get("price_tags"), truth.get("price_tags_visible"))

    gt_comp = truth.get("competitor_present")
    if gt_comp is None and gt_brands is not None:
        gt_comp = has_competitor(gt_fridge, {normalize_brand(b) for b in gt_brands})
    check("competitor", has_competitor(pred.get("fridge_brand"), pred.get("brands")), gt_comp)
    return s, failures


def fmt(v):
    if isinstance(v, set):
        return sorted(v)
    return v


def version_label(path):
    """vlm_results_v2.jsonl -> 'v2'. The label is the part of the file name after 'vlm_results_'."""
    stem = Path(path).stem
    return stem[len("vlm_results"):].lstrip("_") or stem


def evaluate_file(results_path, truth, label):
    """Score one results file. Returns (eval_rows, fail_rows), each row tagged with the version label."""
    preds, parse_fail, seconds = predictions(load_results(results_path))
    eval_rows, fail_rows = [], []
    for (mode, video), pred in sorted(preds.items()):
        if video not in truth:
            print(f"warning: no ground truth for {video}, skipped")
            continue
        s, failures = score(pred, truth[video])
        n_answers = sum(len(v) for v in pred["_raw"].values())
        eval_rows.append({"version": label, "mode": mode, "video": video, "answers": n_answers,
                          "parse_failures": parse_fail[(mode, video)],
                          "mean_seconds": round(sum(seconds[(mode, video)]) / n_answers, 2), **s})
        for question, expected, got in failures:
            raw_key = {"competitor": "brands"}.get(question, question)
            fail_rows.append({"version": label, "mode": mode, "video": video, "question": question,
                              "expected": json.dumps(fmt(expected)), "got": json.dumps(fmt(got)),
                              "raw_answer_example": (pred["_raw"].get(raw_key) or [""])[0][:300]})
    return eval_rows, fail_rows


def main():
    parser = argparse.ArgumentParser(description="Score VLM answers against ground truth.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--results", nargs="+",
                        help="results file(s) (default: every vlm_results_*.jsonl in results_dir)")
    parser.add_argument("--truth", help="ground truth JSON (default from config)")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))["vlm"]
    results_dir = ROOT / cfg["results_dir"]
    results_paths = [Path(r) for r in args.results] if args.results else sorted(results_dir.glob("vlm_results*.jsonl"))
    if not results_paths:
        raise SystemExit(f"No vlm_results*.jsonl files found in {results_dir}")
    truth_path = Path(args.truth) if args.truth else ROOT / cfg["ground_truth"]
    truth = {k: v for k, v in json.loads(truth_path.read_text(encoding="utf-8")).items() if not k.startswith("_")}

    eval_rows, fail_rows, labels = [], [], []
    for path in results_paths:
        label = version_label(path)
        labels.append(label)
        e, f = evaluate_file(path, truth, label)
        eval_rows += e
        fail_rows += f
        print(f"{label}: {sum(r['answers'] for r in e)} answers from {path.name}")

    if not eval_rows:
        raise SystemExit("Nothing to score: check that video names match the ground truth file.")
    results_dir.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in eval_rows for k in r))  # all columns, first-seen order
    with open(results_dir / "vlm_eval.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(eval_rows)
    with open(results_dir / "vlm_failures.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["version", "mode", "video", "question", "expected", "got",
                                          "raw_answer_example"])
        w.writeheader()
        w.writerows(fail_rows)

    # Summary: mean of every numeric score over the videos, one column per (version, mode)
    columns = [(v, m) for v in labels for m in ("image", "video")]
    print(f"\n{'metric':24s}" + "".join(f"{v + ' ' + m:>12s}" for v, m in columns))
    metrics = [k for k in fields if k not in ("version", "mode", "video", "hallucinated_brands", "missed_brands")]
    for metric in metrics:
        line = f"{metric:24s}"
        for v, m in columns:
            vals = [r[metric] for r in eval_rows
                    if r["version"] == v and r["mode"] == m and isinstance(r.get(metric), (int, float))]
            line += f"{sum(vals) / len(vals):12.2f}" if vals else f"{'-':>12s}"
        print(line)
    print(f"\n{len(fail_rows)} failures -> {results_dir / 'vlm_failures.csv'}")
    print(f"per-video scores -> {results_dir / 'vlm_eval.csv'}")


if __name__ == "__main__":
    main()
