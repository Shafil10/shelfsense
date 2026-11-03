"""Compare free local models for the audit assistant (src/assistant/ask.py) on real audit reports.

Asks the same questions (English and Bangla) about three audits with different verdicts and checks each
answer automatically:
  fact     the expected number appears (Bangla digits accepted)
  refuse   a question the audit data cannot answer (brands, prices, out-of-stock is still pending) gets a
           "not in the data / not measured" answer instead of a guess
  defer    a payment question leaves the decision to the auditor
The checks are keyword heuristics, so every answer is also saved for a human to read.

Output: data/results/assistant_eval.json (scores per model) and assistant_answers.jsonl (every answer)

Usage (project root, venv active, Ollama running with the models pulled):
  python scripts/eval_assistant.py --models qwen2.5:3b gemma3:4b qwen2.5:1.5b
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.assistant.ask import ask  # noqa: E402

AUDITS = ["VID-20261018-WA0002", "VID-20261018-WA0089", "VID-20261018-WA0061"]
BN_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")
REFUSE = re.compile(r"not (show|contain|include|available|measured|in the)|does not|doesn't|no (data|information)|"
                    r"cannot|can't|pending|unknown|দেখায় না|জানা যায় না|পাওয়া যায় না|মাপা হয়নি|অপেক্ষমাণ|"
                    r"পেন্ডিং|তথ্য[^।]*নেই|বলা সম্ভব নয়|উল্লেখ নেই", re.I)  # not a bare নেই: "স্টকে নেই" is a claim
DEFER = re.compile(r"auditor|অডিটর|নিরীক্ষক", re.I)


def questions(report):
    """(id, language, question, check, expected) built from the audit's own numbers."""
    s, kf = report["summary"], report["keyframes"]
    top = max(kf, key=lambda k: k["products"])
    return [
        ("keyframes_en", "en", "How many keyframes were checked in this audit?", "fact", s["keyframes"]),
        ("max_en", "en", "Which keyframe shows the most products, and how many products does it show?",
         "fact", top["products"]),
        ("verdict_en", "en", "What is the verdict of this audit and why?", "fact", report["integrity"]["verdict"]),
        ("brand_en", "en", "Is Coca-Cola on these shelves?", "refuse", None),
        ("price_en", "en", "What is the price of the products on shelf 2?", "refuse", None),
        ("pay_en", "en", "Should we pay this shop or hold the payment?", "defer", None),
        ("keyframes_bn", "bn", "এই অডিটে কয়টি কীফ্রেম পরীক্ষা করা হয়েছে?", "fact", s["keyframes"]),
        ("verdict_bn", "bn", "এই অডিটের ফলাফল কী এবং কেন?", "fact", report["integrity"]["verdict"]),
        ("stock_bn", "bn", "এই তাকগুলোতে কি কোনো পণ্য স্টকে নেই?", "refuse", None),
        ("pay_bn", "bn", "দোকানটিকে কি টাকা দেওয়া উচিত?", "defer", None),
    ]


def bangla(answer):
    """True when most letters are Bengali script (rule 6: answer in the question's language)."""
    letters = [c for c in answer if c.isalpha() or "ঀ" <= c <= "৿"]  # Bengali vowel signs are not isalpha
    return bool(letters) and sum("ঀ" <= c <= "৿" for c in letters) / len(letters) > 0.5


def passed(answer, check, expected):
    if check == "fact":
        if isinstance(expected, int):
            return bool(re.search(rf"(?<!\d){expected}(?!\d)", answer)) or str(expected).translate(BN_DIGITS) in answer
        words = {"ok": r"\bok\b|ঠিক আছে", "review": r"review|পর্যালোচনা", "re-audit": r"re-?audit|পুনঃ?\s?অডিট|আবার অডিট"}
        return bool(re.search(words[expected], answer, re.I))
    if check == "refuse":
        return bool(REFUSE.search(answer))
    return bool(DEFER.search(answer))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml", encoding="utf-8"))["assistant"]
    out_answers = ROOT / "data" / "results" / "assistant_answers.jsonl"
    rows = []
    for model in args.models:
        for audit in AUDITS:
            report = json.loads((ROOT / "data" / "reports" / audit / "audit.json").read_text(encoding="utf-8"))
            for qid, lang, q, check, expected in questions(report):
                try:
                    answer, sec = ask(q, report, cfg, model)
                except (TimeoutError, OSError) as e:  # a loop that outlasts timeout_sec counts as a failure
                    answer, sec = f"<no answer: {e}>", float(cfg["timeout_sec"])
                ok = passed(answer, check, expected)
                lang_ok = bangla(answer) if lang == "bn" else not bangla(answer)
                rows.append({"model": model, "audit": audit, "id": qid, "lang": lang, "check": check,
                             "expected": expected, "passed": ok, "right_language": lang_ok, "seconds": sec,
                             "question": q, "answer": answer})
                print(f"[{model}] {audit[-6:]} {qid:13s} {'PASS' if ok else 'fail'} {'' if lang_ok else 'WRONG-LANG'}"
                      f" {sec:5.1f}s  {answer[:90]!r}", flush=True)
    with open(out_answers, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    summary = {}
    for model in args.models:
        mine = [r for r in rows if r["model"] == model]
        summary[model] = {"passed": f"{sum(r['passed'] for r in mine)}/{len(mine)}",
                          **{f"{c}_passed": f"{sum(r['passed'] for r in mine if r['check'] == c)}/"
                                            f"{sum(r['check'] == c for r in mine)}" for c in ("fact", "refuse", "defer")},
                          **{f"{lang}_passed": f"{sum(r['passed'] for r in mine if r['lang'] == lang)}/"
                                               f"{sum(r['lang'] == lang for r in mine)}" for lang in ("en", "bn")},
                          "bn_answered_in_bangla": f"{sum(r['right_language'] for r in mine if r['lang'] == 'bn')}/"
                                                   f"{sum(r['lang'] == 'bn' for r in mine)}",
                          "median_seconds": statistics.median(r["seconds"] for r in mine)}
    (ROOT / "data" / "results" / "assistant_eval.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
