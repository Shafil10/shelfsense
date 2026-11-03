"""Sprint 5: "Ask about this audit": a free local language model answers questions about one audit report.

The model (run by Ollama on this PC, no API key, no internet) only sees a compact copy of one
data/reports/<video>/audit.json: summary, integrity checks, KPIs and, per keyframe, its time, product
count per shelf and evidence image (not the box coordinates). The rules in SYSTEM make it:
  - answer only from that data and cite the keyframe time or check it used,
  - say so when the data does not contain the answer, instead of guessing,
  - never decide payments (Pay / Hold / Re-audit stays with the rules and the auditor),
  - answer in the language of the question (English or Bangla).

Ollama: E:\\tools\\ollama\\ollama.exe serve (models in OLLAMA_MODELS=E:\\ollama\\models).

Usage (project root, venv active, Ollama running):
  python -m src.assistant.ask --audit data/reports/VID-20261018-WA0089/audit.json "Why is this audit flagged?"
  python -m src.assistant.ask --audit ... --model gemma3:4b "এই অডিটে কয়টি তাক আছে?"
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"

SYSTEM = """You are the ShelfSense audit assistant. You help a human auditor understand ONE shelf audit.
Rules:
1. Use ONLY the audit data given below. Do not use outside knowledge about shops, brands or products.
2. When you state a fact, cite where it comes from: a keyframe time (for example "keyframe at 4.0 s")
   or a check name (for example "integrity check reused_footage").
3. If the data does not contain the answer, say exactly that the audit data does not show it.
   Never guess, never invent numbers, brands or prices.
4. You do not decide payments. If asked whether to pay, hold or re-audit, report the pipeline's
   verdict and its reasons, then end with: "The payment decision is the auditor's."
   (In Bangla: "টাকা দেওয়ার সিদ্ধান্ত অডিটরের।")
5. KPIs with status "pending" have not been measured yet; say so instead of estimating them.
6. Answer in the same language as the question (English or Bangla). Keep answers short: 1-5 sentences."""


def compact(report):
    """The parts of an audit.json the model needs, small enough for a 3B model's context."""
    return {
        "video": {k: report["video"].get(k) for k in ("file", "duration_sec", "width", "height", "modified_utc")},
        "integrity": report["integrity"],
        "summary": report["summary"],
        "kpis": report["kpis"],
        "keyframes": [{"time_sec": k["time_sec"], "products": k["products"],
                       "products_per_shelf_top_to_bottom": k["products_per_shelf"],
                       "evidence_image": k["evidence"]} for k in report["keyframes"]],
    }


def language_hint(question):
    """Small models answer Bangla questions in English unless told again next to the question."""
    if any("ঀ" <= c <= "৿" for c in question):
        return "\n\nউত্তর শুধু বাংলায় দিন। (Answer only in Bangla.)"
    return ""


def ask(question, report, cfg, model=None):
    """Returns (answer, seconds). Raises URLError if Ollama is not running."""
    body = {
        "model": model or cfg["model"],
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Audit data (JSON):\n" + json.dumps(compact(report), ensure_ascii=False)
                                        + "\n\nQuestion: " + question + language_hint(question)},
        ],
        "stream": False,
        "options": {"temperature": 0, "num_ctx": cfg["num_ctx"], "num_predict": cfg.get("max_tokens", 300)},
    }
    req = urllib.request.Request(cfg["url"].rstrip("/") + "/api/chat", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=cfg["timeout_sec"]) as r:
        answer = json.loads(r.read())["message"]["content"].strip()
    return answer, round(time.perf_counter() - start, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question")
    ap.add_argument("--audit", required=True, help="path to an audit.json from src.report.audit")
    ap.add_argument("--model", help="override assistant.model")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))["assistant"]
    report = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    answer, sec = ask(args.question, report, cfg, args.model)
    print(f"{answer}\n({args.model or cfg['model']}, {sec} s)")


if __name__ == "__main__":
    main()
