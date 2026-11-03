"""Tests for src/assistant/ask.py and the answer checks of scripts/eval_assistant.py (no Ollama needed).

Run from the project root:  python -m pytest -v
"""
import io
import json
import sys
from pathlib import Path

from src.assistant import ask as asst

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import eval_assistant as ev  # noqa: E402

REPORT = {
    "video": {"file": "v.mp4", "sha256": "abc", "duration_sec": 10.0, "width": 576, "height": 1024,
              "modified_utc": "2026-10-18T10:00:00+00:00", "fps": 30.0},
    "integrity": {"reuse_matches": [], "longest_blur_gap_sec": 0.0, "flags": [], "verdict": "ok"},
    "summary": {"keyframes": 2, "products_per_keyframe": {"median": 15, "max": 20}, "shelves_median": 3},
    "kpis": {"share_of_shelf": {"status": "pending", "reason": "needs brands"}},
    "keyframes": [
        {"time_sec": 0.0, "products": 10, "products_per_shelf": [4, 6], "evidence": "evidence/a.jpg",
         "phash": "ff", "boxes": [{"xyxy": [1, 2, 3, 4], "conf": 0.9, "shelf": 1}] * 10},
        {"time_sec": 1.0, "products": 20, "products_per_shelf": [10, 10], "evidence": "evidence/b.jpg",
         "phash": "0f", "boxes": [{"xyxy": [1, 2, 3, 4], "conf": 0.9, "shelf": 1}] * 20},
    ],
}
CFG = {"url": "http://localhost:11434", "model": "qwen2.5:3b", "num_ctx": 8192, "timeout_sec": 5}


def test_compact_drops_boxes_and_hashes():
    c = asst.compact(REPORT)
    assert "boxes" not in json.dumps(c) and "phash" not in json.dumps(c) and "sha256" not in json.dumps(c)
    assert c["keyframes"][1] == {"time_sec": 1.0, "products": 20, "products_per_shelf_top_to_bottom": [10, 10],
                                 "evidence_image": "evidence/b.jpg"}


def test_ask_sends_rules_data_and_question(monkeypatch):
    sent = {}

    def fake_urlopen(req, timeout):
        sent.update(json.loads(req.data))
        return io.BytesIO(json.dumps({"message": {"content": " 2 keyframes (keyframe at 0.0 s). "}}).encode())

    monkeypatch.setattr(asst.urllib.request, "urlopen", fake_urlopen)
    answer, _ = asst.ask("How many keyframes?", REPORT, CFG)
    assert answer == "2 keyframes (keyframe at 0.0 s)."
    assert sent["model"] == "qwen2.5:3b" and sent["options"]["temperature"] == 0
    assert sent["messages"][0]["content"] == asst.SYSTEM
    assert '"keyframes": 2' in sent["messages"][1]["content"] and "How many keyframes?" in sent["messages"][1]["content"]


def test_eval_checks():
    assert ev.passed("The audit checked 7 keyframes.", "fact", 7)
    assert ev.passed("এই অডিটে ৭টি কীফ্রেম পরীক্ষা করা হয়েছে।", "fact", 7)
    assert not ev.passed("The audit checked 17 keyframes.", "fact", 7)
    assert ev.passed("The verdict is re-audit because of reused footage.", "fact", "re-audit")
    assert ev.passed("The audit data does not show brands; share of shelf is pending.", "refuse", None)
    assert ev.passed("এই তথ্য অডিটে নেই।", "refuse", None)
    assert not ev.passed("এই তাকগুলোতে কোনো পণ্য স্টকে নেই।", "refuse", None)
    assert not ev.passed("Yes, Coca-Cola is on shelf 2.", "refuse", None)
    assert ev.passed("The pipeline verdict is ok; the decision is the auditor's.", "defer", None)
    assert not ev.passed("Yes, pay the shop.", "defer", None)
    assert ev.bangla("এই অডিটে ৭টি কীফ্রেম (keyframe at 4.0 s)।") and not ev.bangla("This audit sampled 7 keyframes.")
