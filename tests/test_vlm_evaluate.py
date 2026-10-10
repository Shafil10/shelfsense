"""Tests for src/vlm/evaluate.py using hand-written fake model answers.

Run from the project root:  python -m pytest -v
"""
import csv
import json
import sys

import pytest

from src.vlm import evaluate as ev


@pytest.mark.parametrize("text, expected", [
    ('{"fridge_brand": "Pepsi"}', {"fridge_brand": "Pepsi"}),
    ('```json\n{"brands": ["Sprite", "Fanta"]}\n```', {"brands": ["Sprite", "Fanta"]}),
    ('Sure! Here it is: {"empty_space": True, "where": "top"}', {"empty_space": True, "where": "top"}),
    ('{"fridge_brand": None}', {"fridge_brand": None}),
    ("I cannot tell from this image.", None),
    ('["Sprite"]', None),
    ("", None),
])
def test_parse_answer(text, expected):
    assert ev.parse_answer(text) == expected


@pytest.mark.parametrize("name, expected", [
    ("Coca-Cola", "coca-cola"),
    ("Coca Cola Zero", "coca-cola"),
    ("COKE", "coca-cola"),
    ("Mtn Dew", "mountain dew"),
    ("7-Up", "7up"),
    ("Aquafina Water", "aquafina"),
    ("Some Local Juice", "some local juice"),
    ("none", None),
    (None, None),
])
def test_normalize_brand(name, expected):
    assert ev.normalize_brand(name) == expected


def test_extract_handles_odd_shapes():
    assert ev.extract("brands", {"brands": [{"brand": "Sprite"}, "Fanta", "unknown"]}) == {"sprite", "fanta"}
    assert ev.extract("brands", {"brands": "Sprite"}) is None
    assert ev.extract("counts", {"counts": {"Coke": 5, "Coca-Cola": 3, "Sprite": "many"}}) == {"coca-cola": 8}
    assert ev.extract("empty_space", {"empty_space": "yes"}) is True
    assert ev.extract("price_tags", {"price_tags_visible": "maybe"}) is None


def test_combine_frames():
    assert ev.combine_frames("fridge_brand", ["pepsi", "coca-cola", "coca-cola", None]) == "coca-cola"
    assert ev.combine_frames("brands", [{"sprite"}, {"fanta", "sprite"}, None]) == {"sprite", "fanta"}
    assert ev.combine_frames("counts", [{"sprite": 4}, {"sprite": 6, "fanta": 2}]) == {"sprite": 6, "fanta": 2}
    assert ev.combine_frames("empty_space", [False, True, None]) is True
    assert ev.combine_frames("empty_space", [None, None]) is None


def test_has_competitor():
    assert ev.has_competitor("coca-cola", {"sprite", "fanta"}) is False
    assert ev.has_competitor("mojo", {"mojo", "coca-cola"}) is True
    assert ev.has_competitor("mountain dew", {"aquafina"}) is False  # both PepsiCo
    assert ev.has_competitor("unknown brand", {"sprite"}) is None
    assert ev.has_competitor("coca-cola", {"local juice"}) is False  # unknown owner ignored


def test_score_brands_and_failures():
    truth = {"fridge_brand": "Coca-Cola", "brands": ["Coca-Cola", "Sprite", "Fanta"],
             "empty_space": False, "price_tags_visible": None}
    pred = {"fridge_brand": "coca-cola", "brands": {"sprite", "pepsi"}, "empty_space": True}
    s, failures = ev.score(pred, truth)
    assert s["fridge_brand_correct"] == 1
    assert s["brand_precision"] == 0.5 and s["brand_recall"] == 0.333
    assert s["hallucinated_brands"] == "pepsi" and s["missed_brands"] == "coca-cola;fanta"
    assert s["empty_space_correct"] == 0
    assert "price_tags_correct" not in s  # not labelled -> skipped
    assert s["competitor_correct"] == 0  # pepsi in a coke fridge was predicted, truth has none
    assert {f[0] for f in failures} == {"brands", "empty_space", "competitor"}


def test_main_end_to_end(tmp_path, monkeypatch):
    def row(mode, frame, pid, answer):
        return {"mode": mode, "video": "fridge_x", "frame": frame, "prompt_id": pid,
                "answer": answer, "seconds": 2.0}

    rows = [
        row("image", "a.jpg", "fridge_brand", '{"fridge_brand": "Coca-Cola"}'),
        row("image", "b.jpg", "fridge_brand", '{"fridge_brand": "Coke"}'),
        row("image", "a.jpg", "brands", '{"brands": ["Sprite"]}'),
        row("image", "b.jpg", "brands", '```json\n{"brands": ["Fanta", "Coca-Cola"]}\n```'),
        row("image", "a.jpg", "empty_space", '{"empty_space": false, "where": ""}'),
        row("image", "a.jpg", "price_tags", "no idea"),
        row("video", None, "fridge_brand", '{"fridge_brand": "Pepsi"}'),
        row("video", None, "brands", '{"brands": ["Sprite", "Pepsi"]}'),
    ]
    results = tmp_path / "vlm_results.jsonl"
    results.write_text("\n".join(json.dumps(r) for r in rows))
    truth = tmp_path / "gt.json"
    truth.write_text(json.dumps({
        "_note": "keys starting with _ are ignored",
        "fridge_x": {"fridge_brand": "Coca-Cola", "brands": ["Coca-Cola", "Sprite", "Fanta"],
                     "empty_space": False, "price_tags_visible": False},
    }))
    config = tmp_path / "config.yaml"
    config.write_text(f"vlm:\n  results_dir: {tmp_path.as_posix()}\n  ground_truth: {truth.as_posix()}\n")

    monkeypatch.setattr(sys, "argv", ["evaluate", "--config", str(config), "--results", str(results)])
    ev.main()

    with open(tmp_path / "vlm_eval.csv", newline="") as f:
        scores = {r["mode"]: r for r in csv.DictReader(f)}
    assert scores["image"]["fridge_brand_correct"] == "1"
    assert scores["image"]["brand_recall"] == "1.0"
    assert scores["image"]["parse_failures"] == "1"
    assert scores["video"]["fridge_brand_correct"] == "0"
    with open(tmp_path / "vlm_failures.csv", newline="") as f:
        failed = {(r["mode"], r["question"]) for r in csv.DictReader(f)}
    assert ("image", "price_tags") in failed  # unparseable answer counts as wrong
    assert ("video", "fridge_brand") in failed and ("video", "brands") in failed
