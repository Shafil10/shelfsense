"""Tests for src/detection/count_compare.py (answer parsing, metrics, corrupt-image check).

Run from the project root:  python -m pytest -v
"""
import cv2
import numpy as np
import pytest

from src.detection import count_compare as cc


@pytest.mark.parametrize("answer, expected", [
    ('{"product_count": 120}', (120, True)),
    ('```json\n{\n  "product_count": 100\n}\n```', (100, True)),
    ('{"product_count": 87.0}', (87, True)),
    ("There are about 45 products on the shelves.", (45, False)),
    ('{"product_count": "many"}', (None, False)),
    ("I cannot count them.", (None, False)),
])
def test_vlm_count(answer, expected):
    assert cc.vlm_count(answer) == expected


def test_metrics():
    truth = {"a": 100, "b": 200, "c": 50}
    pred = {"a": 95, "b": 240, "c": None}      # c has no answer: left out, not counted as 0
    m = cc.metrics(pred, truth)
    assert m["images"] == 2
    assert m["MAE"] == 22.5                   # (5 + 40) / 2
    assert m["MAPE_pct"] == 12.5              # (5% + 20%) / 2
    assert m["within_10pct"] == 0.5
    assert m["undercounted"] == 0.5


def test_corrupt_detects_damaged_jpeg(tmp_path):
    img = (np.random.default_rng(0).random((200, 300, 3)) * 255).astype(np.uint8)
    good = tmp_path / "good.jpg"
    cv2.imwrite(str(good), img)
    data = bytearray(good.read_bytes())
    start = data.find(b"\xff\xda") + 20         # inside the compressed image data (after start-of-scan)
    data[start:start + 400] = bytes(400)       # wipe part of it, keep the end-of-image marker
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(bytes(data))
    assert not cc.corrupt(good)
    assert cc.corrupt(bad)
