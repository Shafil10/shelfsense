"""Tests for the dataset helpers of src/detection/train_nutricoco.py (no training).

Run from the project root:  python -m pytest -v
"""
from pathlib import Path

from src.detection.train_nutricoco import canonical, group_of, regroup


def test_class_names_are_normalised():
    assert canonical("NutriCoco Cup") == "nutricoco_cup" and canonical("nutricoco-box ") == "nutricoco_box"
    assert canonical("Empty gap") == "empty_gap"


def test_group_is_the_visit_part_of_the_file_name():
    assert group_of("shwapno-m10_2026-11-20__03_jpg.rf.9f8e7d") == "shwapno-m10_2026-11-20"
    assert group_of("shwapno-m10_2026-11-20__04") == "shwapno-m10_2026-11-20"
    assert group_of("IMG_1234") == "IMG_1234"                      # no group in the name: each photo alone


def test_regroup_keeps_each_visit_in_one_split():
    pairs = [(Path(f"v{g}__{k}.jpg"), None, "train") for g in range(20) for k in range(3)]
    out = regroup(pairs, seed=0)
    split_of = {}
    for img, _, split in out:
        split_of.setdefault(group_of(img.stem), set()).add(split)
    assert all(len(s) == 1 for s in split_of.values())             # no visit is split across train and test
    counts = {s: sum(1 for *_, x in out if x == s) for s in ("train", "valid", "test")}
    assert counts == {"train": 42, "valid": 9, "test": 9}
