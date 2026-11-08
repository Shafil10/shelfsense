"""Tests for src/rules/agreement.py on hand-made detector boxes (no model needed).

Run from the project root:  python -m pytest -v
"""
from src.rules.agreement import check_agreement, compare

CLASSES = ["nutricoco_cup", "nutricoco_box", "other_product", "empty_gap"]


def box(cls, x, shelf, w=40):
    return {"cls": cls, "shelf": shelf, "xyxy": [x, 100 * shelf, x + w, 100 * shelf + 80], "conf": 0.9}


def report(frames, flags=(), classes=CLASSES):
    return {"pipeline": {"classes": classes}, "integrity": {"flags": list(flags)},
            "keyframes": [{"time_sec": i, "evidence": f"evidence/{i}.jpg", "boxes": b} for i, b in enumerate(frames)]}


AGREE = {"id": 7, "cups": 4, "shelf": 2, "zone": "left"}
# shelf 2 is 0..600 px wide: left third is 0-200
GOOD = [box("nutricoco_cup", x, 2) for x in (0, 50, 100, 150)] + [box("other_product", x, 2) for x in (300, 400, 550)] + [box("other_product", 0, 1)]


def test_pending_without_brand_classes_or_agreement():
    assert check_agreement(report([GOOD], classes=["product"]), AGREE)["status"] == "pending"
    assert check_agreement(report([GOOD]), None)["reason"] == "no agreement set for this shop"


def test_compliant_uses_the_best_view():
    r = check_agreement(report([GOOD[:2], GOOD]), AGREE)                     # frame 0 shows only 2 cups
    assert (r["result"], r["cups_visible"], r["cups_in_place"], r["best_view"]["time_sec"]) == ("compliant", 4, 4, 1)
    assert r["shelf_share"] == round(4 / 7, 3) and r["findings"] == []


def test_missing_and_misplaced_cups():
    frame = ([box("nutricoco_cup", x, 2) for x in (0, 450)] + [box("nutricoco_cup", 0, 3)]          # one right, one on shelf 3
             + [box("other_product", x, 2) for x in (100, 250, 560)] + [box("empty_gap", 300, 2)])
    r = check_agreement(report([frame]), AGREE)
    assert r["result"] == "violation" and r["cups_visible"] == 3 and r["cups_in_place"] == 1 and r["gaps_on_shelf"] == 1
    assert r["findings"][:3] == ["3 of 4 agreed cups visible (1 missing)", "2 cup(s) not in the agreed place (shelf 2, left)",
                                 "1 empty gap(s) on the agreed shelf"]


def test_poor_coverage_makes_a_shortfall_unclear_not_a_violation():
    flags = [{"check": "coverage", "severity": "warning", "detail": "3 s blurred"}]
    r = check_agreement(report([GOOD[:2]], flags), AGREE)
    assert r["result"] == "unclear" and r["findings"][-1].startswith("the video did not cover")


def test_front_and_any_zones_accept_every_cup_on_the_shelf():
    frame = [box("nutricoco_cup", x, 2) for x in (0, 300, 500, 560)] + [box("other_product", 200, 2)]
    for zone in ("front", "any"):
        assert check_agreement(report([frame]), {**AGREE, "zone": zone})["result"] == "compliant"
    assert check_agreement(report([frame]), {**AGREE, "shelf": None, "zone": None})["cups_in_place"] == 4


def test_before_after():
    before = check_agreement(report([GOOD[:2] + GOOD[4:]]), AGREE)            # 2 cups
    after = check_agreement(report([GOOD]), AGREE)                             # 4 cups
    c = compare(before, after)
    assert c["cups_visible"] == {"before": 2, "after": 4, "change": 2} and c["trend"] == "better"
    assert c["result"] == {"before": "violation", "after": "compliant"}
    assert compare(after, before)["trend"] == "worse" and compare(after, after)["trend"] == "same"
    assert compare({"status": "pending"}, after)["status"] == "pending"
