"""Tests for src/report/reuse.py on a synthetic shelf picture (no video needed).

Run from the project root:  python -m pytest -v
"""
import cv2
import numpy as np

from src.report.reuse import compare, prepare, same_picture

CFG = {"orb_min_inliers": 40, "orb_max_reproj_px": 2.0, "orb_min_ncc": 0.98}


def shelf(seed, h=1024, w=576):
    """Random 'products' (textured rectangles) on shelves: many corners for ORB."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 200, np.uint8)
    for row in range(5):
        y = 40 + row * 190
        cv2.rectangle(img, (0, y + 150), (w, y + 160), (90, 60, 30), -1)
        x = 10
        while x < w - 40:
            bw = int(rng.integers(30, 70))
            colour = tuple(int(c) for c in rng.integers(0, 255, 3))
            cv2.rectangle(img, (x, y), (x + bw, y + 145), colour, -1)
            cv2.putText(img, str(int(rng.integers(10, 99))), (x + 4, y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            x += bw + 6
    return img


def test_cropped_and_resized_copy_is_the_same_picture():
    a = shelf(1)
    crop = a[51:-51, 29:-29]                                                      # 10% crop
    half = cv2.resize(a, (a.shape[1] // 2, a.shape[0] // 2), interpolation=cv2.INTER_AREA)
    for copy in (crop, half):
        m = compare(prepare(copy), prepare(a))
        assert same_picture(m, CFG), m


def test_another_shelf_or_a_changed_shelf_is_not():
    a, b = shelf(1), shelf(2)
    assert not same_picture(compare(prepare(a), prepare(b)), CFG)
    changed = a.copy()                                                            # same shelf, other light and products moved
    changed = cv2.convertScaleAbs(changed, alpha=0.8, beta=30)
    changed[230:375] = np.roll(changed[230:375], 120, axis=1)
    changed[610:755] = np.roll(changed[610:755], -90, axis=1)
    m = compare(prepare(changed), prepare(a))
    assert not same_picture(m, CFG), m


def test_blank_pictures_do_not_crash():
    blank = np.full((400, 300, 3), 128, np.uint8)
    m = compare(prepare(blank), prepare(shelf(3)))
    assert m["inliers"] == 0 and not same_picture(m, CFG)
