import cv2
import numpy as np
import pandas as pd

from src.detection.prepare_sku110k import COLS, convert, yolo_lines


def test_yolo_lines_normalises_clips_and_drops_empty_boxes():
    g = pd.DataFrame([["a.jpg", 0, 0, 50, 100, "object", 200, 100],      # left quarter, full height
                      ["a.jpg", 190, 90, 260, 140, "object", 200, 100],  # runs past the corner -> clipped
                      ["a.jpg", 300, 10, 320, 20, "object", 200, 100]],  # fully outside -> dropped
                     columns=COLS)
    assert yolo_lines(g, 200, 100) == ["0 0.12500 0.50000 0.25000 1.00000",
                                       "0 0.97500 0.95000 0.05000 0.10000"]


def test_convert_skips_bad_images_and_shrinks(tmp_path):
    raw = tmp_path / "raw"
    (raw / "images").mkdir(parents=True)
    (raw / "annotations").mkdir()
    rows = []
    for i in range(4):
        cv2.imwrite(str(raw / "images" / f"t{i}.jpg"), np.full((1500, 2000, 3), 128, np.uint8))
        rows.append(f"t{i}.jpg,100,100,500,700,object,2000,1500")
    (raw / "images" / "t0.jpg").write_bytes(b"not a jpeg")             # unreadable
    rows[1] = rows[1].replace("2000,1500", "1500,2000")                 # size disagrees with the image
    (raw / "annotations" / "annotations_train.csv").write_text("\n".join(rows) + "\n")

    out = tmp_path / "yolo"
    s = convert(raw, "annotations_train.csv", out, "train", n=10, max_side=1000, seed=0)
    assert (s["images"], s["skipped"], s["boxes"]) == (2, 2, 2)
    assert sorted(s["files"]) == ["t2.jpg", "t3.jpg"]
    im = cv2.imread(str(out / "images" / "train" / "t2.jpg"))
    assert im.shape[:2] == (750, 1000)                                  # longest side shrunk to max_side
    assert (out / "labels" / "train" / "t2.txt").read_text() == "0 0.15000 0.26667 0.20000 0.40000\n"
