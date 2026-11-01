"""Make altered copies of a video to test the audit's re-use check (src/report/audit.py).

A dishonest rep would not always re-submit the exact file: the footage may be re-encoded (sent through
WhatsApp again), cropped or shrunk. Each variant is written frame by frame with OpenCV, which also
re-encodes it, to data/reuse_test/<stem>_<variant>.mp4.

Usage (project root, venv active):
  python scripts/make_reuse_variants.py Test_Videos/VID-20261018-WA0061.mp4
  python -m src.report.audit --video data/reuse_test/*.mp4 --weights runs/detect/sku110k_v8n/weights/best.pt
"""
import sys
from pathlib import Path

import cv2

OUT = Path(__file__).resolve().parents[1] / "data" / "reuse_test"
VARIANTS = {
    "reencoded": lambda f: f,                                          # only the new encoding
    "crop10": lambda f: f[f.shape[0] // 20: -f.shape[0] // 20, f.shape[1] // 20: -f.shape[1] // 20],
    "half": lambda f: cv2.resize(f, (f.shape[1] // 2, f.shape[0] // 2), interpolation=cv2.INTER_AREA),
}


def main():
    src = Path(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    for name, change in VARIANTS.items():
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS)
        writer = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = change(frame)
            if writer is None:
                writer = cv2.VideoWriter(str(OUT / f"{src.stem}_{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (frame.shape[1], frame.shape[0]))
            writer.write(frame)
        cap.release()
        writer.release()
        print(f"-> {OUT / f'{src.stem}_{name}.mp4'}")


if __name__ == "__main__":
    main()
