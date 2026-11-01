"""Sprint 4: export the detector to ONNX and time it on the CPU (no GPU, as on the team laptops).

Our frames are 576x1024 portrait, so the ONNX model is exported with that fixed input size
(1024 high x 576 wide, both multiples of 32) instead of a 1024x1024 square that would be 44% padding.
Each model is timed on the same frames: PyTorch on CPU and ONNX Runtime on CPU, after a few warm-up
runs. Box counts are compared too, to check that the export did not change the detections.

Output: data/results/onnx_cpu_timing.json (+ the .onnx file next to the weights)

Usage (project root, venv active; pip install onnx onnxruntime onnxslim):
  python -m src.detection.export_onnx --weights runs/detect/sku110k_v8n/weights/best.pt
"""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


def time_model(model, frames, kw, warmup=3):
    for p in frames[:warmup]:
        model.predict(str(p), **kw)
    ms, counts = [], []
    for p in frames:
        t = time.perf_counter()
        r = model.predict(str(p), **kw)[0]
        ms.append((time.perf_counter() - t) * 1000)
        counts.append(len(r.boxes))
    return {"median_ms": round(statistics.median(ms)), "mean_ms": round(statistics.mean(ms)),
            "fps": round(1000 / statistics.median(ms), 2)}, counts


def cpu_name():
    try:
        import subprocess
        out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return out or platform.processor()
    except Exception:
        return platform.processor()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--frames", default=str(ROOT / "data" / "annotation_batch" / "test"))
    ap.add_argument("--n", type=int, default=30, help="frames to time")
    args = ap.parse_args()

    full = yaml.safe_load(open(args.config, encoding="utf-8"))
    imgsz, det = full["sku110k"]["img_size"], full["detection"]
    frames = sorted(Path(args.frames).glob("*.jpg"))[: args.n]
    kw = dict(imgsz=imgsz, conf=det["conf_threshold"], max_det=det["max_det"], device="cpu", verbose=False)

    shape = (imgsz, imgsz * 9 // 16 // 32 * 32)    # (1024, 576): height, width of a portrait frame
    onnx_path = YOLO(args.weights).export(format="onnx", imgsz=shape, device="cpu", simplify=True)

    pt_stats, pt_counts = time_model(YOLO(args.weights), frames, kw)
    onnx_stats, onnx_counts = time_model(YOLO(onnx_path, task="detect"), frames, kw | {"imgsz": shape})
    diff = np.abs(np.array(pt_counts) - np.array(onnx_counts))

    result = {
        "weights": args.weights, "onnx": str(Path(onnx_path)),
        "input": f"{shape[0]}x{shape[1]} (h x w)", "frames": len(frames),
        "cpu": cpu_name(), "torch_threads": torch.get_num_threads(),
        "pytorch_cpu": pt_stats, "onnxruntime_cpu": onnx_stats,
        "onnx_size_mb": round(Path(onnx_path).stat().st_size / 1e6, 1),
        "box_count_median": {"pytorch": int(np.median(pt_counts)), "onnx": int(np.median(onnx_counts))},
        "box_count_abs_diff": {"mean": round(float(diff.mean()), 2), "max": int(diff.max())},
    }
    out = ROOT / "data" / "results" / "onnx_cpu_timing.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
