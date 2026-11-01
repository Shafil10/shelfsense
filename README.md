# ShelfSense: AI-Based Retail Merchandising Audit

CSE499A Senior Design I, North South University, Fall 2026
Group 1 (Section 13): Syed Ashfaq Shafin, Sadid Mehal Habib, Shafil Ahmed, Rutaba Adiba Rahman

ShelfSense takes a walkthrough video of store shelves and produces an audit report:
out-of-stock gaps, share of shelf, planogram compliance, and price tag / POSM checks.

**Live dashboard (clickable prototype, sample data):** https://shafil10.github.io/shelfsense/

## Pipeline
1. **Extraction** (`src/extraction`): sample frames, drop blurry and duplicate frames
2. **Detection** (`src/detection`): YOLO nano model finds products and empty gaps
3. **VLM** (`src/vlm`): small vision-language model identifies brands, reads tags
4. **Rules** (`src/rules`): compare results against planogram / KPIs
5. **Report** (`src/report`): generate the audit report

## Where things run
- Laptop (CPU): coding, frame extraction, rules, report, demo app
- Google Colab / Kaggle (free GPU): YOLO training and VLM inference (`notebooks/`)

## Setup (Windows)
```
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python scripts/check_env.py
python -m pytest
```

## Extract keyframes
Put videos in `data/raw_videos/`, then:
```
python -m src.extraction.extract_frames --video data/raw_videos
```
Kept frames, `frames_log.csv` and `stats.json` are written to `data/frames/<video_name>/`.
Use `--fps 4` for very short clips.

## Baseline VLM (Sprint 3)
Run `notebooks/02_baseline_vlm.ipynb` on Kaggle (steps are in the notebook), save its output in
`data/results/` (e.g. `vlm_results_v2.jsonl`), fill in `data/annotations/vlm_ground_truth.json`, then:
```
python -m src.vlm.evaluate
```

## Product detector (Sprint 4)
Trained on a GPU PC (GTX 1650, CUDA PyTorch); SKU-110K must be unpacked in `datasets/SKU110K_fixed`.
```
python -m src.detection.prepare_sku110k                     # 2,000 train / 500 val images, YOLO format
python -m src.detection.train_sku110k                       # YOLOv8n, mAP on the 500 val images
python -m src.detection.prepare_sku110k --n-train 4000 --out datasets/sku110k_yolo_4k
python -m src.detection.train_sku110k --yolo-dir datasets/sku110k_yolo_4k --epochs 12 --weights runs/detect/sku110k_v8n/weights/best.pt --name sku110k_v8n_4k
python -m src.detection.prelabel --weights runs/detect/sku110k_v8n/weights/best.pt      # boxes for Roboflow
python -m src.detection.export_onnx --weights runs/detect/sku110k_v8n/weights/best.pt   # CPU timing
python -m src.detection.count_compare tune --weights runs/detect/sku110k_v8n/weights/best.pt
python -m src.detection.count_compare detect --weights runs/detect/sku110k_v8n/weights/best.pt
python -m src.detection.count_compare ask --device 0        # VLM counting (GPU)
python -m src.detection.count_compare score                 # detector vs VLM counting
python -m src.vlm.crop_brands crops --weights runs/detect/sku110k_v8n/weights/best.pt   # brand reading on crops
```
Weights (`*.pt`) are shared through Google Drive, not Git.

## Dashboard
`app/mockup/index.html` is a one-file prototype (no build step): double-click `start_dashboard.bat`
(http://localhost:8000, works offline). After editing it, run `python scripts/build_mockup_site.py`
and copy `app/mockup/site/index.html` to the `gh-pages` branch to update the live site.

## Data
Large files (videos, frames, datasets) are not stored in Git. Keep them in
`data/` locally and share through Google Drive. SKU-110K is for academic, non-commercial
use only: never upload its photos (GitHub, Drive, Roboflow).

## Sprints
| Sprint | Focus |
|---|---|
| 1 | Setup and problem definition |
| 2 | Data and keyframe extraction |
| 3 | Baseline VLM |
| 4 | Product detection |
| 5 | Integration and rule engine |
| 6 | Evaluation and demo |
