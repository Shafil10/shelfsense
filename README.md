# ShelfSense: AI-Based Retail Merchandising Audit

CSE499A Senior Design I, North South University, Fall 2026
Group 1 (Section 13): Syed Ashfaq Shafin, Sadid Mehal Habib, Shafil Ahmed, Rutaba Adiba Rahman

ShelfSense takes a walkthrough video of store shelves and produces an audit report:
out-of-stock gaps, share of shelf, planogram compliance, and price tag / POSM checks.

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

## Data
Large files (videos, frames, datasets) are not stored in Git. Keep them in
`data/` locally and share through Google Drive. SKU-110K is downloaded directly
inside Colab/Kaggle for training.

## Sprints
| Sprint | Focus |
|---|---|
| 1 | Setup and problem definition |
| 2 | Data and keyframe extraction |
| 3 | Baseline VLM |
| 4 | Product detection |
| 5 | Integration and rule engine |
| 6 | Evaluation and demo |
