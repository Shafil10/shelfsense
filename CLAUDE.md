# ShelfSense: AI-Based Retail Merchandising Audit

Capstone project for **CSE499A Senior Design I**, North South University, Fall 2026 (Section 10, Group 5).

## Team
| Name | Student ID | Email |
|---|---|---|
| Shafil Ahmed | 2231927042 | shafil.ahmed@northsouth.edu |
| Sadid Mehal Habib | 2312064042 | sadid.habib@northsouth.edu |
| Syed Ashfaq Shafin | 2312845042 | syed.shafin.2312845@northsouth.edu |
| Rutaba Adiba Rahman | 2231975042 | rutaba.rahman@northsouth.edu |

GitHub repo: https://github.com/Shafil10/shelfsense

## What the project does
Assigned topic: *video understanding with VLMs*. Application: a field merchandiser records a
walkthrough video of store shelves, and ShelfSense produces an audit report.

KPIs:
- **Out-of-stock**: empty shelf gaps where a product should be
- **Share of shelf**: % of facings for the client brand vs competitors
- **Planogram compliance**: products in the agreed position (eye level, order)
- **Price tag / POSM**: price labels and promo material present and visible

Product categories in scope: soaps & detergents, biscuits & snacks, beverages.

## Architecture (hybrid detector + VLM)
VLMs are weak at counting densely packed items, so detection and reasoning are split:
1. `src/extraction`: sample frames (1 fps), drop blurry and near-duplicate frames
2. `src/detection`: YOLO nano detects every product facing and empty gap (category-agnostic, no brand training)
3. `src/vlm`: small VLM identifies brands on crops, reads price tags (zero-shot, no training)
4. `src/rules`: compare results to planogram / KPIs
5. `src/report`: JSON audit report, later a Streamlit/Gradio demo in `app/`

## Hard constraints (important)
- Laptop: Ryzen 5 5600U, **8 GB RAM, no NVIDIA GPU, CPU-only PyTorch**, Windows 11, PowerShell, Python 3.14 in `.venv`.
- Cloud compute: **free Kaggle (preferred, ~30 GPU h/week) and Colab only**. All training and VLM inference goes in `notebooks/` to run there.
- Use small models only: YOLOv8n / YOLO11n; VLMs around 2B (Qwen2-VL-2B, SmolVLM, Moondream2).
- Small datasets: subset of SKU-110K (~1,000–2,000 images), a few hundred annotated frames.
- **Public datasets only for now** (no shop recording yet). SKU-110K has photos, not videos, so we
  simulate walkthrough videos by panning across photos (`simulate_video.py`).
- Process frames one at a time; never load a whole video into memory.
- Never commit data, videos, or model weights (`.gitignore` covers `data/`, `*.mp4`, `*.pt`, `runs/`, `datasets/`). Share data via Google Drive.
- If a library won't install on Python 3.14, recreate the venv with Python 3.12.
- Settings live in `configs/config.yaml`; scripts read defaults from there.

## Six-sprint plan (one weekly report each)
| Sprint | Focus | Deliverable | Status |
|---|---|---|---|
| 1 | Setup and problem definition | Repo, env, scope, KPIs, initial lit review | Done (verify GitHub push + collaborators added) |
| 2 | Data and keyframe extraction | Simulator, extractor, dataset stats | Extractor + tests done and run on 3 real fridge clips; simulator, SKU-110K notebook and longer real videos still to do |
| 3 | Baseline VLM | Zero-shot VLM results + failure cases (Kaggle notebook) | Not started |
| 4 | Product detection | Annotated subset, fine-tuned YOLO nano, mAP | Not started |
| 5 | Integration and rule engine | End-to-end pipeline → JSON report | Not started |
| 6 | Evaluation and demo | Metrics vs manual counts, demo app, design report | Not started |

## Sprint 2 files
Done:
- `src/extraction/extract_frames.py`: timestamp-based sampling, resize (never upscales), blur filter,
  pHash dedup; writes kept JPEGs, `frames_log.csv` and `stats.json` per video in `data/frames/<video>/`
- `tests/test_extract_frames.py`: pytest suite on synthetic shelf images and videos (9 tests)

Not built yet:
- `src/extraction/simulate_video.py`: pan across SKU-110K photos with shake and motion-blur bursts
- `notebooks/01_get_sku110k_sample.ipynb`: run on Kaggle to download SKU-110K and zip ~30 sample photos

Real data: 3 short phone clips of branded beverage fridges in `data/raw_videos/fridge_01..03.mp4`
(576x1024 portrait, 2.3 s / 6.4 s / 2.4 s, WhatsApp-compressed). Too short for dataset stats;
longer full-quality recordings (15-30 s per fridge) are needed.

Design decision from testing on the real fridge clips: Laplacian variance failed as a blur metric
because it depends on scene content (sharp close-ups scored 49, motion-blurred frames 147, no
single threshold works). Replaced by the **Crete et al. (2007) blur effect** score (0 sharp, 1 blurred):
real frames 0.24-0.39, artificially blurred copies 0.56-0.72, so `blur_max: 0.5`. Laplacian is
still logged in `frames_log.csv` for comparison. At 4 fps the 3 clips give 45 sampled, 30 kept,
0 blurry, 15 duplicates. The clips contain no real motion blur, so the blur filter is still
untested on real blurry footage.

Run commands (PowerShell, from project root, venv active):
```
python -m pytest -v
python -m src.extraction.extract_frames --video data/raw_videos --fps 4
```

## Weekly reports
- Written in LaTeX on Overleaf using the course template (IEEEtran, onecolumn journal), with `ref.bib`.
- Title: `[CSE499A(10) - Group\#5] Weekly Report: ShelfSense: AI-Based Retail Merchandising Audit`
- Each report section has: Follow-up from last week (status of last week's targets), Work
  narrative (tasks completed, challenges faced, progress made; figures/tables welcome),
  Target for next weekly update (numbered list, reused as next week's follow-up), Individual
  contribution table (Task List | Member Assigned | Comments; split tasks so not everyone is on every task).
- Report 1 is done. Report 2 should be written from the **actual** `stats.json` / `frames_log.csv`
  results, never invented numbers. Report 1's targets for next week were: finalise categories,
  get shop permission (now replaced by public datasets), get footage, build extraction script,
  report dataset stats, extend lit review with a comparison table.
- References already in `ref.bib`: goldman2019sku110k, liu2023llava, wang2024qwen2vl,
  fu2024videomme, redmon2016yolo, radford2021clip.

## Working style
- The team is learning: explain steps clearly, give exact PowerShell commands.
- Test scripts before declaring them done; report real outputs.
- Commit after each working milestone with a clear message (e.g. "Sprint 2: ...").
