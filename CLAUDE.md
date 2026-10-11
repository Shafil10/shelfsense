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

**Real problem (from the team, Bangladesh retail):** brand companies (Coca-Cola, Pepsi/Transcom,
Akij, ...) pay shops for merchandising: they supply branded fridges on condition that only their
products are stored in them, and pay extra for prime placement (front / eye level). Field audit
representatives are often bribed by shopkeepers and fabricate their reports. ShelfSense replaces the
rep's word with **verifiable photo/video evidence** and an automatic audit, so companies spend less on
audits and re-checks, and shopkeepers stay accountable. Goal: **cost reduction for the companies**.

Design consequences:
- Every finding must point to evidence (keyframe + timestamp, later a bounding box) a human can check.
- **Precision over recall for violations**: a false "competitor in your fridge" accuses an honest
  shopkeeper. VLM brand hallucination (Sprint 3) is therefore unacceptable on whole-fridge images.
- Key KPIs: **cooler purity** (only the owner's brands in a branded fridge) and **paid placement**
  (agreed brand on the agreed shelf / front), then out-of-stock and share of shelf.
- Evidence integrity is part of the problem: a dishonest rep could film another fridge, reuse an old
  video, or skip shelves. Cheap checks in scope: fridge identity (asset sticker / "property of" text),
  re-used footage via perceptual hashes across audits, full-coverage check, file metadata + SHA-256.
  A tamper-proof capture app (GPS, live capture) is future work.

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
| 3 | Baseline VLM | Zero-shot VLM results + failure cases (Kaggle notebook) | Notebook + evaluator written and tested; needs Kaggle run and verified ground truth |
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
python -m src.vlm.evaluate   # after the Kaggle run; scores every data/results/vlm_results_*.jsonl
```

## Sprint 3 files
- `notebooks/02_baseline_vlm.ipynb`: Kaggle (GPU T4, Internet on). Qwen2-VL-2B, fp16, zero-shot.
  5 JSON-answer prompts (fridge_brand, brands, counts, empty_space, price_tags) in **image mode**
  (each keyframe) and **video mode** (up to 8 keyframes as one video). Writes `vlm_results_<version>.jsonl`
  (one line per answer, resumable). Input: Kaggle dataset made from `data/results/shelfsense_frames.zip`.
- `src/vlm/evaluate.py`: scores every `data/results/vlm_results_*.jsonl` against
  `data/annotations/vlm_ground_truth.json`; writes `vlm_eval.csv` and `vlm_failures.csv`.
  Tolerates code fences / extra text / True-False in answers; normalises brand spellings;
  image-mode answers combined per video (brand union, majority fridge brand, any-yes).
  Cooler purity = any product brand owned by a different company than the fridge brand (`BRAND_OWNER`).
- `tests/test_vlm_evaluate.py`: 21 tests incl. an end-to-end run on fake answers.
- Prompt versions (`PROMPT_VERSION` in the notebook; output `vlm_results_<version>.jsonl`; the evaluator
  scores every version side by side). Model, image size, tokens and generation settings stay fixed
  so versions differ only in the prompts.
  - **v1 results** (165 answers, draft ground truth): image/video brand precision 0.27/0.30, recall
    1.00/0.50; fridge brand 0.67/0.33; empty space 0.33/0.67; price tags 0/0; 24 answers unparseable.
    Failure cases: invented brands typical of fridges (Pepsi, 7up, Dr Pepper in a Coca-Cola fridge);
    every count = 10 (or 1), i.e. no real counting (argument for YOLO); invented price sequences
    (1, 2, 3... / TK 10, TK 15...) and copying the `<price>` placeholder; repetition loops until the
    256-token limit; OCR slip "nojo" for Mojo; shop sign / owner text given as fridge brand.
  - **v2 prompts**: products behind the glass door only, each brand once (max 8), no guessing common
    brands, ignore signs/menus/"property of" text, explicit empty answers (`"prices": []`).
  - **v2 results** (image/video): parse failures 8/0 (v1: 22/2), mean 2.9 s/answer (v1: 3.6);
    fridge brand 0.67/0.67; empty space 0.67/0.67; price tags 0.33/0; brand precision 0.13/0.25
    (worse), recall 1.00/1.00. Numeric limits became targets ("at most 8 brands" -> 8 brands listed,
    "at most 5 prices" -> 5 invented prices). Per-frame brand precision fridge_01 0.55 -> 0.48.
    Counts still 10/10/10 or 0/0/0.
  - **Conclusion**: prompting fixes output format (parsing, loops, speed, vague answers) but not
    perception (invented brands, counting, price tags) -> supports YOLO detect + per-crop VLM.
    Stop prompt tuning. Next: crop test (single-bottle crops -> "what brand is this?"), finish ground
    truth (fridge_03 brands, counts), Report 3. With 3 videos, report failure patterns, not significance.
- Ground truth is a DRAFT: fridge_03 brands unknown (red cans may be Mojo, not Coca-Cola); must be
  verified by a team member before Report 3 numbers are used.
- Zip frames with `tar -a -c -f data\results\shelfsense_frames.zip -C data\frames fridge_01 ...`,
  NOT `Compress-Archive` (PowerShell 5.1 writes backslash paths that break on Kaggle's Linux).

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
