# ShelfSense: AI-Based Retail Merchandising Audit

Capstone project for **CSE499A Senior Design I**, North South University, Fall 2026 (Section 10, Group 1).

## Team (list in this order on reports)
| Name | Student ID | Email |
|---|---|---|
| Syed Ashfaq Shafin | 2312845042 | syed.shafin.2312845@northsouth.edu |
| Sadid Mehal Habib | 2312064042 | sadid.habib@northsouth.edu |
| Shafil Ahmed | 2231927042 | shafil.ahmed@northsouth.edu |
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
| 4 | Product detection | Annotated subset, fine-tuned YOLO nano, mAP | YOLOv8n trained on SKU-110K (mAP50 0.913), 222 frames pre-labelled; next: correct in Roboflow, fine-tune, mAP on our test frames |
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

**Supermarket footage (2026-10-18):** `Test_Videos/` (gitignored, share via Drive) holds 119 MP4s:
115 unique (4 "(1)" files are byte-identical copies), ~22 min total, median 11 s.
- 111 own recordings from one Bangladeshi supermarket, 576x1024 portrait WhatsApp, ~30 fps. They show
  **open shelves, not branded fridges**: personal care, biscuits/snacks, noodles, tea/coffee, juice,
  soft-drink cans (WA0063-66 only), cereal, milk, rice/lentils, oil... TK price tags and "OFFER" labels
  visible -> good for product detection (Sprint 4), share of shelf, price tags; not for cooler purity.
- 4 stock clips (`<id>_1920_1080_<fps>.mp4`): wide aisle shots, excluded (context only).
- `data/annotations/video_manifest.csv`: category per video (guessed from the middle frame,
  `verified=no`), in-scope category, and **split by blocks of consecutive videos** (neighbouring clips
  film the same shelf): test = WA0002-05, WA0054-60, WA0061-66, WA0089-94, WA0106-111 (29 videos),
  train = the other 82. Never train on test videos.
- Extraction at 4 fps (`data/results/dataset_stats.json`): train 3,771 sampled / 3,420 kept;
  test 1,298 / 1,071; 169 dropped blurry, 537 duplicates over all 115.
- First real motion blur (WA0089-92, hand-held milk cartons): blur effect ranks frames correctly
  (sharp 0.25-0.35, smeared 0.58-0.62) but there is no clean gap near 0.5 (some kept 0.49 frames are
  soft, some dropped 0.51 frames readable). Kept `blur_max: 0.5`; 0.45 would be stricter.
- `scripts/inventory_videos.py`: metadata + SHA-256 per video, flags exact duplicates.
- `scripts/select_annotation_frames.py`: sharpest frame per time-part of each video ->
  `data/annotation_batch/{train,test}/` (2 per video: 164 train, 58 test) for Roboflow/CVAT.

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

## Sprint 4 files
Detector = YOLOv8n, one class `product`, trained on a SKU-110K subset. Run locally on the E: PC (below)
or on Kaggle with the notebook. Settings in `configs/config.yaml` (`detection`, `sku110k`, `finetune`).
- `src/detection/prepare_sku110k.py`: random subset (2,000 train / 500 val, seed 0) -> YOLO format in
  `datasets/sku110k_yolo`, images shrunk to max 1280 px, skips unreadable / EXIF-mismatched images (0 skipped).
  SKU-110K (12.2 GB, `trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz`) unpacked in
  `datasets/SKU110K_fixed`. **License: academic, non-commercial, no redistribution** -> never upload its
  photos (Drive, GitHub, Roboflow).
- `src/detection/train_sku110k.py`: trains (or `--resume`, `--eval-only`), then validates 3 ways ->
  `data/results/yolo_sku110k_v8n_eval.json`. Weights `runs/detect/sku110k_v8n/weights/best.pt` (Drive only).
- `src/detection/prelabel.py`: pre-labels -> `data/annotation_batch/prelabels_roboflow.zip` (train/test,
  classes product + empty_gap), `prelabel_counts.csv`, `prelabel_examples.jpg`; `--all-frames` ->
  `detections_per_frame.csv` (boxes on every kept keyframe).
- `src/detection/finetune_ours.py`: after Roboflow correction: fine-tune on our train frames (val = 15% of
  train VIDEOS), test mAP (product, empty_gap) + counting MAE before/after on our test frames. Smoke-tested only.
- `notebooks/03_yolo_sku110k_prelabel.ipynb`: same pipeline for Kaggle (not run; local GPU was used instead).
- **Results (SKU-110K val, 500 images):** imgsz 1024 + max_det 1000: mAP50 0.913, mAP50-95 0.564, P 0.897,
  R 0.844. max_det 299: 0.889 / 0.549 (1 SKU image has 718 products, 11/500 have >300). imgsz 640:
  0.866 / 0.510. 30 epochs, 163.5 min on GTX 1650; mAP still rising at epoch 30.
  Ultralytics 8.4 silently raises max_det when it equals the default 300 in val, so compare with 299.
- **On our frames:** 4,558 keyframes, 151,279 boxes, median 29/frame, 27 frames with none
  (`detections_summary.json`). Annotation frames: median 30 boxes. Good on packed shelves at normal
  distance; fails on big close-up products (milk cartons median 6 boxes/frame vs canned food 53): SKU-110K
  objects are small. Fix: fine-tune on our frames; film shelves from ~1 m.
- `src/vlm/crop_brands.py` (steps `crops`, `ask`, `score`): Qwen2-VL-2B fp16 zero-shot on 120 random
  detector crops (conf >= 0.5, >= 40 px) from the 12 annotation test frames of WA0061-66 (drinks), vs the
  same model on the whole frames. Labels `data/annotations/crop_ground_truth.csv` (67 readable / 53 not;
  readable = enough of the printed name to identify it, logos alone do not count) and `frame_brands.csv`
  were **drafted by Claude, need a team check**. Results (`data/results/crop_vlm_eval.json`):
  crops 45/67 readable correct (0.67), with a brand catalogue (difflib cutoff 0.8, catalogue = the 24
  labelled brands, so an upper bound) 48/67 (0.72); frame-level brand precision/recall crops 0.94/0.63 vs
  whole frame 1.00/0.30 (catalogue), raw 0.64/0.60 vs 0.56/0.25. 6 of 12 whole-frame answers looped to the
  256-token limit (no usable list). Wrong real brand after catalogue: 2/120 crops, both "Coca-Cola".
  No answer mostly on Thai/Bengali script. Gatorade/Fanta named from logos on "unreadable" crops.
  21 s/crop on the 4 GB GPU (model partly offloaded to CPU), 176 s/whole frame.
- Figures for reports/demo: `data/results/figures/` (training curves, prelabel examples, good/bad
  detections, crop brand examples, blur check, video overview).
- Progress doc for the faculty meeting (2026-10-18): https://claude.ai/code/artifact/7921bd16-4f06-4b87-bc81-39976b203606
- Next: correct pre-labels in Roboflow (test frames checked by two people: they are ground truth and
  pre-labels bias annotators), then `finetune_ours.py`.

## The E: PC (second machine, differs from the laptop above)
i7-8565U, 20 GB RAM, **GTX 1650 Max-Q 4 GB (CUDA)**, C: nearly full -> everything goes on E:
(`.venv` Python 3.12 with torch 2.11+cu128, pip cache `E:\pip-cache`, HF models `E:\hf-cache`, datasets,
runs). Lessons:
- GPU memory: batch 2 at 1024 px (3 GB). Batch 4 hit 4.8 GB on dense images and spilled into shared
  memory (2.7x slower); batch 8 needs 7.5 GB. AMP is auto-disabled on GTX 16xx (fp32).
- Windows commit memory: each data-loader worker holds its own torch + CUDA (~1 GB) and validation uses
  twice the training workers; 2 workers grew the pagefile on C: to 13 GB. Use `workers: 1`.
- Scripts that use data-loader workers must have `if __name__ == "__main__":` and run from a file
  (piped stdin scripts hang on Windows).

## Weekly reports
- Written in LaTeX on Overleaf using the course template (IEEEtran, onecolumn journal), with `ref.bib`.
- Title: `[CSE499A(10) - Group\#1] Weekly Report: ShelfSense: AI-Based Retail Merchandising Audit`
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
