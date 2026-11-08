# Labelling NutriCoco's photos (team guide)

Goal: boxes that teach the detector NutriCoco's products, so the agreement check (`src/rules/agreement.py`) can count
cups and check their position. Tool: Roboflow (free plan), one project "nutricoco-fridges", object detection.

## Classes (spell them exactly like this)
| Class | Box around | Notes |
|---|---|---|
| `NutriCoco cup` | every visible Coco Fresh cup (any flavour) | also partly hidden cups if you can tell it is ours |
| `NutriCoco box` | every Coco Fresh carton | |
| `other product` | every other product on the **same shelves** as NutriCoco cups | needed for shelf share; skip far-away shelves |
| `empty gap` | an empty space where a product would fit | only on shelves that hold NutriCoco products |

## Rules
- Tight boxes: touching the visible edges of the cup, not the shelf.
- When unsure whether a cup is NutriCoco's, **do not** label it `NutriCoco cup` (a false cup can hide a missing one).
- Label every NutriCoco item in the photo: a missing box teaches the model to ignore cups.
- Photos without any NutriCoco product stay in the set (with their `other product` boxes): they teach "no cups here".
- Do not crop or rotate photos in Roboflow; no augmentations at export (training adds its own).

## Test photos need two people
The test split is the ground truth for the numbers in the report. One person labels, a second checks every
box. Do not start from the model's pre-labels for test photos: pre-labels make annotators accept the model's
mistakes.

## Export and train
1. Roboflow -> Export -> format **YOLOv8**, download zip, unzip on the E: PC (e.g. `E:\ShelfSense\datasets\nutricoco_export`).
   Never upload NutriCoco's photos anywhere else (NDA).
2. Train (GPU PC, about 1-2 hours):
   ```
   python -m src.detection.train_nutricoco --export datasets\nutricoco_export --regroup
   ```
3. Read `data/results/nutricoco_eval.json`: `mAP50_95_per_class.nutricoco_cup` and `cup_count_mae`
   (average cups counted wrong per photo). Report these, not the training-set numbers.
4. Restart the server: it uses `runs/detect/nutricoco/weights/best.pt` automatically, and every new fridge
   video gets the agreement check (cups vs agreed cups, placement, empty gaps, before/after).
