# Solar Filament Segmentation (MAGFiLO / Kaggle Solar Filament Segmentation Challenge 2026)

SEA710 Group 7 — instance segmentation of solar filaments in GONG H-alpha full-disk images,
evaluated with Panoptic Quality (PQ).

## Pipeline
1. **Preprocessing** (`src/preprocess.py`, `src/build_cache.py`): Hough-based disk localisation re-ranked by limb
   contrast, radial limb-darkening flattening, uint8 cache. Agreement-weighted soft labels (pixel = fraction of
   annotators who marked it) from the 1–3 independent annotation sets per image.
2. **Splits** (`src/splits.py`): 5 folds grouped by year-month (filaments persist for days–weeks), stratified by GONG station.
3. **Models** (`src/model.py`, `src/train.py`): U-Net (scSE decoder) trained on 1024² full-resolution crops,
   BCE + soft Dice, AMP, multi-GPU. Same architecture trained from scratch vs ImageNet-pretrained.
4. **Inference** (`src/infer.py`, `src/predict.py`): full 2048² disk, flip TTA.
5. **Instances** (`src/postprocess.py`, `src/tune.py`): threshold / gap-closing / min-area tuned for PQ on validation.
6. **Evaluation** (`src/metrics.py`): PQ, SQ, RQ, Dice, one-to-many / many-to-one counts, scored against every
   annotator's set (`evaluate_multi`).
7. **Submission** (`src/make_submission.py`): COCO RLE counts per filament.

Classical baseline: `src/classical.py` (morphological black-hat).

## Run (Kaggle, GPU T4 x2)
```bash
python -m src.build_cache --out /kaggle/working/cache
python -m src.train --cache /kaggle/working/cache --fold 0 --encoder resnet34 --pretrained 1 --epochs 30 --out /kaggle/working/runs/r34_f0
python -m src.predict --run /kaggle/working/runs/r34_f0 --split val
python -m src.predict --run /kaggle/working/runs/r34_f0 --split test
python -m src.tune --run /kaggle/working/runs/r34_f0
python -m src.make_submission --runs /kaggle/working/runs/r34_f0 --out /kaggle/working/submission.csv
```
Set `FILAMENT_DATA` to run outside Kaggle.

## Key findings so far
- Inter-annotator agreement on 296 multiply-annotated images: **PQ 0.345** (SQ 0.636, Dice 0.614).
- Classical black-hat baseline: PQ 0.089 (fold-0 subset).
