# Solar Filament Segmentation (MAGFiLO · Kaggle Solar Filament Segmentation Challenge 2026)

SEA710 Computer Vision — Group 7, Seneca Polytechnic.
Instance segmentation of solar filaments in GONG H-alpha full-disk images (2048×2048), evaluated with
**Panoptic Quality (PQ)**.

## Results

OOF = out-of-fold over all 707 training images (5 folds grouped by year-month), scored against every annotator.

| Method | OOF PQ | Kaggle private / public |
|---|---|---|
| Human vs human (inter-annotator agreement, 296 images) | 0.345 | – |
| Classical baseline (morphological black-hat) | ~0.09 | 0.08 / 0.08 |
| U-Net ResNet-34 **from scratch**, 5-fold ensemble | 0.390 | 0.31 / 0.34 |
| U-Net ResNet-34 **ImageNet-pretrained**, 5-fold ensemble | 0.409 | 0.33 / 0.35 |
| + 8-view test-time augmentation | 0.415 | 0.33 / 0.36 |
| **+ learned candidate scorer + PQ-aware selection** | **0.424** | see report |

Key findings
- Annotators disagree strongly: two experts labelling the same image agree at PQ 0.345.
- ImageNet pretraining adds +0.02 PQ at identical training cost; scratch models produce more false positives and fragments.
- The best selection threshold for the learned scorer is τ = 0.22 ≈ PQ/2, matching the theory that a mask only pays
  off in PQ if its expected matched IoU exceeds PQ/2.
- Robust across the 6 GONG observatories (PQ 0.385–0.417); weakest at solar minimum (2019–2021).
- 57 ms per 2048² image for one model on an RTX 4090.

## Pipeline
1. **Preprocessing** (`src/preprocess.py`, `src/build_cache.py`): Hough-based disk localisation re-ranked by limb contrast
   (with radius sanity refit), radial limb-darkening flattening, agreement-weighted soft labels (pixel = fraction of
   annotators who marked it).
2. **Splits** (`src/splits.py`, `folds.csv`): 5 folds grouped by year-month (filaments persist for days–weeks), stratified by station.
3. **Model** (`src/model.py`, `src/train.py`): U-Net with ResNet-34 encoder and scSE decoder, 1024² full-resolution crops,
   BCE + soft Dice, AMP. Same architecture trained from scratch and from ImageNet weights.
4. **Inference** (`src/infer.py`, `src/predict.py`): full 2048² disk; 3- or 8-view test-time augmentation.
5. **Instances** (`src/postprocess.py`, `src/tune.py`): threshold, gap closing, size and confidence filters tuned for PQ on OOF maps.
6. **Learned scorer** (`src/scorer.py`): candidates at 7 threshold/gap settings → LightGBM predicts each candidate's
   expected matched IoU (cross-fitted by fold) → greedy non-overlapping selection above τ.
7. **Evaluation** (`src/metrics.py`): PQ / SQ / RQ, Dice, one-to-many and many-to-one counts, scored against every annotator.
8. **Submission** (`src/make_submission.py`): COCO RLE counts per filament.

## Reproduce
```bash
pip install -r requirements.txt
export FILAMENT_DATA=/path/to/MAGFiLO_1.0_Kaggle_2026          # Windows: set FILAMENT_DATA=C:\...\MAGFiLO_1.0_Kaggle_2026
python -m src.build_cache --out cache --folds folds.csv          # same split as the trained weights
```
Then open `notebooks/04_reproduce_results.ipynb`, set `ROOT`, and run all cells. With the weights in `weights/`
(fp16, copy each `weights/<run>.pt` to `runs/<run>/best.pt` and `weights/<run>.json` to `runs/<run>/config.json`)
it rebuilds every table, figure and submission without retraining. Retraining all 10 models takes ~3 GPU-hours on an RTX 4090.

## References
- Ahmadzadeh et al. (2024). MAGFiLO: Manually Annotated GONG Filaments from H-alpha Observations. *Scientific Data*. doi:10.1038/s41597-024-03876-y
- Kirillov et al. (2019). Panoptic Segmentation. *CVPR*. doi:10.1109/CVPR.2019.00963
- Ronneberger et al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI*.
- Guillaume (2026). Multi-family Mask2Former approach (10th-place write-up). Kaggle. doi:10.34740/kaggle/w/126726 — idea of PQ/2-based candidate selection.

## AI assistance
Parts of the code and documentation were developed with the assistance of an AI model.
