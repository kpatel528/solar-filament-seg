"""Learned candidate scoring + PQ-aware selection (idea from the 10th-place prize solution, adapted to U-Net maps).

Why: adding a mask to an image's prediction always adds 1/2 to the PQ denominator (per annotator); it adds IoU to the
numerator only if it matches (IoU > 0.5). So a mask is worth keeping only if its EXPECTED matched IoU exceeds ~PQ/2.

Pipeline
  1. Candidates: connected components of the probability map at several (threshold, gap-closing) settings
     -> many overlapping candidate masks per image.
  2. Features per candidate: size, shape, probability statistics, image contrast, distance to the limb, and
     stability (how many other settings produce the same mask).
  3. Target per candidate: mean over the image's annotators of its best IoU with an annotated filament,
     counted as 0 below 0.5 (= exactly what it contributes to the PQ numerator).
  4. LightGBM regression, cross-fitted by fold (each fold's candidates scored by a model trained on the other folds).
  5. Selection: greedy by predicted score, keep a candidate if score > tau and it overlaps no kept mask by more than
     10% of the smaller one. tau is tuned on the out-of-fold predictions.
  6. Test: same candidates from the ensemble-averaged test maps, scored by a model trained on all folds.

Usage:
  python -m src.scorer --runs <run_f0> ... <run_f4> --cache <cache_dir> --out <results_dir>/scorer [--workers 16]
"""
import argparse
import itertools
import json
import os

import cv2
import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pycocotools import mask as mask_utils

from .data import load_annotation_sets
from .metrics import evaluate_multi
from .postprocess import extract_components
from .preprocess import radius_map
from .submission import write_submission

SETTINGS = [(0.2, 0), (0.2, 3), (0.3, 3), (0.4, 3), (0.5, 0), (0.5, 3), (0.6, 3)]
TAUS = [0.10, 0.15, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40]
FEATURES = ["thr", "close_px", "area", "log_area", "p_mean", "p_max", "p_p90", "p_std", "frac_hi",
            "img_in", "img_ring", "contrast", "r_mean", "r_max", "elong", "major", "fill", "stability",
            "n_cands_setting", "img_fg_frac"]


def candidate_table(prob, img, rr):
    """All candidates for one image: list of RLEs and a feature DataFrame (one row per candidate)."""
    rles, rows = [], []
    disk = rr < 0.98
    img_fg_frac = float((prob[disk] > 0.5).mean())
    for thr, cp in SETTINGS:
        comps = extract_components(prob, rr, thr, cp, min_area=100)
        for rle, area, score in comps:
            x, y, w, h = [int(v) for v in mask_utils.toBbox(rle)]
            pad = 8
            y0, y1, x0, x1 = max(y - pad, 0), min(y + h + pad, prob.shape[0]), max(x - pad, 0), min(x + w + pad, prob.shape[1])
            m = mask_utils.decode(rle)[y0:y1, x0:x1].astype(bool)
            ring = cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & ~m
            pc, ic, rc = prob[y0:y1, x0:x1][m], img[y0:y1, x0:x1].astype(np.float32) / 255.0, rr[y0:y1, x0:x1][m]
            ys, xs = np.nonzero(m)
            cov = np.cov(np.vstack([xs, ys])) if len(xs) > 2 else np.eye(2)
            ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
            img_in, img_ring = float(ic[m].mean()), float(ic[ring].mean()) if ring.any() else float(ic[m].mean())
            rles.append(rle)
            rows.append(dict(thr=thr, close_px=cp, area=area, log_area=np.log(area), p_mean=float(pc.mean()),
                             p_max=float(pc.max()), p_p90=float(np.percentile(pc, 90)), p_std=float(pc.std()),
                             frac_hi=float((pc > 0.8).mean()), img_in=img_in, img_ring=img_ring,
                             contrast=img_ring - img_in, r_mean=float(rc.mean()), r_max=float(rc.max()),
                             elong=float(np.sqrt(ev[0] / max(ev[1], 1e-6))), major=float(4 * np.sqrt(max(ev[0], 0))),
                             fill=area / max(w * h, 1), n_cands_setting=len(comps), img_fg_frac=img_fg_frac))
    df = pd.DataFrame(rows, columns=[c for c in FEATURES if c != "stability"])
    if rles:
        iou = np.asarray(mask_utils.iou(rles, rles, [0] * len(rles)), dtype=float).reshape(len(rles), len(rles))
        df["stability"] = (iou > 0.5).sum(1) - 1
    else:
        iou = np.zeros((0, 0))
        df["stability"] = []
    return rles, df.astype(float), iou


def targets(rles, gt_sets):
    """Mean over annotators of best IoU with an annotated filament (0 if <= 0.5)."""
    if not rles:
        return np.zeros(0)
    t = np.zeros(len(rles))
    for gt in gt_sets:
        if not gt:
            continue
        iou = np.asarray(mask_utils.iou(rles, gt, [0] * len(gt)), dtype=float).reshape(len(rles), len(gt)).max(1)
        t += np.where(iou > 0.5, iou, 0.0)
    return t / len(gt_sets)


def select(rles, q, iou, tau, max_overlap=0.1):
    """Greedy PQ-aware selection: highest predicted score first, keep if q > tau and no big overlap with kept masks."""
    if not rles:
        return []
    areas = mask_utils.area(rles).astype(float) if len(rles) <= 255 else \
        np.concatenate([mask_utils.area(rles[i:i + 255]).astype(float) for i in range(0, len(rles), 255)])
    inter = iou * (areas[:, None] + areas[None, :]) / (1 + iou)
    kept = []
    for i in np.argsort(-q):
        if q[i] <= tau:
            break
        if all(inter[i, j] <= max_overlap * min(areas[i], areas[j]) for j in kept):
            kept.append(i)
    return [rles[i] for i in kept]


def fit_lgb(X, y, seeds=(0, 1, 2)):
    models = []
    for s in seeds:
        m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=s, verbose=-1)
        models.append(m.fit(X, y))
    return models


def predict_lgb(models, X):
    return np.mean([m.predict(X) for m in models], axis=0) if len(X) else np.zeros(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val_name", default="val_probs")
    ap.add_argument("--test_name", default="test_probs")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    meta = pd.read_csv(f"{a.cache}/meta.csv", keep_default_na=False).set_index("stem")
    gt_sets = load_annotation_sets()
    by_stem, fold_of = {}, {}
    for run in a.runs:
        fold = json.load(open(f"{run}/config.json"))["fold"]
        for f in sorted(os.listdir(f"{run}/{a.val_name}")):
            by_stem.setdefault(f[:-4], []).append(run)
            fold_of[f[:-4]] = fold
    stems = sorted(by_stem)
    print(f"OOF images: {len(stems)}")

    def build(stem, runs, name):
        r = meta.loc[stem]
        prob = np.mean([np.load(f"{run}/{name}/{stem}.npy").astype(np.float32) for run in runs], axis=0)
        img = cv2.imread(r.img_path, cv2.IMREAD_GRAYSCALE)
        return candidate_table(prob, img, radius_map(r.cx, r.cy, r.r))

    oof = Parallel(n_jobs=a.workers, backend="threading", verbose=5)(
        delayed(build)(s, by_stem[s], a.val_name) for s in stems)
    frames = []
    for s, (rles, df, _) in zip(stems, oof):
        df = df.copy()
        df["stem"], df["fold"], df["target"] = s, fold_of[s], targets(rles, gt_sets[s])
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data[FEATURES] = data[FEATURES].astype(float)
    print(f"candidates: {len(data)} | with target > 0: {(data.target > 0).mean():.1%}")

    data["q"] = 0.0
    for k in sorted(data.fold.unique()):
        tr, te = data.fold != k, data.fold == k
        data.loc[te, "q"] = predict_lgb(fit_lgb(data.loc[tr, FEATURES], data.loc[tr, "target"]), data.loc[te, FEATURES])
    corr = np.corrcoef(data.q, data.target)[0, 1]
    print(f"OOF correlation(pred, target) = {corr:.3f} | p_mean alone: {np.corrcoef(data.p_mean, data.target)[0, 1]:.3f}")

    offsets = np.cumsum([0] + [len(r[0]) for r in oof])
    gts = {s: gt_sets[s] for s in stems}
    rows = []
    for tau in TAUS:
        preds = {s: select(oof[i][0], data.q.values[offsets[i]:offsets[i + 1]], oof[i][2], tau)
                 for i, s in enumerate(stems)}
        _, summ = evaluate_multi(gts, preds)
        rows.append(dict(tau=tau, **{k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v)
                                     for k, v in summ.items()}))
    grid = pd.DataFrame(rows).sort_values("pq_global", ascending=False)
    grid.to_csv(f"{a.out}/tau_grid.csv", index=False)
    print(grid.to_string(index=False))
    tau = float(grid.iloc[0].tau)

    final = fit_lgb(data[FEATURES], data.target)
    imp = pd.Series(np.mean([m.booster_.feature_importance("gain") for m in final], axis=0), index=FEATURES)
    (imp / imp.sum()).sort_values(ascending=False).round(4).to_csv(f"{a.out}/feature_importance.csv")
    print("top features:", (imp / imp.sum()).sort_values(ascending=False).head(8).round(3).to_dict())
    data.drop(columns=[]).to_csv(f"{a.out}/oof_candidates.csv", index=False)
    json.dump(dict(tau=tau, oof=grid.iloc[0].to_dict(), corr=float(corr), settings=SETTINGS),
              open(f"{a.out}/scorer.json", "w"), indent=2)

    if all(os.path.isdir(f"{run}/{a.test_name}") for run in a.runs):
        te = meta[meta.split == "test"].index.tolist()
        tst = Parallel(n_jobs=a.workers, backend="threading", verbose=5)(
            delayed(build)(s, a.runs, a.test_name) for s in te)
        preds = {s: select(r, predict_lgb(final, df[FEATURES]), iou, tau) for s, (r, df, iou) in zip(te, tst)}
        sub = write_submission(preds, f"{a.out}/submission.csv")
        print(f"test submission: {len(sub)} rows, {len(sub) / len(te):.2f} per image -> {a.out}/submission.csv")


if __name__ == "__main__":
    main()
