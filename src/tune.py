"""Fast post-processing search for PQ on out-of-fold probability maps, scored against EVERY
annotator's set (each image weighted equally).

Speed trick: components and their IoU against each annotator are computed once per
(image, thr, close_px); the min_area / min_score filters are then just row selections.

Usage:
  python -m src.tune --runs /kaggle/working/runs/r34_f0 [/kaggle/input/.../runs/r34_f1 ...] \
                     --cache /kaggle/working/cache --out /kaggle/working/postproc.json
"""
import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pycocotools import mask as mask_utils

from .data import load_annotation_sets
from .metrics import evaluate_multi
from .postprocess import extract_components, instances_from_response
from .preprocess import radius_map

THR = [0.2, 0.3, 0.4, 0.5]
CLOSE = [0, 3, 6, 10]
MIN_AREA = [100, 250, 400, 500, 650, 800, 1000]
MIN_SCORE = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8]


def image_tables(prob, rr, gt_sets):
    """For each (thr, close): areas, scores, and IoU matrices vs each annotator set."""
    out = {}
    for thr, cp in itertools.product(THR, CLOSE):
        comps = extract_components(prob, rr, thr, cp, min_area=min(MIN_AREA))
        rles = [c[0] for c in comps]
        areas = np.array([c[1] for c in comps], dtype=float)
        scores = np.array([c[2] for c in comps], dtype=float)
        ious = [np.asarray(mask_utils.iou(rles, gt, [0] * len(gt)), dtype=float).reshape(len(rles), len(gt))
                if rles and gt else np.zeros((len(rles), len(gt))) for gt in gt_sets]
        out[(thr, cp)] = (areas, scores, ious)
    return out


def score_tables(tables, ma, ms):
    """Weighted totals (TP, FP, FN, sum IoU) over all images for one filter setting."""
    T = P = N = S = 0.0
    for per_img in tables:
        areas, scores, ious = per_img
        keep = (areas >= ma) & (scores >= ms)
        w = 1.0 / len(ious)
        for iou in ious:
            sub = iou[keep]
            hits = sub[sub > 0.5]          # IoU>0.5 matches are unique for disjoint predictions
            tp = hits.size
            T += w * tp; P += w * (int(keep.sum()) - tp); N += w * (iou.shape[1] - tp); S += w * hits.sum()
    return T, P, N, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run dirs, each with config.json and val_probs/")
    ap.add_argument("--cache", default=None, help="dir with meta.csv (defaults to first run's config)")
    ap.add_argument("--out", default=None, help="where to write postproc.json (default: first run dir)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--probs_name", default="val_probs", help="folder with OOF maps inside each run")
    a = ap.parse_args()

    cfg0 = json.load(open(f"{a.runs[0]}/config.json"))
    meta = pd.read_csv(f"{a.cache or cfg0['cache']}/meta.csv", keep_default_na=False).set_index("stem")
    gt_sets = load_annotation_sets()
    # Group runs by stem: several architectures trained on the same fold are AVERAGED (= the test ensemble).
    by_stem, folds = {}, set()
    for run in a.runs:
        fold = json.load(open(f"{run}/config.json"))["fold"]
        folds.add(fold)
        for f in sorted(os.listdir(f"{run}/{a.probs_name}")):
            by_stem.setdefault(f[:-4], []).append(run)
    items = sorted(by_stem.items())
    print(f"OOF images: {len(items)} from folds {sorted(folds)} | models per image: "
          f"{sorted(set(len(v) for v in by_stem.values()))}")

    def load_prob(stem, runs):
        return np.mean([np.load(f"{run}/{a.probs_name}/{stem}.npy").astype(np.float32) for run in runs], axis=0)

    def build(stem, runs):
        r = meta.loc[stem]
        return image_tables(load_prob(stem, runs), radius_map(r.cx, r.cy, r.r), gt_sets[stem])

    tabs = Parallel(n_jobs=a.workers, backend="threading", verbose=5)(delayed(build)(s, runs) for s, runs in items)

    rows = []
    for thr, cp in itertools.product(THR, CLOSE):
        per = [t[(thr, cp)] for t in tabs]
        for ma, ms in itertools.product(MIN_AREA, MIN_SCORE):
            T, P, N, S = score_tables(per, ma, ms)
            rows.append(dict(thr=thr, close_px=cp, min_area=ma, min_score=ms,
                             pq=S / (T + 0.5 * P + 0.5 * N), sq=S / max(T, 1e-9),
                             tp=round(T, 1), fp=round(P, 1), fn=round(N, 1),
                             preds_per_img=round((T + P) / len(items), 2)))
    grid = pd.DataFrame(rows).sort_values("pq", ascending=False)
    out_dir = os.path.dirname(a.out) if a.out else a.runs[0]
    grid.to_csv(f"{out_dir}/tune_grid.csv", index=False)
    best = grid.iloc[0]
    params = dict(thr=float(best.thr), close_px=int(best.close_px),
                  min_area=int(best.min_area), min_score=float(best.min_score))
    print(grid.head(15).round(4).to_string())

    # full evaluation (with Dice and fragmentation counts) at the chosen setting
    preds, gts = {}, {}
    for stem, runs in items:
        r = meta.loc[stem]
        preds[stem] = instances_from_response(load_prob(stem, runs), radius_map(r.cx, r.cy, r.r), **params)
        gts[stem] = gt_sets[stem]
    df, summ = evaluate_multi(gts, preds)
    df.to_csv(f"{out_dir}/oof_per_image.csv", index=False)
    summ = {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in summ.items()}
    print("best:", params)
    print("OOF summary at best:", summ)
    json.dump(dict(params=params, oof=summ, n_images=len(items), runs=a.runs),
              open(a.out or f"{a.runs[0]}/postproc.json", "w"), indent=2)


if __name__ == "__main__":
    main()
