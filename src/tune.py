"""Tune post-processing (threshold, gap closing, min area) on validation probability maps for PQ,
scoring against every annotator's set. Writes <run>/postproc.json and <run>/tune_grid.csv.

Usage: python -m src.tune --run /kaggle/working/runs/r34_f0
"""
import argparse
import itertools
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .data import load_annotation_sets
from .metrics import evaluate_multi
from .postprocess import instances_from_response
from .preprocess import radius_map

GRID = dict(thr=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7], close_px=[0, 3, 6, 10], min_area=[100, 250, 500, 800])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    cfg = json.load(open(f"{a.run}/config.json"))
    meta = pd.read_csv(f"{cfg['cache']}/meta.csv", keep_default_na=False)
    va = meta[(meta.split == "train") & (meta.fold == cfg["fold"])]
    gt_sets = load_annotation_sets()
    probs = {r.stem: (np.load(f"{a.run}/val_probs/{r.stem}.npy").astype(np.float32),
                      radius_map(r.cx, r.cy, r.r)) for r in va.itertuples()}
    gt_val = {s: gt_sets[s] for s in probs}

    def run(thr, close_px, min_area):
        preds = {s: instances_from_response(p, rr, thr, close_px, min_area) for s, (p, rr) in probs.items()}
        _, summ = evaluate_multi(gt_val, preds)
        return dict(thr=thr, close_px=close_px, min_area=min_area,
                    preds_per_img=round(float(np.mean([len(v) for v in preds.values()])), 2), **summ)

    combos = list(itertools.product(*GRID.values()))
    res = Parallel(n_jobs=a.workers, backend="threading", verbose=5)(delayed(run)(*c) for c in combos)
    grid = pd.DataFrame(res).sort_values("pq_global", ascending=False)
    grid.to_csv(f"{a.run}/tune_grid.csv", index=False)
    best = grid.iloc[0]
    params = dict(thr=float(best.thr), close_px=int(best.close_px), min_area=int(best.min_area))
    json.dump(dict(params=params, val_pq_global=float(best.pq_global)), open(f"{a.run}/postproc.json", "w"), indent=2)
    print(grid.head(10).round(3).to_string())
    print("best:", params)


if __name__ == "__main__":
    main()
