"""Turn test probability maps (averaged over one or more runs) into the Kaggle CSV.

Usage: python -m src.make_submission --runs /kaggle/working/runs/r34_f0 --out /kaggle/working/submission.csv
       (post-processing parameters are read from the first run's postproc.json unless overridden)
"""
import argparse
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .masks import counts_str_to_rle, rle_to_mask
from .postprocess import instances_from_response
from .preprocess import radius_map
from .submission import write_submission


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="/kaggle/working/submission.csv")
    ap.add_argument("--thr", type=float)
    ap.add_argument("--close_px", type=int)
    ap.add_argument("--min_area", type=int)
    ap.add_argument("--min_score", type=float)
    ap.add_argument("--params", default=None, help="postproc.json (default: first run's)")
    ap.add_argument("--cache", default=None, help="dir with meta.csv (default: first run's config)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    cfg = json.load(open(f"{a.runs[0]}/config.json"))
    params = json.load(open(a.params or f"{a.runs[0]}/postproc.json"))["params"]
    for k in ("thr", "close_px", "min_area", "min_score"):
        if getattr(a, k) is not None:
            params[k] = getattr(a, k)
    print("post-processing:", params, "| runs:", a.runs)

    meta = pd.read_csv(f"{a.cache or cfg['cache']}/meta.csv", keep_default_na=False)
    te = meta[meta.split == "test"]

    def one(r):
        p = np.mean([np.load(f"{run}/test_probs/{r.stem}.npy").astype(np.float32) for run in a.runs], axis=0)
        return r.stem, instances_from_response(p, radius_map(r.cx, r.cy, r.r), **params)

    preds = dict(Parallel(n_jobs=a.workers, backend="threading")(delayed(one)(r) for r in te.itertuples()))
    sub = write_submission(preds, a.out)
    n_img = sub.filament_id.str.rsplit("_", n=1).str[0].nunique() if len(sub) else 0
    print(f"rows: {len(sub)} | images with >=1 prediction: {n_img}/{len(te)} | "
          f"mean per image: {len(sub) / len(te):.2f}")
    if len(sub):
        print("round-trip pixels of first row:", int(rle_to_mask(counts_str_to_rle(sub.segmentation_rle.iloc[0])).sum()))


if __name__ == "__main__":
    main()
