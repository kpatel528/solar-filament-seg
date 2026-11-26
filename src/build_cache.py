"""Build the preprocessing cache used by every model:
  * disk parameters (cx, cy, r) for all train + test images
  * limb-flattened uint8 PNGs (quiet Sun ~ 127)
  * agreement-weighted soft labels for train images: pixel = fraction of annotators that marked it
  * meta.csv with paths, folds, station/year metadata

Usage:  python -m src.build_cache --out /kaggle/working/cache --workers 4
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pycocotools import mask as mask_utils

from .data import TEST_IMG_DIR, TRAIN_IMG_DIR, file_table, load_annotation_sets
from .preprocess import find_disk, flatten_disk, read_gray, to_uint8
from .splits import make_folds


def soft_label(sets, shape=(2048, 2048)):
    """Agreement-weighted target: mean over annotators of each annotator's union mask -> uint8 0..255."""
    acc = np.zeros(shape, np.float32)
    for rles in sets:
        acc += mask_utils.decode(mask_utils.merge(rles)).astype(np.float32)
    return np.round(acc / len(sets) * 255).astype(np.uint8)


def process(row, out_dir, sets):
    img = read_gray(row["path"])
    cx, cy, r = find_disk(img)
    flat, _ = flatten_disk(img, cx, cy, r)
    img_path = f"{out_dir}/img/{row['stem']}.png"
    cv2.imwrite(img_path, to_uint8(flat))
    soft_path = ""
    if sets:
        soft_path = f"{out_dir}/soft/{row['stem']}.png"
        cv2.imwrite(soft_path, soft_label(sets))
    return dict(stem=row["stem"], cx=cx, cy=cy, r=r, img_path=img_path, soft_path=soft_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/kaggle/working/cache")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--folds", default=None, help="CSV with columns stem,fold to reuse an existing split "
                                                  "(fold assignment can differ between scikit-learn versions)")
    a = ap.parse_args()
    os.makedirs(f"{a.out}/img", exist_ok=True)
    os.makedirs(f"{a.out}/soft", exist_ok=True)

    gt_sets = load_annotation_sets()
    tr = make_folds(file_table(TRAIN_IMG_DIR))
    if a.folds:
        tr["fold"] = tr.stem.map(pd.read_csv(a.folds).set_index("stem").fold).astype(int)
    tr["split"], tr["path"] = "train", TRAIN_IMG_DIR + "/" + tr.file
    te = file_table(TEST_IMG_DIR)
    te["split"], te["fold"], te["path"] = "test", -1, TEST_IMG_DIR + "/" + te.file
    df = pd.concat([tr, te], ignore_index=True)

    recs = Parallel(n_jobs=a.workers, verbose=5)(
        delayed(process)(row, a.out, gt_sets.get(row["stem"]) if row["split"] == "train" else None)
        for row in df.to_dict("records"))
    meta = pd.concat([df.reset_index(drop=True), pd.DataFrame(recs).drop(columns="stem")], axis=1)
    meta["n_sets"] = meta.stem.map(lambda s: len(gt_sets.get(s, [])))
    med = meta.r.median()
    meta["disk_outlier"] = (meta.r - med).abs() > 25
    meta.to_csv(f"{a.out}/meta.csv", index=False)
    print(meta.groupby("split").size().to_dict(), "| radius:", meta.r.describe().round(1).to_dict())
    print("disk outliers (|r - median| > 25 px):")
    print(meta.loc[meta.disk_outlier, ["stem", "split", "cx", "cy", "r"]].to_string())


if __name__ == "__main__":
    main()
