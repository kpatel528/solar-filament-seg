"""Save flip-TTA probability maps (float16 .npy) for a validation fold or the test set.

Usage:
  python -m src.predict --run /kaggle/working/runs/r34_f0 --split val
  python -m src.predict --run /kaggle/working/runs/r34_f0 --split test
"""
import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd
import torch

from .infer import predict_full
from .model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--weights", default="best.pt")
    ap.add_argument("--tta", type=int, default=3, help="views: 1 (none), 3 (flips), 8 (all rotations+flips)")
    ap.add_argument("--out_name", default=None, help="folder name for the maps (default: <split>_probs)")
    ap.add_argument("--cache", default=None, help="dir with meta.csv (default: run's config)")
    a = ap.parse_args()

    cfg = json.load(open(f"{a.run}/config.json"))
    meta = pd.read_csv(f"{a.cache or cfg['cache']}/meta.csv", keep_default_na=False)
    rows = meta[(meta.split == "train") & (meta.fold == cfg["fold"])] if a.split == "val" \
        else meta[meta.split == "test"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg["encoder"], pretrained=False).to(dev)
    sd = torch.load(f"{a.run}/{a.weights}", map_location=dev)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})

    out = f"{a.run}/{a.out_name or a.split + '_probs'}"
    os.makedirs(out, exist_ok=True)
    for i, r in enumerate(rows.itertuples(), 1):
        img = cv2.imread(r.img_path, cv2.IMREAD_GRAYSCALE)
        np.save(f"{out}/{r.stem}.npy", predict_full(model, img, dev, tta=a.tta).astype(np.float16))
        if i % 20 == 0 or i == len(rows):
            print(f"{i}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
