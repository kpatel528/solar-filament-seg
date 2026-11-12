"""Kaggle submission writer: one row per predicted filament, RLE counts only (no quotes)."""
import pandas as pd

from .masks import rle_to_counts_str


def write_submission(pred_by_stem, path="submission.csv"):
    rows = [(f"{s}_{i}", rle_to_counts_str(r))
            for s, rles in pred_by_stem.items() for i, r in enumerate(rles, 1)]
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    df.to_csv(path, index=False)
    return df
