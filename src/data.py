"""Dataset indexing: per-image annotation sets (one list per annotator) and file metadata."""
import json
import os
import re

import pandas as pd

from .masks import seg_to_rle

BASE = os.environ.get("FILAMENT_DATA",
                      "/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026")
TRAIN_IMG_DIR = f"{BASE}/train/train_images"
TEST_IMG_DIR = f"{BASE}/test/test_images"
ANN_PATH = f"{BASE}/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"


def stem(fn):
    return os.path.splitext(os.path.basename(fn))[0]


def parse_meta(fn):
    """GONG filename YYYYMMDDhhmmss<Station>h -> metadata."""
    m = re.match(r"(\d{4})(\d{2})(\d{2})(\d{6})([A-Z])h", stem(fn))
    return dict(year=int(m[1]), month=int(m[2]), station=m[5], ym=f"{m[1]}-{m[2]}")


def load_annotation_sets(ann_path=ANN_PATH):
    """{stem: [[rle, ...], [rle, ...], ...]} -- each inner list is one annotator's set."""
    with open(ann_path) as f:
        coco = json.load(f)
    entry2stem = {im["id"]: stem(im["file_name"]) for im in coco["images"]}
    per_entry = {}
    for a in coco["annotations"]:
        rle = seg_to_rle(a["segmentation"])
        if rle is not None:
            per_entry.setdefault(a["image_id"], []).append(rle)
    sets = {}
    for eid in sorted(per_entry):
        sets.setdefault(entry2stem[eid], []).append(per_entry[eid])
    return sets


def file_table(img_dir):
    files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    return pd.DataFrame([dict(file=f, stem=stem(f), **parse_meta(f)) for f in files])
