"""Mask/RLE utilities shared by every module: COCO polygon <-> RLE <-> numpy <-> Kaggle string."""
import numpy as np
from pycocotools import mask as mask_utils

H = W = 2048


def seg_to_rle(segmentation, h=H, w=W):
    """COCO segmentation (polygon list or RLE dict) -> one compressed RLE, or None if degenerate."""
    if isinstance(segmentation, list):
        polys = [p for p in segmentation if len(p) >= 6]
        if not polys:
            return None
        return mask_utils.merge(mask_utils.frPyObjects(polys, h, w))
    if isinstance(segmentation.get("counts"), list):
        return mask_utils.frPyObjects(segmentation, h, w)
    return segmentation


def rle_to_mask(rle):
    return mask_utils.decode(rle).astype(bool)


def mask_to_rle(mask):
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))


def rle_to_counts_str(rle):
    """Kaggle submission format: RLE counts only, as a plain string."""
    c = rle["counts"]
    return c.decode("ascii") if isinstance(c, bytes) else c


def counts_str_to_rle(counts, h=H, w=W):
    return {"size": [h, w], "counts": counts.encode("ascii")}
