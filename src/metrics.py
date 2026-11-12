"""Panoptic Quality (Kirillov et al., CVPR 2019) + Dice + fragmentation diagnostics.
Operates entirely on COCO RLEs so full 2048x2048 masks never need to be held in memory."""
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment


def _area(rles, chunk=255):
    """pycocotools.area overflows (uint8 bug under NumPy 2) for >255 RLEs -> chunk it."""
    if not rles:
        return np.zeros(0)
    return np.concatenate([np.asarray(mask_utils.area(rles[i:i + chunk]), dtype=float)
                           for i in range(0, len(rles), chunk)])


def _pairwise(gt, pr):
    """IoU and intersection-area matrices, shape (n_gt, n_pred)."""
    if not gt or not pr:
        z = np.zeros((len(gt), len(pr)))
        return z, z
    iou = np.asarray(mask_utils.iou(pr, gt, [0] * len(gt)), dtype=float).reshape(len(pr), len(gt)).T
    a_g, a_p = _area(gt)[:, None], _area(pr)[None, :]
    return iou, iou * (a_g + a_p) / (1.0 + iou)


def image_scores(gt, pr, iou_thr=0.5, touch_frac=0.1):
    """gt, pr: lists of RLE dicts for ONE image."""
    n_g, n_p = len(gt), len(pr)
    iou, inter = _pairwise(gt, pr)
    tp_ious = []
    if n_g and n_p:
        r, c = linear_sum_assignment(-iou)
        tp_ious = [iou[i, j] for i, j in zip(r, c) if iou[i, j] > iou_thr]
    tp = len(tp_ious)
    fp, fn = n_p - tp, n_g - tp
    denom = tp + 0.5 * fp + 0.5 * fn
    sum_iou = float(np.sum(tp_ious))
    o2m = m2o = 0
    if n_g and n_p:
        a_p, a_g = np.maximum(_area(pr), 1), np.maximum(_area(gt), 1)
        o2m = int(((inter / a_p[None, :] >= touch_frac).sum(1) >= 2).sum())
        m2o = int(((inter / a_g[:, None] >= touch_frac).sum(0) >= 2).sum())
        ug, up = mask_utils.merge(gt), mask_utils.merge(pr)
        i = float(mask_utils.area(mask_utils.merge([ug, up], intersect=True)))
        dice = 2 * i / (float(mask_utils.area(ug)) + float(mask_utils.area(up)))
    else:
        dice = 1.0 if (n_g == 0 and n_p == 0) else 0.0
    return dict(n_gt=n_g, n_pred=n_p, tp=tp, fp=fp, fn=fn, sum_iou=sum_iou,
                pq=sum_iou / denom if denom > 0 else 1.0,
                sq=sum_iou / tp if tp else 0.0,
                rq=tp / denom if denom > 0 else 1.0,
                dice=dice, one_to_many=o2m, many_to_one=m2o)


def _summarise(df, w):
    T, P, N, S = (df.tp * w).sum(), (df.fp * w).sum(), (df.fn * w).sum(), (df.sum_iou * w).sum()
    return dict(pq_mean_per_image=df.groupby("image").pq.mean().mean(),
                pq_global=S / (T + 0.5 * P + 0.5 * N) if (T + P + N) else 1.0,
                sq_global=S / T if T else 0.0,
                dice_mean=df.groupby("image").dice.mean().mean(),
                tp=round(T, 1), fp=round(P, 1), fn=round(N, 1),
                one_to_many=int(df.one_to_many.sum()), many_to_one=int(df.many_to_one.sum()))


def evaluate(gt_by_img, pred_by_img, **kw):
    """Single GT set per image. Both args: {image_key: [RLE, ...]}."""
    keys = sorted(set(gt_by_img) | set(pred_by_img))
    df = pd.DataFrame([dict(image=k, **image_scores(gt_by_img.get(k, []), pred_by_img.get(k, []), **kw))
                       for k in keys])
    return df, _summarise(df, pd.Series(1.0, index=df.index))


def evaluate_multi(gt_sets, pred_by_img, **kw):
    """gt_sets: {img: [[rle..], [rle..], ...]} one list per annotator.
    Scores the prediction against EVERY annotation set (expected PQ vs a random annotator);
    each image is weighted equally regardless of how many annotators it has."""
    rows = [dict(image=k, ann_set=j, **image_scores(gt, pred_by_img.get(k, []), **kw))
            for k, sets in gt_sets.items() for j, gt in enumerate(sets)]
    df = pd.DataFrame(rows)
    return df, _summarise(df, 1.0 / df.groupby("image").image.transform("size"))
