"""Losses for soft (agreement-weighted) targets."""
import torch
import torch.nn.functional as F


def soft_dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    den = p.sum((1, 2, 3)) + target.sum((1, 2, 3))
    return (1 - (2 * inter + eps) / (den + eps)).mean()


def bce_dice_loss(logits, target, w_bce=1.0, w_dice=1.0):
    """BCE handles per-pixel calibration on soft labels; Dice counters the ~0.6% foreground imbalance."""
    return w_bce * F.binary_cross_entropy_with_logits(logits, target) + w_dice * soft_dice_loss(logits, target)
