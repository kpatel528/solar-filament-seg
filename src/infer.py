"""Full-disk inference with optional test-time augmentation (TTA).

tta = 0/False : single pass
tta = 3/True  : identity + horizontal flip + vertical flip
tta = 8       : all 8 dihedral views (4 rotations x optional flip), as in the 10th-place solution
"""
import numpy as np
import torch

from .dataset import normalise


def _views(n):
    if n == 8:
        return [(k, f) for k in range(4) for f in (False, True)]
    if n == 3:
        return [(0, False), (0, "h"), (0, "v")]
    return [(0, False)]


@torch.no_grad()
def predict_full(model, img_u8, device="cuda", tta=True):
    """img_u8: (H, W) limb-flattened uint8 -> (H, W) float32 filament probability."""
    n = 3 if tta is True else (1 if tta is False else int(tta))
    model.eval()
    x = torch.from_numpy(normalise(img_u8))[None, None].to(device)
    acc = torch.zeros_like(x)
    views = _views(n)
    with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
        for k, f in views:
            v = torch.rot90(x, k, (-2, -1))
            if f is True or f == "h":
                v = v.flip(-1)
            elif f == "v":
                v = v.flip(-2)
            p = torch.sigmoid(model(v).float())
            if f is True or f == "h":
                p = p.flip(-1)
            elif f == "v":
                p = p.flip(-2)
            acc += torch.rot90(p, -k, (-2, -1))
    return (acc / len(views))[0, 0].float().cpu().numpy()
