"""Full-disk inference with optional flip test-time augmentation."""
import numpy as np
import torch

from .dataset import normalise


@torch.no_grad()
def predict_full(model, img_u8, device="cuda", tta=True):
    """img_u8: (H, W) limb-flattened uint8 -> (H, W) float32 filament probability."""
    model.eval()
    x = torch.from_numpy(normalise(img_u8))[None, None].to(device)
    use_amp = device.startswith("cuda")
    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        p = torch.sigmoid(model(x).float())
        if tta:
            p = p + torch.sigmoid(model(x.flip(-1)).float()).flip(-1)
            p = p + torch.sigmoid(model(x.flip(-2)).float()).flip(-2)
            p = p / 3
    return p[0, 0].float().cpu().numpy()
