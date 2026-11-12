"""PyTorch datasets: full-resolution random crops with agreement-weighted soft labels."""
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

MEAN, STD = 0.5, 0.25


def normalise(img_u8):
    return (img_u8.astype(np.float32) / 255.0 - MEAN) / STD


class CropDataset(Dataset):
    """Each item: a random `crop`x`crop` window from a random training image.
    With probability `pos_frac` the window is placed so it contains a labelled pixel, because
    filaments occupy only ~0.6% of the disk. Augmentation: rot90, flips, contrast/brightness/gamma."""

    def __init__(self, meta, crop=1024, samples=1000, pos_frac=0.7, augment=True):
        self.meta = meta.reset_index(drop=True)
        self.crop, self.samples, self.pos_frac, self.augment = crop, samples, pos_frac, augment

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        rng = np.random.default_rng()
        row = self.meta.iloc[rng.integers(len(self.meta))]
        img = cv2.imread(row.img_path, cv2.IMREAD_GRAYSCALE)
        soft = cv2.imread(row.soft_path, cv2.IMREAD_GRAYSCALE)
        H, W = img.shape
        c = self.crop
        ys, xs = (np.nonzero(soft) if rng.random() < self.pos_frac else ((), ()))
        if len(ys):
            k = rng.integers(len(ys))
            y0 = int(np.clip(ys[k] - rng.integers(c // 8, 7 * c // 8), 0, H - c))
            x0 = int(np.clip(xs[k] - rng.integers(c // 8, 7 * c // 8), 0, W - c))
        else:
            y0, x0 = int(rng.integers(0, H - c + 1)), int(rng.integers(0, W - c + 1))
        img, soft = img[y0:y0 + c, x0:x0 + c], soft[y0:y0 + c, x0:x0 + c]

        if self.augment:
            k = int(rng.integers(4))
            img, soft = np.rot90(img, k), np.rot90(soft, k)
            if rng.random() < 0.5:
                img, soft = img[:, ::-1], soft[:, ::-1]
        x = img.astype(np.float32) / 255.0
        if self.augment:
            x = (x - 0.5) * rng.uniform(0.8, 1.2) + 0.5 + rng.uniform(-0.05, 0.05)
            x = np.clip(x, 0, 1) ** rng.uniform(0.8, 1.25)
        x = (x - MEAN) / STD
        y = np.ascontiguousarray(soft, dtype=np.float32) / 255.0
        return torch.from_numpy(np.ascontiguousarray(x))[None], torch.from_numpy(y)[None]
