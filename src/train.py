"""Train a U-Net on agreement-weighted soft labels (full-resolution crops, AMP, multi-GPU).

Usage:
  python -m src.train --cache /kaggle/working/cache --fold 0 --encoder resnet34 --pretrained 1 \
                      --epochs 30 --out /kaggle/working/runs/r34_f0
  --fold -1 trains on all training images (no validation; saves last epoch as best.pt).
"""
import argparse
import json
import os
import shutil
import time

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import CropDataset
from .infer import predict_full
from .losses import bce_dice_loss
from .model import build_model


def majority_dice(p, t, eps=1.0):
    """Dice between prediction>0.5 and the majority-vote label (soft>=0.5). Used for checkpointing."""
    pb, tb = p > 0.5, t >= 0.5
    return float((2 * (pb & tb).sum() + eps) / (pb.sum() + tb.sum() + eps))


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/kaggle/working/cache")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--pretrained", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--samples", type=int, default=1000, help="random crops per epoch")
    ap.add_argument("--crop", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val_every", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/config.json", "w") as f:
        json.dump(vars(a), f, indent=2)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    meta = pd.read_csv(f"{a.cache}/meta.csv", keep_default_na=False)
    tr = meta[(meta.split == "train") & (meta.fold != a.fold)]
    va = meta[(meta.split == "train") & (meta.fold == a.fold)]
    print(f"train images: {len(tr)} | val images: {len(va)} | device: {dev} x{torch.cuda.device_count()}")

    model = build_model(a.encoder, bool(a.pretrained)).to(dev)
    net = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    dl = DataLoader(CropDataset(tr, a.crop, a.samples), batch_size=a.batch, shuffle=False,
                    num_workers=a.workers, pin_memory=True, drop_last=True,
                    persistent_workers=a.workers > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(dl), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")

    best, log = -1.0, []
    for ep in range(a.epochs):
        net.train()
        t0, tot = time.time(), 0.0
        for x, y in dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                logits = net(x)
            loss = bce_dice_loss(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        rec = dict(epoch=ep + 1, loss=round(tot / len(dl), 4), lr=opt.param_groups[0]["lr"],
                   sec=round(time.time() - t0, 1))

        if len(va) and ((ep + 1) % a.val_every == 0 or ep + 1 == a.epochs):
            scores = []
            for r in va.itertuples():
                img = cv2.imread(r.img_path, cv2.IMREAD_GRAYSCALE)
                soft = cv2.imread(r.soft_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
                scores.append(majority_dice(predict_full(model, img, dev, tta=False), soft))
            rec["val_dice"] = round(float(np.mean(scores)), 4)
            if rec["val_dice"] > best:
                best = rec["val_dice"]
                torch.save(model.state_dict(), f"{a.out}/best.pt")
                rec["saved"] = "best"
        torch.save(model.state_dict(), f"{a.out}/last.pt")
        log.append(rec)
        pd.DataFrame(log).to_csv(f"{a.out}/log.csv", index=False)
        print(rec, flush=True)

    if not len(va):
        shutil.copy(f"{a.out}/last.pt", f"{a.out}/best.pt")
    print("done. best val dice:", best)


if __name__ == "__main__":
    main()
