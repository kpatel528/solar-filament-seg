"""Train a U-Net with resumable, epoch-level checkpoints.

Resume an interrupted run with:
  python -m src.train --cache C:/fil/cache --fold 0 --epochs 30 \
      --out C:/fil/runs/r34_f0 \
      --resume C:/fil/runs/r34_f0/last_checkpoint.pt
"""
import argparse
import json
import os
import random
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
    """Dice between prediction>0.5 and the majority-vote label (soft>=0.5)."""
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
    ap.add_argument(
        "--resume",
        default=None,
        help="path to last_checkpoint.pt; resumes at the next completed epoch",
    )
    return ap.parse_args()


def save_checkpoint(path, model, opt, sched, scaler, epoch, best, log, args):
    """Write a complete checkpoint atomically."""
    checkpoint = {
        "epoch": epoch,
        "best": best,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "scaler": scaler.state_dict(),
        "log": log,
        "args": vars(args),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

    temp_path = path + ".tmp"
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, path)


def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)

    # Keep the original run configuration when resuming.
    config_path = os.path.join(a.out, "config.json")
    if not a.resume:
        with open(config_path, "w") as f:
            json.dump(vars(a), f, indent=2)

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    meta = pd.read_csv(f"{a.cache}/meta.csv", keep_default_na=False)
    tr = meta[(meta.split == "train") & (meta.fold != a.fold)]
    va = meta[(meta.split == "train") & (meta.fold == a.fold)]
    print(
        f"train images: {len(tr)} | val images: {len(va)} | "
        f"device: {dev} x{torch.cuda.device_count()}",
        flush=True,
    )

    model = build_model(a.encoder, bool(a.pretrained)).to(dev)
    net = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model

    dl = DataLoader(
        CropDataset(tr, a.crop, a.samples),
        batch_size=a.batch,
        shuffle=False,
        num_workers=a.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=a.workers > 0,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs * len(dl), pct_start=0.1
    )
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")

    best, log, start_epoch = -1.0, [], 0

    if a.resume:
        if not os.path.isfile(a.resume):
            raise FileNotFoundError(f"Checkpoint not found: {a.resume}")

        # Check that scheduler/training dimensions match the original run.
        checkpoint = torch.load(a.resume, map_location=dev, weights_only=False)
        saved_args = checkpoint.get("args", {})
        required_same = (
            "cache", "fold", "encoder", "pretrained", "epochs",
            "samples", "crop", "batch", "lr", "val_every", "seed",
        )
        mismatches = [
            key for key in required_same
            if key in saved_args and saved_args[key] != getattr(a, key)
        ]
        if mismatches:
            raise ValueError(
                "Resume settings differ from the checkpoint for: "
                + ", ".join(mismatches)
                + ". Use the original training settings when resuming."
            )

        model.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["optimizer"])
        sched.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"])
        best = float(checkpoint["best"])
        log = checkpoint.get("log", [])

        # Restore RNG states after model/optimizer construction.
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if dev == "cuda" and checkpoint.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])

        print(
            f"Resuming from epoch {start_epoch}; next epoch is {start_epoch + 1}; "
            f"best val dice: {best:.4f}",
            flush=True,
        )

    if start_epoch >= a.epochs:
        print(f"Checkpoint already completed {a.epochs} epochs; nothing to train.")
        return

    for ep in range(start_epoch, a.epochs):
        net.train()
        t0, tot = time.time(), 0.0

        for x, y in dl:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                logits = net(x)

            loss = bce_dice_loss(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()

        rec = dict(
            epoch=ep + 1,
            loss=round(tot / len(dl), 4),
            lr=opt.param_groups[0]["lr"],
            sec=round(time.time() - t0, 1),
        )

        if len(va) and ((ep + 1) % a.val_every == 0 or ep + 1 == a.epochs):
            scores = []
            for r in va.itertuples():
                img = cv2.imread(r.img_path, cv2.IMREAD_GRAYSCALE)
                soft = (
                    cv2.imread(r.soft_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
                    / 255.0
                )
                scores.append(
                    majority_dice(predict_full(model, img, dev, tta=False), soft)
                )

            rec["val_dice"] = round(float(np.mean(scores)), 4)
            if rec["val_dice"] > best:
                best = rec["val_dice"]
                torch.save(model.state_dict(), f"{a.out}/best.pt")
                rec["saved"] = "best"

        # Keep weights-only files for inference scripts that expect these names.
        torch.save(model.state_dict(), f"{a.out}/last.pt")
        log.append(rec)
        pd.DataFrame(log).to_csv(f"{a.out}/log.csv", index=False)

        # Full recovery point: model + optimizer + scheduler + scaler + RNG + epoch.
        save_checkpoint(
            f"{a.out}/last_checkpoint.pt",
            model, opt, sched, scaler, ep + 1, best, log, a
        )

        print(rec, flush=True)

    if not len(va):
        shutil.copy(f"{a.out}/last.pt", f"{a.out}/best.pt")

    print("done. best val dice:", best)


if __name__ == "__main__":
    main()
