"""Export trained checkpoints as half-precision state dicts (< GitHub's 100 MB file limit).

Usage: python -m src.export_weights --runs C:/fil/runs/r34_f0 C:/fil/runs/r34_f1 --out weights
Writes <out>/<run_name>.pt (fp16) and <out>/<run_name>.json (training config)."""
import argparse
import json
import os
import shutil

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="weights")
    ap.add_argument("--weights", default="best.pt")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for run in a.runs:
        name = os.path.basename(os.path.normpath(run))
        sd = torch.load(f"{run}/{a.weights}", map_location="cpu")
        torch.save({k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}, f"{a.out}/{name}.pt")
        shutil.copy(f"{run}/config.json", f"{a.out}/{name}.json")
        print(f"{name}: {os.path.getsize(f'{a.out}/{name}.pt') / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
