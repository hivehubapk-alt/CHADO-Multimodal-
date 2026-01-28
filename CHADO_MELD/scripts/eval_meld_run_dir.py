#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import subprocess
import sys


def pick_ckpt(run_dir: str) -> str:
    # Prefer common "best" names first
    candidates = []
    patterns = [
        os.path.join(run_dir, "best.pt"),
        os.path.join(run_dir, "*best*.pt"),
        os.path.join(run_dir, "*.pt"),
    ]
    for p in patterns:
        candidates.extend(glob.glob(p))

    candidates = [c for c in candidates if os.path.isfile(c)]
    if not candidates:
        raise FileNotFoundError(f"No .pt checkpoints found in: {run_dir}")

    # Sort by modified time (newest last)
    candidates.sort(key=lambda x: os.path.getmtime(x))
    # Prefer exact best.pt if present
    for c in candidates[::-1]:
        if os.path.basename(c) == "best.pt":
            return c
    # Else take newest "*best*.pt" if any
    for c in candidates[::-1]:
        if "best" in os.path.basename(c).lower():
            return c
    # Else newest checkpoint
    return candidates[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    ckpt = pick_ckpt(args.run_dir)
    print(f"[CKPT] {ckpt}")

    cmd = [
        sys.executable,
        "scripts/eval_meld_metrics.py",
        "--config", args.config,
        "--ckpt", ckpt,
        "--split", args.split,
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
    ]
    if args.report:
        cmd.append("--report")

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
