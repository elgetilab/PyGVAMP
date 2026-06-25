#!/usr/bin/env python
"""Split-leakage Test 1 — frozen re-evaluation (no retraining).

Load an already-trained VAMPNet checkpoint and recompute the **full-validation**
VAMP-2 under different train/val partitions of the SAME time-lagged pair dataset:

  (a) random/interleaved 30% val indices   — our current (leaky) condition
  (b) temporally blocked 30% val indices    — honest (seam-buffered)
  (c) the whole trajectory                  — ceiling sanity (must be <= n_states)

Only the *evaluation partition* changes; the model is frozen. This isolates the
eval-set contribution to any score inflation. See
docs/split_leakage_implementation_plan.md (Test 1) and split_leakage_test_plan.md.

Reuses the exact training scorer (VAMPNet.evaluate -> VAMPScore) — nothing is
re-implemented, so the only variable between conditions is the partition.

Usage (run with a GPU shard; 1M-frame inference is slow on CPU):
  srun --partition=gputraining --gres=shard:1 --cpus-per-task=2 --mem=16G \
    python cluster_scripts/split_leak_test1_eval.py \
      --config   /mnt/hdd/experiments/trpcage_repro_v1/seed_00/.../config.yaml \
      --model    /mnt/hdd/experiments/trpcage_repro_v1/seed_00/.../models/best_model.pt \
      --cache_dir /mnt/hdd/experiments/trpcage_repro_v1/seed_00/.../cache \
      --val_frac 0.3 --n_blocks 10 --seed 0
"""
import argparse
import os

import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader

from pygv.pipe.training import create_dataset_and_loader
from pygv.vampnet import VAMPNet
from pygv.dataset.splits import make_random_split, make_blocked_split


def build_args_from_config(config_path, overrides):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    args = argparse.Namespace(**cfg)
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def eval_on_indices(model, dataset, idx, batch_size, device, label):
    loader = DataLoader(torch.utils.data.Subset(dataset, list(idx)),
                        batch_size=batch_size, shuffle=False)
    scores = model.evaluate(loader, device)
    if scores is None:
        print(f"  {label:8s}: (empty)")
        return None
    print(f"  {label:8s}: concat={scores['concat']:.4f}  "
          f"perbatch={scores['perbatch_mean']:.4f}±{scores['perbatch_std']:.4f}  "
          f"(n_pairs={len(idx)})")
    return scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, help='config.yaml from the trained run')
    p.add_argument('--model', required=True, help='best_model.pt checkpoint')
    p.add_argument('--cache_dir', default=None, help='dataset cache dir (fast path)')
    p.add_argument('--val_frac', type=float, default=0.3)
    p.add_argument('--n_blocks', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--batch_size', type=int, default=1000)
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Device: {device}")

    # Build the FULL time-lagged pair dataset exactly as training did (cache hit
    # if --cache_dir points at the run's cache). test_split=0 -> we take the full
    # dataset and partition it ourselves below.
    ds_args = build_args_from_config(args.config, {
        'split_mode': 'random',
        'use_cache': args.cache_dir is not None,
        'cache_dir': args.cache_dir,
        'batch_size': args.batch_size,
        'cpu': args.cpu,
        'runtime_stride': 1,
    })
    # test_split>0 so the internal loaders build (an empty test set crashes the
    # RandomSampler); we ignore that split and partition `dataset` ourselves below.
    dataset, *_ = create_dataset_and_loader(ds_args, is_frame_loader=False,
                                            test_split=0.3, seed=args.seed)
    n_pairs = len(dataset)
    lag_frames = int(dataset.t1_indices[0]) - int(dataset.t0_indices[0])
    print(f"Full pair dataset: {n_pairs} pairs, "
          f"n_frames={dataset.n_frames}, lag_frames={lag_frames}")

    # Load frozen model.
    model = VAMPNet.load_complete_model(args.model, map_location=device)
    model.to(device).eval()

    # Partitions.
    _, rand_val = make_random_split(n_pairs, val_frac=args.val_frac, seed=args.seed)
    _, blk_val, n_drop = make_blocked_split(
        dataset.t0_indices, dataset.t1_indices, dataset.n_frames,
        val_frac=args.val_frac, n_blocks=args.n_blocks, seed=args.seed)
    print(f"random val: {len(rand_val)} pairs | blocked val: {len(blk_val)} pairs "
          f"({n_drop} dropped at seams)")

    print("\n=== Full-validation VAMP-2 (frozen model, varying eval partition) ===")
    print("  (ceiling = n_states; concat is the honest estimator, perbatch is paper-comparable)")
    eval_on_indices(model, dataset, rand_val, args.batch_size, device, "random")   # (a)
    eval_on_indices(model, dataset, blk_val, args.batch_size, device, "blocked")   # (b)
    eval_on_indices(model, dataset, np.arange(n_pairs), args.batch_size, device, "whole")  # (c)

    print("\nRead: random >> blocked  -> eval-partition leakage is real (blocked is honest).")
    print("      random ~= blocked    -> our eval split is NOT leaking; look elsewhere for the 0.14.")


if __name__ == '__main__':
    main()
