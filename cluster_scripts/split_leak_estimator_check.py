#!/usr/bin/env python
"""Estimator-bias check — is the ~0.14 gap a per-batch VAMP-2 artifact?

The paper-comparable score is the **per-batch** VAMP-2 (mean over mini-batches),
which is biased HIGH for small batches: with batch B << N the covariance
estimates are less rank-deficient, so ||K||_F^2 (and thus VAMP-2) is larger than
the honest full-set ("concat") estimate. This sweeps batch size with SHUFFLED
batches and reports both estimators, to quantify how much the gap to ~4.79 is the
estimator rather than the model.

Frozen model, no retraining. Reuses VAMPNet.evaluate()/VAMPScore. Evaluates on a
fixed representative random subset of pairs (spans all states) so the only thing
that varies is the batch size.

Run (GPU shard):
  srun --partition=gputraining --gres=shard:1 --cpus-per-task=2 --mem=16G \
    python cluster_scripts/split_leak_estimator_check.py --config ... --model ... --cache_dir ...
"""
import argparse

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from pygv.pipe.training import create_dataset_and_loader
from pygv.vampnet import VAMPNet

# reuse the config->args helper from the Test 1 script
import importlib.util, os
_spec = importlib.util.spec_from_file_location(
    "t1", os.path.join(os.path.dirname(__file__), "split_leak_test1_eval.py"))
_t1 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_t1)
build_args_from_config = _t1.build_args_from_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--cache_dir', default=None)
    p.add_argument('--subset', type=int, default=200_000, help='pairs to evaluate on (representative)')
    p.add_argument('--batch_sizes', type=int, nargs='+',
                   default=[64, 128, 256, 512, 1024, 2048])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Device: {device}")

    ds_args = build_args_from_config(args.config, {
        'split_mode': 'random', 'use_cache': args.cache_dir is not None,
        'cache_dir': args.cache_dir, 'batch_size': 1000, 'cpu': args.cpu,
        'runtime_stride': 1,
    })
    dataset, *_ = create_dataset_and_loader(ds_args, is_frame_loader=False,
                                            test_split=0.3, seed=args.seed)
    n_pairs = len(dataset)

    # Fixed representative subset (random across the whole trajectory -> spans states).
    rng = np.random.default_rng(args.seed)
    n_sub = min(args.subset, n_pairs)
    sub = np.sort(rng.choice(n_pairs, size=n_sub, replace=False))
    subset = torch.utils.data.Subset(dataset, sub.tolist())
    print(f"Evaluating on {n_sub} pairs (of {n_pairs}); concat should be ~flat, "
          f"perbatch should rise as batch shrinks.\n")

    model = VAMPNet.load_complete_model(args.model, map_location=device)
    model.to(device).eval()

    print(f"{'batch':>6}  {'concat':>8}  {'perbatch_mean':>13}  {'perbatch_std':>12}  {'n_batches':>9}")
    print("-" * 58)
    for bs in args.batch_sizes:
        loader = DataLoader(subset, batch_size=bs, shuffle=True, drop_last=True)
        s = model.evaluate(loader, device)
        if s is None:
            print(f"{bs:>6}  (empty)")
            continue
        n_batches = n_sub // bs
        print(f"{bs:>6}  {s['concat']:>8.4f}  {s['perbatch_mean']:>13.4f}  "
              f"{s['perbatch_std']:>12.4f}  {n_batches:>9}")

    print("\nRead: if perbatch_mean climbs toward ~4.79 at small batch while concat "
          "stays ~4.67,\n      the gap to the reference is the (biased) per-batch "
          "estimator, not the model.")


if __name__ == '__main__':
    main()
