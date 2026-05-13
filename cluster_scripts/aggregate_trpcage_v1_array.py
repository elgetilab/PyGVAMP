#!/usr/bin/env python3
"""
Aggregate the Trp-cage v1 10-seed array into cross-seed statistics for
both scoring methodologies.

For each seed in seed_00..seed_09, parses the training log to extract:
  1. Best concat (the saved model's score)
  2. The epoch at which best concat was reached
  3. The perbatch_mean ± perbatch_std at THAT exact epoch (the
     paper-comparable number for that seed)

Then computes cross-seed mean ± stdev for both metrics, the natural
statistics for comparison against Ghorbani 2022's reported
"VAMP-2 = 4.79 ± 0.01 averaged across 10 different trainings" (Table S1).

Usage:
  python cluster_scripts/aggregate_trpcage_v1_array.py
  python cluster_scripts/aggregate_trpcage_v1_array.py --root /mnt/hdd/experiments/trpcage_repro_v1
  python cluster_scripts/aggregate_trpcage_v1_array.py --csv summary.csv
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Optional


# --- Log parsing -----------------------------------------------------------

# Matches: "Epoch 68/100, Train VAMP: 3.7169, Val VAMP: concat=3.7293, perbatch=3.6710±0.2151"
EPOCH_LINE = re.compile(
    r"Epoch\s+(\d+)/(\d+),\s*Train VAMP:\s*([0-9.]+),\s*"
    r"Val VAMP:\s*concat=([0-9.]+),\s*perbatch=([0-9.]+)\s*±\s*([0-9.]+)"
)
NEW_BEST = re.compile(r"New best model with score:\s*([0-9.]+)")
LOADED_BEST = re.compile(r"Loaded best model with score:\s*([0-9.]+)")
# Fallback completion marker — some runs miss the "Loaded best" line at
# the end of the experiment-level log (occasional buffer flush quirk;
# the saved best_model.pt still exists and the data is intact).
TRAINING_DONE = re.compile(r"Training completed successfully")


@dataclass
class SeedResult:
    seed: int
    log_path: str
    best_concat: float
    best_epoch: int
    perbatch_mean: float
    perbatch_std: float
    final_epoch: int
    completed: bool


def latest_log(seed_dir: Path) -> Optional[Path]:
    """Find the most recent training log under a seed directory.

    Layout:
      seed_dir/exp_trpcage_<TIMESTAMP>/logs/log_<TIMESTAMP>.txt
    """
    candidates = sorted(seed_dir.glob("exp_trpcage_*/logs/log_*.txt"))
    return candidates[-1] if candidates else None


def parse_log(log_path: Path, seed: int) -> Optional[SeedResult]:
    with log_path.open() as f:
        lines = f.readlines()

    epoch_data: dict[int, tuple[float, float, float]] = {}
    final_epoch = 0
    for line in lines:
        m = EPOCH_LINE.search(line)
        if m:
            epoch = int(m.group(1))
            concat = float(m.group(4))
            pm = float(m.group(5))
            ps = float(m.group(6))
            epoch_data[epoch] = (concat, pm, ps)
            final_epoch = max(final_epoch, epoch)

    if not epoch_data:
        return None

    completed = False
    best_concat = None
    training_done = False
    for line in reversed(lines):
        if not completed and LOADED_BEST.search(line):
            best_concat = float(LOADED_BEST.search(line).group(1))
            completed = True
            break
        if not training_done and TRAINING_DONE.search(line):
            training_done = True
    if best_concat is None:
        best_concat = max(c for c, _, _ in epoch_data.values())
        if training_done:
            completed = True

    best_epoch = None
    for epoch in sorted(epoch_data.keys()):
        c, _, _ = epoch_data[epoch]
        if abs(c - best_concat) < 1e-4:
            best_epoch = epoch
            break

    if best_epoch is None:
        best_epoch = max(epoch_data.keys(),
                         key=lambda e: epoch_data[e][0])

    _, pm, ps = epoch_data[best_epoch]

    return SeedResult(
        seed=seed,
        log_path=str(log_path),
        best_concat=best_concat,
        best_epoch=best_epoch,
        perbatch_mean=pm,
        perbatch_std=ps,
        final_epoch=final_epoch,
        completed=completed,
    )


# --- Aggregation ----------------------------------------------------------

def aggregate(root: Path, n_seeds: int = 10) -> list[SeedResult]:
    results: list[SeedResult] = []
    for seed in range(n_seeds):
        seed_dir = root / f"seed_{seed:02d}"
        if not seed_dir.is_dir():
            print(f"  seed_{seed:02d}: directory not found, skipping",
                  file=sys.stderr)
            continue
        log = latest_log(seed_dir)
        if log is None:
            print(f"  seed_{seed:02d}: no training log found, skipping",
                  file=sys.stderr)
            continue
        r = parse_log(log, seed)
        if r is None:
            print(f"  seed_{seed:02d}: log present but no epoch lines parsed, "
                  "skipping", file=sys.stderr)
            continue
        results.append(r)
    return results


def print_table(results: list[SeedResult]) -> None:
    header = (f"{'seed':>4}  {'epoch':>5}  {'best concat':>11}  "
              f"{'perbatch_mean':>13}  {'perbatch_std':>12}  "
              f"{'epochs':>6}  status")
    print(header)
    print("-" * len(header))
    for r in results:
        status = "ok" if r.completed else f"INCOMPLETE (last ep {r.final_epoch})"
        print(f"{r.seed:>4d}  {r.best_epoch:>5d}  {r.best_concat:>11.4f}  "
              f"{r.perbatch_mean:>13.4f}  {r.perbatch_std:>12.4f}  "
              f"{r.final_epoch:>6d}  {status}")


def print_summary(results: list[SeedResult], paper_mean=4.79,
                  paper_sigma=0.01) -> None:
    completed = [r for r in results if r.completed]
    if len(completed) < len(results):
        print(f"\nNote: {len(results) - len(completed)} of {len(results)} "
              "seeds did not reach completion ('Loaded best model' missing).")
    use = completed if completed else results
    if len(use) < 2:
        print(f"\nNot enough seeds for cross-seed statistics "
              f"(need at least 2 completed; have {len(use)}).",
              file=sys.stderr)
        return

    concat_vals = [r.best_concat for r in use]
    pb_vals = [r.perbatch_mean for r in use]

    cm, cs = mean(concat_vals), stdev(concat_vals)
    pm, ps = mean(pb_vals), stdev(pb_vals)

    print(f"\n=== Cross-seed summary (n={len(use)}) ===")
    print(f"  best concat        : {cm:.4f} ± {cs:.4f}")
    print(f"  perbatch_mean      : {pm:.4f} ± {ps:.4f}    "
          f"(paper: {paper_mean} ± {paper_sigma})")

    delta = pm - paper_mean
    sigma_paper = abs(delta) / paper_sigma if paper_sigma > 0 else float('inf')
    sigma_ours = abs(delta) / ps if ps > 0 else float('inf')
    print(f"  Δ vs paper         : {delta:+.4f}  "
          f"({sigma_paper:.1f}σ in paper's σ, "
          f"{sigma_ours:.1f}σ in ours)")

    print(f"\n  best/worst seeds (concat):       "
          f"{max(use, key=lambda r: r.best_concat).seed:>2d}={max(r.best_concat for r in use):.4f}  /  "
          f"{min(use, key=lambda r: r.best_concat).seed:>2d}={min(r.best_concat for r in use):.4f}")
    print(f"  best/worst seeds (perbatch_mean):"
          f" {max(use, key=lambda r: r.perbatch_mean).seed:>2d}={max(r.perbatch_mean for r in use):.4f}  /  "
          f"{min(use, key=lambda r: r.perbatch_mean).seed:>2d}={min(r.perbatch_mean for r in use):.4f}")


def write_csv(results: list[SeedResult], path: Path) -> None:
    with path.open("w") as f:
        f.write("seed,best_epoch,best_concat,perbatch_mean,perbatch_std,"
                "final_epoch,completed,log_path\n")
        for r in results:
            f.write(f"{r.seed},{r.best_epoch},{r.best_concat:.6f},"
                    f"{r.perbatch_mean:.6f},{r.perbatch_std:.6f},"
                    f"{r.final_epoch},{int(r.completed)},{r.log_path}\n")
    print(f"\nCSV written to: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path,
                        default=Path("/mnt/hdd/experiments/trpcage_repro_v1"),
                        help="Parent directory containing seed_00..seed_09 subdirs.")
    parser.add_argument("--n-seeds", type=int, default=10,
                        help="Number of seeds to look for (default 10).")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Optional path to write per-seed CSV.")
    parser.add_argument("--paper-mean", type=float, default=4.79,
                        help="Paper-reported mean for comparison (default 4.79).")
    parser.add_argument("--paper-sigma", type=float, default=0.01,
                        help="Paper-reported sigma for comparison (default 0.01).")
    args = parser.parse_args()

    if not args.root.is_dir():
        print(f"ERROR: root not found: {args.root}", file=sys.stderr)
        sys.exit(1)

    print(f"Aggregating from: {args.root}\n")
    results = aggregate(args.root, n_seeds=args.n_seeds)
    if not results:
        print("No parseable seeds found.", file=sys.stderr)
        sys.exit(1)

    print_table(results)
    print_summary(results, paper_mean=args.paper_mean,
                  paper_sigma=args.paper_sigma)

    if args.csv is not None:
        write_csv(results, args.csv)


if __name__ == "__main__":
    main()
