#!/usr/bin/env python3
"""
Aggregate a RevGraphVAMP (reversible 3-phase) 10-seed array into cross-seed
VAMP-2 and VAMP-E statistics.

Parses each seed's training log for the validation lines emitted by
``RevVAMPNet.fit_three_phase`` (the algebraic-init line and each phase-3 epoch):

    [RevGraphVAMP] algebraic U/S init done (Stage 2)  val VAMP-2=4.1234  VAMP-E=4.0987
      epoch 12: val VAMP-2=4.4051  VAMP-E=4.3760  (best VAMP-2=4.4051)

The reported per-seed number is the (VAMP-2, VAMP-E) pair at the epoch with the
highest val VAMP-2 (the model-selection point — matches fit_three_phase's
best-model rule). Then computes cross-seed mean ± stdev for both metrics, for
comparison with RevGraphVAMP Table 2 (alanine 4.41/4.38; Aβ42 3.99/3.99).

Usage:
  python cluster_scripts/aggregate_reversible_array.py --root /mnt/hdd/experiments/alanine_rev_v1 \
      --paper-vamp2 4.41 --paper-vampe 4.38
  python cluster_scripts/aggregate_reversible_array.py --root /mnt/hdd/experiments/ab42_rev_v1 \
      --paper-vamp2 3.99 --paper-vampe 3.99
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Optional
import re

# Matches both the init line and the per-epoch line (both carry the pair):
#   "... val VAMP-2=4.4051  VAMP-E=4.3760 ..."
VAL_LINE = re.compile(r"val VAMP-2=([0-9.]+)\s+VAMP-E=([0-9.]+)")
# Since 2026-07-24 EVERY phase logs a val line ("  [chi] epoch 3: val VAMP-2=..."),
# but fit_three_phase only does model selection on the algebraic-init line and the
# final 'all' phase.  Scoring chi/us lines here would report a peak that does NOT
# correspond to the saved best_model.pt (e.g. alanine seed 0: chi 4.4946 vs
# all 4.4875).  Only count lines the selector actually considered.
SELECTABLE = re.compile(r"\[all\]\s+epoch|algebraic U/S init done")
# Legacy logs (pre-2026-07-24) had no phase tag: "  epoch 12: val VAMP-2=...".
LEGACY_EPOCH = re.compile(r"^\s*epoch\s+\d+:")
# run_training prints this after train_model returns (→ stdout → log).
TRAINING_DONE = re.compile(r"Training completed successfully")


@dataclass
class SeedResult:
    seed: int
    log_path: str
    best_vamp2: float
    vampe_at_best: float     # VAMP-E at the max-VAMP-2 epoch
    n_val_points: int
    completed: bool


def latest_log(seed_dir: Path) -> Optional[Path]:
    """Most recent training log: seed_dir/exp_*/logs/log_*.txt."""
    candidates = sorted(seed_dir.glob("exp_*/logs/log_*.txt"))
    return candidates[-1] if candidates else None


def parse_log(log_path: Path, seed: int) -> Optional[SeedResult]:
    with log_path.open() as f:
        lines = f.readlines()

    pairs = []  # (vamp2, vampe)
    completed = False
    for line in lines:
        m = VAL_LINE.search(line)
        if m and (SELECTABLE.search(line) or LEGACY_EPOCH.search(line)):
            pairs.append((float(m.group(1)), float(m.group(2))))
        if TRAINING_DONE.search(line):
            completed = True

    if not pairs:
        return None

    best_vamp2, vampe_at_best = max(pairs, key=lambda p: p[0])
    return SeedResult(
        seed=seed, log_path=str(log_path),
        best_vamp2=best_vamp2, vampe_at_best=vampe_at_best,
        n_val_points=len(pairs), completed=completed,
    )


def aggregate(root: Path, n_seeds: int) -> list[SeedResult]:
    results = []
    for seed in range(n_seeds):
        seed_dir = root / f"seed_{seed:02d}"
        if not seed_dir.is_dir():
            print(f"  seed_{seed:02d}: dir not found, skipping", file=sys.stderr)
            continue
        log = latest_log(seed_dir)
        if log is None:
            print(f"  seed_{seed:02d}: no training log, skipping", file=sys.stderr)
            continue
        r = parse_log(log, seed)
        if r is None:
            print(f"  seed_{seed:02d}: no val VAMP lines parsed, skipping",
                  file=sys.stderr)
            continue
        results.append(r)
    return results


def print_table(results: list[SeedResult]) -> None:
    header = f"{'seed':>4}  {'best VAMP-2':>11}  {'VAMP-E@best':>11}  {'val pts':>7}  status"
    print(header)
    print("-" * len(header))
    for r in results:
        status = "ok" if r.completed else "INCOMPLETE"
        print(f"{r.seed:>4d}  {r.best_vamp2:>11.4f}  {r.vampe_at_best:>11.4f}  "
              f"{r.n_val_points:>7d}  {status}")


def _summ(label, vals, paper, sigma_paper):
    m, s = mean(vals), (stdev(vals) if len(vals) > 1 else 0.0)
    delta = m - paper
    sp = abs(delta) / sigma_paper if sigma_paper > 0 else float('inf')
    so = abs(delta) / s if s > 0 else float('inf')
    print(f"  {label:<8}: {m:.4f} ± {s:.4f}   (paper {paper} ± {sigma_paper})   "
          f"Δ={delta:+.4f} ({sp:.1f}σ paper, {so:.1f}σ ours)")


def print_summary(results, paper_vamp2, paper_vampe, sigma=0.01):
    completed = [r for r in results if r.completed]
    if len(completed) < len(results):
        print(f"\nNote: {len(results)-len(completed)}/{len(results)} seeds "
              "not marked completed.")
    use = completed if completed else results
    if len(use) < 2:
        print(f"\nNeed >=2 completed seeds for cross-seed stats; have {len(use)}.",
              file=sys.stderr)
        return
    print(f"\n=== Cross-seed summary (n={len(use)}) ===")
    _summ("VAMP-2", [r.best_vamp2 for r in use], paper_vamp2, sigma)
    _summ("VAMP-E", [r.vampe_at_best for r in use], paper_vampe, sigma)
    print(f"\n  best/worst VAMP-2:  "
          f"{max(use, key=lambda r: r.best_vamp2).seed}={max(r.best_vamp2 for r in use):.4f}  /  "
          f"{min(use, key=lambda r: r.best_vamp2).seed}={min(r.best_vamp2 for r in use):.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True,
                   help="Parent dir with seed_00..seed_NN.")
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--paper-vamp2", type=float, required=True)
    p.add_argument("--paper-vampe", type=float, required=True)
    p.add_argument("--paper-sigma", type=float, default=0.01)
    args = p.parse_args()

    if not args.root.is_dir():
        print(f"ERROR: root not found: {args.root}", file=sys.stderr)
        sys.exit(1)
    print(f"Aggregating from: {args.root}\n")
    results = aggregate(args.root, args.n_seeds)
    if not results:
        print("No parseable seeds found.", file=sys.stderr)
        sys.exit(1)
    print_table(results)
    print_summary(results, args.paper_vamp2, args.paper_vampe, args.paper_sigma)


if __name__ == "__main__":
    main()
