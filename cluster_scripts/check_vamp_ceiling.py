#!/usr/bin/env python
"""Check reversible-run val VAMP-2 traces against the theoretical ceiling.

VAMP-2 = 1 + sum_{i=2..k} sigma_i^2 with every sigma_i <= 1, so a k-state model
CANNOT score above k. `aggregate_reversible_array.py` selects each seed at its
max val VAMP-2 over epochs; if the estimator spikes, that rule selects the spike.
This script reports, per seed, the selected (max) value next to the converged
value, and counts how many selection-eligible epochs breach the ceiling.

Usage:
  python cluster_scripts/check_vamp_ceiling.py --k 4 \
      --logs '/mnt/hdd/experiments/logs/ab42_rev_838_*.out'
"""
import argparse
import glob
import re
import statistics as st
from pathlib import Path

EPOCH = re.compile(
    r"\[(chi|us|all)\] epoch (\d+): val VAMP-2=([0-9.]+)\s+VAMP-E=([0-9.]+)")
INIT = re.compile(
    r"algebraic U/S init done .*?val VAMP-2=([0-9.]+)\s+VAMP-E=([0-9.]+)")
SEED = re.compile(r"_(\d+)\.out$")


def analyse(path: Path, k: int, tail: int):
    txt = path.read_text(errors="ignore")
    # Selection-eligible points only: the algebraic init + the 'all' phase,
    # matching aggregate_reversible_array.py's model-selection rule.
    pts = [float(m.group(3)) for m in EPOCH.finditer(txt) if m.group(1) == "all"]
    init = INIT.search(txt)
    if init:
        pts.insert(0, float(init.group(1)))
    if not pts:
        return None
    over = [v for v in pts if v > k]
    return dict(
        n=len(pts),
        selected=max(pts),
        converged=st.median(pts[-tail:]),
        n_over=len(over),
        worst=max(over) if over else 0.0,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help="glob for the array .out files")
    ap.add_argument("--k", type=int, required=True, help="n_states (= the ceiling)")
    ap.add_argument("--tail", type=int, default=10,
                    help="epochs at the end to median for the converged value")
    ap.add_argument("--paper", type=float, default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(args.logs), key=lambda p: int(SEED.search(p).group(1))
                   if SEED.search(p) else 0)
    if not files:
        raise SystemExit(f"no logs matched {args.logs}")

    rows = []
    print(f"ceiling for k={args.k} is VAMP-2 <= {args.k}.0\n")
    print(f"{'seed':>4} {'n_pts':>6} {'selected':>9} {'converged':>10} "
          f"{'#>ceil':>7} {'worst':>8} {'excess':>8}")
    for f in files:
        p = Path(f)
        m = SEED.search(p.name)
        seed = int(m.group(1)) if m else -1
        r = analyse(p, args.k, args.tail)
        if r is None:
            print(f"{seed:>4}   (no validation lines parsed)")
            continue
        r["seed"] = seed
        rows.append(r)
        excess = max(0.0, r["selected"] - args.k)
        print(f"{seed:>4} {r['n']:>6} {r['selected']:>9.4f} {r['converged']:>10.4f} "
              f"{r['n_over']:>7} {r['worst']:>8.4f} {excess:>8.4f}")

    if not rows:
        return
    sel = [r["selected"] for r in rows]
    con = [r["converged"] for r in rows]
    n_pts = sum(r["n"] for r in rows)
    n_over = sum(r["n_over"] for r in rows)

    def pm(xs):
        return (f"{st.mean(xs):.4f} +- {st.stdev(xs):.4f}" if len(xs) > 1
                else f"{xs[0]:.4f}")

    print(f"\n=== cross-seed (n={len(rows)}) ===")
    print(f"  selected (max-over-epoch) : {pm(sel)}")
    print(f"  converged (last-{args.tail} median): {pm(con)}")
    if args.paper is not None:
        print(f"  paper                     : {args.paper}")
        print(f"  delta (converged - paper) : {st.mean(con) - args.paper:+.4f}")
    print(f"  epochs above ceiling      : {n_over}/{n_pts} "
          f"across {sum(1 for r in rows if r['n_over'])}/{len(rows)} seeds")
    verdict = ("INVALID - selection breaches the ceiling" if n_over
               else "clean - no epoch breaches the ceiling")
    print(f"  verdict                   : {verdict}")


if __name__ == "__main__":
    main()
