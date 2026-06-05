#!/usr/bin/env python3
"""
Cross-seed aggregation of NTL9 v2 *analysis* output (state populations, ITS,
CK test) with confidence intervals.

This is a thin driver that reuses the publication aggregator in
``for_publication/paper_analysis.py`` — it does the hard part (matching state
labels across seeds via the Hungarian algorithm, then mean ± CI for
populations / ITS / CK). We only need it because that script's
``discover_runs`` knows the older ``lag*ns/run_*`` layout, whereas the repro
arrays use ``seed_NN/exp_*/analysis/<lag>states/``. We build the analysis-dir
list from the seed layout and hand it to ``analyze_lag_group`` directly.

Caveat (applies to every system this pipeline analyzes, not just NTL9):
analysis runs over the 50k-frame random subsample (analysis_max_frames), so
state POPULATIONS are unbiased (mean over a uniform sample), but ITS/CK treat
consecutive subsampled rows as consecutive frames — the subsample is
time-ordered but not contiguous, so absolute ITS/CK values carry that
sampling caveat. Reported here for consistency with the per-seed analysis.

Usage:
    MPLBACKEND=Agg python cluster_scripts/aggregate_ntl9_v2_analysis.py
    MPLBACKEND=Agg python cluster_scripts/aggregate_ntl9_v2_analysis.py \
        --root /mnt/hdd/experiments/ntl9_repro_v2
"""
import os
import sys
import argparse
from glob import glob
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "for_publication"))
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import paper_analysis as pa  # noqa: E402

# paper_analysis.load_run_data parses the transition-matrix CSV with
# np.loadtxt, but the pipeline writes it pandas-style (header row + row
# labels), which np.loadtxt can't read. That field is loaded but never used
# by analyze_lag_group, so we install a tolerant shim (header/label-aware)
# rather than touch the committed publication script.
_orig_loadtxt = np.loadtxt
def _tolerant_loadtxt(fname, **kw):
    try:
        return _orig_loadtxt(fname, **kw)
    except ValueError:
        delim = kw.get("delimiter", ",")
        return np.genfromtxt(fname, delimiter=delim, skip_header=1)[:, 1:]
pa.np.loadtxt = _tolerant_loadtxt


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/mnt/hdd/experiments/ntl9_repro_v2",
                    help="Parent dir containing seed_NN/exp_*/analysis/...")
    ap.add_argument("--protein", default="ntl9")
    ap.add_argument("--lag_subdir", default="lag200.0ns_5states",
                    help="Analysis subdir name under each seed's analysis/.")
    ap.add_argument("--lag_str", default="lag200.0ns",
                    help="Lag label parsed by analyze_lag_group (lag<X>ns).")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--timestep", type=float, default=0.2,
                    help="ns per frame (NTL9 DESRES = 0.2).")
    ap.add_argument("--output_dir", default=None,
                    help="Default: <root>/cross_seed_analysis")
    args = ap.parse_args()

    pattern = os.path.join(args.root, "seed_*", "exp_*", "analysis", args.lag_subdir)
    analysis_dirs = sorted(glob(pattern))
    if not analysis_dirs:
        sys.exit(f"No analysis dirs match {pattern}")

    print(f"Found {len(analysis_dirs)} seed analysis dirs:")
    for d in analysis_dirs:
        print(f"  {d}")

    output_dir = args.output_dir or os.path.join(args.root, "cross_seed_analysis")
    os.makedirs(output_dir, exist_ok=True)

    pa.analyze_lag_group(
        lag_str=args.lag_str,
        analysis_dirs=analysis_dirs,
        output_dir=output_dir,
        protein_name=args.protein,
        stride=args.stride,
        timestep=args.timestep,
    )
    print(f"\nFigures + {args.protein}_{args.lag_str}_results.npz saved to: {output_dir}")


if __name__ == "__main__":
    main()
