#!/bin/bash
# Build the git-sized results bundle (experiments/runs/) from the raw run
# outputs in /mnt/hdd/experiments.
#
# Kept verbatim: every *.json, *.csv, *.yaml and *.txt under each run dir
#   (pipeline/analysis summaries, ITS summaries, transition matrices, state
#   diagnostics, configs, metadata).
# Kept as digests: per-run training logs (logs/log_*.txt) and SLURM job logs
#   (logs/*.out, logs/*.err).  A digest drops tqdm progress bars, per-batch
#   "Batch i/n, VAMP" lines, mid-epoch "Quick validation" lines and blank
#   lines; keeps only the first "Lag time ... corresponds to" and "Loaded N
#   trajectories" line; and collapses consecutive identical lines to one line
#   suffixed "[repeated N times]".  Per-epoch Train/Val VAMP lines, best
#   scores, warnings (incl. NaN warnings), errors and tracebacks survive.
# Dropped: models/checkpoints (*.pt), *.pkl, *.npz, *.npy, plots, html, pdb.
#
# Usage: build_results_bundle.sh [SRC] [OUT]
set -euo pipefail
SRC=${1:-/mnt/hdd/experiments}
OUT=${2:-$(cd "$(dirname "$0")/.." && pwd)/runs}
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

digest() {
  tr '\r' '\n' < "$1" \
  | grep -avE 'it/s|s/it|^Batch [0-9]+/[0-9]+, VAMP|^\s*$|[0-9]+%\|.*\| *[0-9]+/[0-9]+|^Quick validation \(' \
  | awk '/^(Lag time .* corresponds to|Loaded [0-9]+ trajectories with frames)/ { if (seen[$0]++) next } { print }' \
  | awk 'function flush() { if (n > 1) print prev "  [repeated " n " times]"; else if (n == 1) print prev }
         { if ($0 == prev) n++; else { flush(); prev = $0; n = 1 } } END { flush() }' \
  || true
}

mkdir -p "$OUT"
cd "$SRC"
find . -path ./logs -prune -o -type f \( -name '*.json' -o -name '*.csv' -o -name '*.yaml' -o -name '*.txt' \) -print \
  | sed 's#^\./##' > "$TMP/all"
grep -vE '/logs/log_[0-9_]+\.txt$' "$TMP/all" > "$TMP/verbatim" || true
grep -E  '/logs/log_[0-9_]+\.txt$' "$TMP/all" > "$TMP/runlogs"  || true

rsync -a --files-from="$TMP/verbatim" "$SRC/" "$OUT/"

while read -r f; do
  mkdir -p "$OUT/$(dirname "$f")"
  digest "$f" > "$OUT/${f%.txt}.digest.txt"
done < "$TMP/runlogs"

mkdir -p "$OUT/logs"
for f in logs/*; do
  [ -f "$f" ] && digest "$f" > "$OUT/logs/$(basename "$f").digest"
done
echo "bundle written to $OUT"
