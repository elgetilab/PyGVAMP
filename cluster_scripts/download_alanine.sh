#!/bin/bash
# ===========================================================================
# Download the alanine dipeptide dataset for the RevGraphVAMP reproduction.
# ===========================================================================
# The standard mdshare/PyEMMA alanine dipeptide set: 3 independent 250 ns
# trajectories at 1 ps/frame (750k frames total), "nowater", 22 atoms / 10 heavy.
# This is the dataset RevGraphVAMP (Huang 2024) uses for its alanine benchmark
# (k=6, lag=20 ps, target VAMP-2 4.41 / VAMP-E 4.38).
#
# mdshare is not installed in the deployed module env, so we pull the files
# directly from the Freie Universität Berlin CMB data mirror (same source
# mdshare uses). No env changes.
#
# Usage:  bash cluster_scripts/download_alanine.sh
# ===========================================================================
set -euo pipefail

DEST=/mnt/hdd/data/alanine
BASE=http://ftp.imp.fu-berlin.de/pub/cmb-data

FILES=(
    alanine-dipeptide-nowater.pdb
    alanine-dipeptide-0-250ns-nowater.xtc
    alanine-dipeptide-1-250ns-nowater.xtc
    alanine-dipeptide-2-250ns-nowater.xtc
)

mkdir -p "$DEST"
for f in "${FILES[@]}"; do
    if [ -s "$DEST/$f" ]; then
        echo "already have $f ($(du -h "$DEST/$f" | cut -f1))"
    else
        echo "downloading $f ..."
        curl -sSf -o "$DEST/$f" "$BASE/$f"
        echo "  -> $(du -h "$DEST/$f" | cut -f1)"
    fi
done
echo "Alanine dipeptide data ready in $DEST"
