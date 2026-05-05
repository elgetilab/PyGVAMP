#!/bin/bash
# ===========================================================================
# PyGVAMP — Ab42 Reduced: State Discovery Only
# ===========================================================================
# Runs preparation + Graph2Vec + clustering to determine n_states.
# Submit this ONCE, then use the discovered n_states in experiment scripts.
#
# Usage:
#   sbatch ab42_red_discovery.sh
#   # Check result:
#   grep "Recommended n_states" logs/ab42r_disc_*.out
# ===========================================================================

#SBATCH --job-name=ab42r_disc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gputraining
#SBATCH --gres=gpu:batch:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ab42r_disc_%j.out
#SBATCH --error=logs/ab42r_disc_%j.err

# ---- CONFIGURATION --------------------------------------------------------

PROTEIN_NAME="ab42_red"
TOPOLOGY="/mnt/hdd/data/ab42/trajectories/red/topol.pdb"
TRAJ_DIR="/mnt/hdd/data/ab42/trajectories/red/"
SELECTION="name CA"
STRIDE=1
OUTPUT_DIR="/mnt/hdd/experiments/${PROTEIN_NAME}/discovery"

# ---- Environment setup -----------------------------------------------------
module purge
source /etc/profile.d/modules.sh
module load 12.8
module load pygvamp/1.0.0

mkdir -p logs

# ---- Run preparation only (skip training) ----------------------------------
echo "============================================================"
echo "PyGVAMP — Ab42 Reduced, State Discovery"
echo "============================================================"
echo "Node:         $(hostname)"
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start:        $(date)"
echo "============================================================"

pygvamp \
    --traj_dir "${TRAJ_DIR}" \
    --top "${TOPOLOGY}" \
    --lag_times 1 \
    --protein_name "${PROTEIN_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --preset medium_schnet \
    --stride "${STRIDE}" \
    --selection "${SELECTION}" \
    --cache \
    --skip_training

EXIT_CODE=$?

echo "============================================================"
echo "Finished:     $(date)"
echo "Exit code:    ${EXIT_CODE}"
echo ""
echo "To use the result in experiment scripts, add:"
echo "  --no_discover_states --n_states <N>"
echo "============================================================"

exit ${EXIT_CODE}
