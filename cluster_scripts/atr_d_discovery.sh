#!/bin/bash
# ===========================================================================
# PyGVAMP — AT1R (d125d) State Discovery
# ===========================================================================
# Discovery-only job for the AT1R / d125d system. Runs preparation +
# Graph2Vec + clustering over the 40 d125d trajectories to recommend an
# n_states. Run this ONCE, read the recommended n_states (and confirm the
# selected-atom count is the receptor only), then submit atr_d_experiment.sh.
#
# Data:      /mnt/hdd/data/julia_ATR   (recursive, d125d/cutted_dt_1ns.xtc -> 40 files)
# Topology:  .../gmm0/r1/d125d/prot_chains.pdb  (chain A = receptor, chain B = 7-mer peptide)
# Selection: "chainid 0 and name CA"   (receptor CA only; peptide chain excluded)
#
# Usage:
#   sbatch cluster_scripts/atr_d_discovery.sh
#   # then, from the log:
#   grep "Recommended n_states" /mnt/hdd/experiments/logs/disc_<jobid>.out
#   grep "Selected"             /mnt/hdd/experiments/logs/disc_<jobid>.out   # ~319 CA expected
# ===========================================================================

# ---- SLURM directives ------------------------------------------------------
#SBATCH --job-name=atr_d_disc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gputraining
#SBATCH --gres=gpu:batch:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128000M
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/hdd/experiments/logs/disc_%j.out
#SBATCH --error=/mnt/hdd/experiments/logs/disc_%j.err

# ---- Environment setup -----------------------------------------------------
module purge
source /etc/profile.d/modules.sh
module load cuda/12.8
module load pygvamp/1.0.0

mkdir -p /mnt/hdd/experiments/logs

# ---- Hardcoded ATR / d125d parameters --------------------------------------
PROTEIN_NAME="atr_d"
TRAJ_DIR="/mnt/hdd/data/julia_ATR"
TOPOLOGY="/mnt/hdd/data/julia_ATR/gmm0/r1/d125d/prot_chains.pdb"
FILE_PATTERN="d125d/cutted_dt_1ns.xtc"
SELECTION="chainid 0 and name CA"
PRESET="large_schnet"
OUTPUT_DIR="/mnt/hdd/experiments/atr_d/discovery"

scontrol update JobId=${SLURM_JOB_ID} Name=atr_d_disc 2>/dev/null

# ---- Print job info --------------------------------------------------------
echo "============================================================"
echo "PyGVAMP — AT1R (d125d) State Discovery"
echo "============================================================"
echo "Trajectories: ${TRAJ_DIR}  (pattern: ${FILE_PATTERN}, recursive)"
echo "Topology:     ${TOPOLOGY}"
echo "Selection:    ${SELECTION}"
echo "Preset:       ${PRESET}"
echo "Output:       ${OUTPUT_DIR}"
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Node:         $(hostname)"
echo "Start:        $(date)"
echo "============================================================"

# State discovery runs by default in the preparation phase; --skip_training
# stops before any VAMPNet training.
pygvamp \
    --traj_dir "${TRAJ_DIR}" \
    --top "${TOPOLOGY}" \
    --file_pattern "${FILE_PATTERN}" \
    --selection "${SELECTION}" \
    --lag_times 1 \
    --protein_name "${PROTEIN_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --preset "${PRESET}" \
    --n_neighbors 20 \
    --stride 1 \
    --cache \
    --skip_training

EXIT_CODE=$?

echo "============================================================"
echo "Finished:     $(date)"
echo "Exit code:    ${EXIT_CODE}"
echo ""
echo "Next steps:"
echo "  grep 'Recommended n_states' /mnt/hdd/experiments/logs/disc_${SLURM_JOB_ID}.out"
echo "  grep 'Selected'             /mnt/hdd/experiments/logs/disc_${SLURM_JOB_ID}.out"
echo "  sbatch cluster_scripts/atr_d_experiment.sh --n_states <N>"
echo "============================================================"

exit ${EXIT_CODE}
