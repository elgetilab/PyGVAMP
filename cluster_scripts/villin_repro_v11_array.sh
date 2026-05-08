#!/bin/bash
# ===========================================================================
# PyGVAMP — Villin reproduction v11 array (10-seed, dual-scoring baseline)
# ===========================================================================
# Cross-seed sweep of the v11 architecture (corrected attention baseline +
# dual-scoring evaluate), after the v11 single-seed probe (seed 0 →
# best concat 3.7293 / perbatch 3.6710 at epoch 68) confirmed the
# pipeline works end-to-end.
#
# WHY 10 SEEDS:
#
# Ghorbani 2022 reports VAMP-2 = 3.78 ± 0.02 averaged across 10
# different trainings.  The ± 0.02 is *cross-seed variability*, not
# within-batch.  v11 single-seed perbatch = 3.6710 cannot be directly
# compared to 3.78 ± 0.02 — those are different statistics.  This array
# produces the matching statistic: cross-seed mean ± stdev of
# `perbatch_mean` at the best-concat epoch.
#
# The array also yields the same statistic for `concat` (our default
# unbiased scoring), letting us report both:
#
#   <concat>_seeds         ± stdev_seeds   (our methodology)
#   <perbatch_mean>_seeds  ± stdev_seeds   (paper's methodology — direct
#                                           comparison to 3.78 ± 0.02)
#
# All settings match cluster_scripts/villin_repro_v11.sh except --seed
# comes from SLURM_ARRAY_TASK_ID (0..9) instead of being pinned to 0.
#
# Per-task wall time: ~3.5-4 h on RTX 5090.  Full array (throttled to
# one GPU at a time): ~35-40 h.  Adjust throttle (`%1`) if more GPUs
# are free.
#
# Submit:
#   sbatch --array=0-9%1 cluster_scripts/villin_repro_v11_array.sh
#
# Aggregate after completion (sketch):
#   for s in 0..9: read pipeline_summary.json best_score (concat)
#                  read log_*.txt for the perbatch=X.XXXX±Y.YYYY at
#                       the same epoch as best_score
#   compute mean and stdev across the 10 seeds for each metric
#
# Timestep gotcha: DESRES DCD metadata reports 1 ps/frame but the actual
# physical timestep is 200 ps/frame.  --timestep 0.2 is MANDATORY.
# ===========================================================================

#SBATCH --job-name=villin_v11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gputraining
#SBATCH --gres=gpu:batch:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=6:00:00
#SBATCH --output=/mnt/hdd/experiments/logs/villin_repro_v11_%A_%a.out
#SBATCH --error=/mnt/hdd/experiments/logs/villin_repro_v11_%A_%a.err

# ---- Environment setup -----------------------------------------------------
module purge
source /etc/profile.d/modules.sh
module load 12.8
module load pygvamp/1.0.0

mkdir -p /mnt/hdd/experiments/logs

if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then
    echo "ERROR: submit as an array job, e.g. sbatch --array=0-9%1 $0"
    exit 1
fi

SEED=${SLURM_ARRAY_TASK_ID}
RUN_DIR=$(printf "/mnt/hdd/experiments/villin_repro_v11/seed_%02d" "${SEED}")

JOB_NAME="villin_v11_seed${SEED}"
scontrol update JobId=${SLURM_JOB_ID} Name=${JOB_NAME} 2>/dev/null

# ---- Job info -------------------------------------------------------------
echo "============================================================"
echo "Villin reproduction v11 array (task ${SLURM_ARRAY_TASK_ID})"
echo "============================================================"
echo "Job:        ${SLURM_JOB_ID}    Seed: ${SEED}    Output: ${RUN_DIR}"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Node:       $(hostname)"
echo "Start:      $(date)"
echo "Target:     VAMP-2 = 3.78 ± 0.02 (Ghorbani 2022, Table I)"
echo "Diff vs v4: --use_attention (corrected baseline) + dual-scoring eval"
echo "============================================================"

# ---- Run -------------------------------------------------------------------
pygvamp \
    --traj_dir /mnt/hdd/data/villin/DESRES-Trajectory_2F4K-0-c-alpha/2F4K-0-c-alpha/ \
    --top      /mnt/hdd/data/villin/DESRES-Trajectory_2F4K-0-c-alpha/topol.pdb \
    --file_pattern '2F4K-0-c-alpha-*.dcd' \
    --protein_name villin \
    --output_dir   "${RUN_DIR}" \
    --timestep     0.2 \
    --seed         "${SEED}" \
    --model        schnet \
    --selection    'name CA' \
    --stride       1 \
    --lag_times    20.0 \
    --n_states     4 \
    --no_discover_states \
    --max_retrains 0 \
    --no_warm_start_retrains \
    --hidden_dim            16 \
    --output_dim            16 \
    --n_interactions        4 \
    --n_neighbors           10 \
    --gaussian_expansion_dim 16 \
    --use_attention \
    --no_use_embedding \
    --clf_num_layers 1 \
    --clf_dropout    0 \
    --clf_norm       none \
    --init_method    xavier_normal \
    --lr           5e-4 \
    --weight_decay 1e-5 \
    --epochs       100 \
    --batch_size   1000 \
    --val_split    0.3 \
    --cache

EXIT_CODE=$?

echo "============================================================"
echo "Finished:   $(date)    Exit: ${EXIT_CODE}"
echo "============================================================"

exit ${EXIT_CODE}