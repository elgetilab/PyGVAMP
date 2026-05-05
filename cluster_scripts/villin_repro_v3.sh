#!/bin/bash
# ===========================================================================
# PyGVAMP — Villin reproduction v3 (single-seed probe)
# ===========================================================================
# Probe whether two more deviations from Ghorbani 2022 close the residual
# ~0.12 gap left by v2:
#
#   1. Plain Adam (paper) instead of AdamW + weight_decay=1e-5.
#      Achieved by --weight_decay 0: with WD=0, AdamW is mathematically
#      identical to torch.optim.Adam (the decoupled-WD term vanishes).
#      No optimizer-constructor swap needed.
#
#   2. Disable training-time input jitter (--training_jitter 0.0).
#      Default is 1e-6 N(0,σ) noise added to node features every forward
#      pass during training; not in the paper.  Flag now plumbed through
#      args so we can keep the feature available for later runs.
#
# Single seed (seed 0) — sanity check before committing to a 10-seed sweep.
# v1 seed_00 best=3.5685, v2 seed_00 best=3.6057.  Threshold for "lever
# worked": v3 seed_00 ≳ 3.65.  If it lands ~3.70+, full v3 array is worth
# the wall-clock.
#
# Note: --training_jitter requires the args.py addition committed alongside
# this run.  Module rebuild needed before submission.
#
# Submit (no --array; single job):
#   sbatch cluster_scripts/villin_repro_v3.sh
#
# Timestep gotcha: DE Shaw DCD metadata reports 1 ps/frame but the actual
# physical timestep is 200 ps/frame.  --timestep 0.2 is MANDATORY.
# ===========================================================================

#SBATCH --job-name=villin_v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gputraining
#SBATCH --gres=gpu:batch:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=6:00:00
#SBATCH --output=/mnt/hdd/experiments/logs/villin_repro_v3_%j.out
#SBATCH --error=/mnt/hdd/experiments/logs/villin_repro_v3_%j.err

# ---- Environment setup -----------------------------------------------------
module purge
source /etc/profile.d/modules.sh
module load 12.8
module load pygvamp/1.0.0

mkdir -p /mnt/hdd/experiments/logs

SEED=0
RUN_DIR=$(printf "/mnt/hdd/experiments/villin_repro_v3/seed_%02d" "${SEED}")

# ---- Job info -------------------------------------------------------------
echo "============================================================"
echo "Villin reproduction v3 (single seed probe)"
echo "============================================================"
echo "Job:        ${SLURM_JOB_ID}    Seed: ${SEED}    Output: ${RUN_DIR}"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Node:       $(hostname)"
echo "Start:      $(date)"
echo "Target:     VAMP-2 = 3.78 ± 0.02 (Ghorbani 2022, Table S1)"
echo "Diffs vs v2: weight_decay=0 (plain Adam), training_jitter=0.0"
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
    --no_use_attention \
    --no_use_embedding \
    --clf_num_layers 1 \
    --clf_dropout    0 \
    --clf_norm       none \
    --lr             5e-4 \
    --weight_decay   0 \
    --training_jitter 0.0 \
    --epochs       100 \
    --batch_size   1000 \
    --val_split    0.3 \
    --cache

EXIT_CODE=$?

echo "============================================================"
echo "Finished:   $(date)    Exit: ${EXIT_CODE}"
echo "============================================================"

exit ${EXIT_CODE}
