#!/bin/bash
# ===========================================================================
# PyGVAMP — Villin reproduction v9 (lag-time off-by-10 hypothesis probe)
# ===========================================================================
# Tests an alternative explanation for the residual ~0.10 VAMP-2 gap to the
# Ghorbani 2022 paper: that *the paper itself* has an off-by-10 error in
# the lag-time reported for villin (or in the trajectory timestep used to
# convert reported-lag → frame-lag).
#
# Single change vs v4: --lag_times 2.0 (instead of 20.0).
#
# WHY THIS HYPOTHESIS:
#   - The user observed similar confusion with ab42, where training at
#     τ=2 ns gave better results than τ=20 ns.
#   - DESRES DCD trajectories have a known timestep mismatch: file
#     metadata reports 1 ps/frame but physical timestep is 200 ps/frame
#     (200× factor).  An off-by-10 in the lag conversion is a similar
#     class of error.
#   - If the paper assumed 1 ps/frame metadata as truth (instead of the
#     200 ps physical step) and reported "lag = 200 ns", their actual
#     trained-on lag in physical units could be 200/100 = 2 ns — which
#     would propagate as "we should use 20 ns" in someone else's
#     reproduction (off by 10×).
#   - Ground truth: at lag = 2 ns the eigenvalue spectrum of the
#     transition matrix is much closer to identity (slow modes haven't
#     decorrelated as much), so VAMP-2 ≈ Σ λ_i² is mechanically higher.
#     A higher score at lag=2 ns is EXPECTED — the question is whether
#     it's higher by a margin that suggests the paper's 3.78 came from
#     this regime, not from the τ=20 ns regime they reported.
#
# Decision rule (vs v4 baseline 3.7126 at τ=20 ns):
#   - v9 ~ 3.7  → lag-time isn't the explanation; effective time
#                 resolution doesn't matter at this level
#   - v9 ~ 3.78 → strong evidence the paper's reported number actually
#                 came from a τ=2 ns run mislabeled as τ=20 ns
#   - v9 >> 3.78 → at τ=2 ns the score saturates because all 4 metastable
#                  states still look identical (lag too short to resolve
#                  slow modes); the "good" number reflects shallow
#                  separability, not real timescales
#
# The v9 result is interpretable ONLY in concert with the implied
# timescale (ITS) plot from the same model — the question is whether
# 4 well-separated states are actually present at τ=2 ns or whether
# the high VAMP-2 reflects sub-equilibrium dynamics.
#
# ONE EXPLORATORY RUN — single seed, no array.  If the hypothesis is
# supported, we'd follow up with an ITS analysis at multiple τ before
# committing to a sweep.
#
# Submit:
#   sbatch cluster_scripts/villin_repro_v9.sh
#
# No module rebuild needed — uses existing --lag_times flag.
#
# Timestep gotcha: DESRES DCD metadata reports 1 ps/frame but the actual
# physical timestep is 200 ps/frame.  --timestep 0.2 (ns) is MANDATORY
# regardless of lag-time choice.  v9 keeps --timestep 0.2 unchanged;
# only --lag_times changes.
# ===========================================================================

#SBATCH --job-name=villin_v9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gputraining
#SBATCH --gres=gpu:batch:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=6:00:00
#SBATCH --output=/mnt/hdd/experiments/logs/villin_repro_v9_%j.out
#SBATCH --error=/mnt/hdd/experiments/logs/villin_repro_v9_%j.err

# ---- Environment setup -----------------------------------------------------
module purge
source /etc/profile.d/modules.sh
module load cuda/12.8
module load pygvamp/1.0.0

mkdir -p /mnt/hdd/experiments/logs

SEED=0
RUN_DIR=$(printf "/mnt/hdd/experiments/villin_repro_v9/seed_%02d" "${SEED}")

# ---- Job info -------------------------------------------------------------
echo "============================================================"
echo "Villin reproduction v9 (lag-time off-by-10 probe)"
echo "============================================================"
echo "Job:        ${SLURM_JOB_ID}    Seed: ${SEED}    Output: ${RUN_DIR}"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Node:       $(hostname)"
echo "Start:      $(date)"
echo "Target:     VAMP-2 = 3.78 ± 0.02 (Ghorbani 2022, Table S1)"
echo "Diff vs v4: --lag_times 2.0 (was 20.0) — testing off-by-10 hypothesis"
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
    --lag_times    2.0 \
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
