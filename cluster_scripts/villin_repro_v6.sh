#!/bin/bash
# ===========================================================================
# PyGVAMP — Villin reproduction v6 (RBF range fix probe)
# ===========================================================================
# Probe whether pinning the Gaussian RBF basis range to the Ghorbani 2022
# reference's hardcoded values closes more of the residual gap to VAMP-2 = 3.78.
#
# Single change vs v4: --distance_min 0.0 --distance_max 3.0 (nm).
#
#   Audit (tests/test_dataset.py:TestRBFAgainstGhorbaniReference) confirmed:
#   - With dmin=0, dmax=3, K=16, our linspace centers exactly match the
#     reference's arange(0, 3.2, 0.2) centers — 16 evenly spaced at 0.2 nm.
#   - The remaining numerical deviation is the σ formula:
#       ours: σ = (dmax-dmin)/K       = 3/16 = 0.1875
#       ref:  σ = step                = 0.2          (ratio 15/16)
#     → ours' Gaussians are ~6.7% narrower; can't be fixed via current CLI.
#
#   Default behavior (v1–v5) used per-protein data-derived [dmin, dmax].
#   For villin Cα-only that's roughly [0.38, ~3.0] nm — different basis
#   range than the reference, even though σ ends up close by coincidence
#   ((dmax-dmin)/K ≈ 0.16 vs ref 0.2).
#
# v6 sets the basis range to the reference's fixed values. The first ~2
# centers (μ=0.0, 0.2) lie below the physical Cα-Cα minimum (~0.38 nm) and
# act as near-zero "dead" features, mirroring the reference's behavior.
#
# Single seed (seed 0) — sanity probe before a 10-seed sweep.
#   v1 seed_00 best = 3.5685
#   v2 seed_00 best = 3.6057
#   v3 seed_00 best = 3.6124  (plain Adam + no jitter — ineffective)
#   v4 seed_00 best = 3.7126  (xavier_normal init — kept)
#   v5 seed_00 best = 3.7074  (encoder v2 per-atom ReLU — ineffective, dropped)
# Decision rule (mirrors v5):
#   v6 seed_00 ≲ 3.72 → RBF range fix doesn't help, abandon
#   v6 seed_00 ~3.72-3.76 → marginal; consider 3 seeds before committing
#   v6 seed_00 ≳ 3.76 → strong → proceed to 10-seed v6 array
#
# Note: --distance_min and --distance_max require the args.py +
# base_config.py + master_pipeline.py + preparation.py + training.py +
# analysis.py changes committed alongside this run. Module rebuild needed
# before submission.
#
# v6 is built on the v4 baseline (no encoder v2): xavier_normal init +
# v1 (no per-atom ReLU) + RBF range pinned. v5's per-atom ReLU was dropped.
#
# Submit (no --array; single job):
#   sbatch cluster_scripts/villin_repro_v6.sh
#
# Timestep gotcha: DE Shaw DCD metadata reports 1 ps/frame but the actual
# physical timestep is 200 ps/frame.  --timestep 0.2 is MANDATORY.
# ===========================================================================

#SBATCH --job-name=villin_v6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gputraining
#SBATCH --gres=gpu:batch:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=6:00:00
#SBATCH --output=/mnt/hdd/experiments/logs/villin_repro_v6_%j.out
#SBATCH --error=/mnt/hdd/experiments/logs/villin_repro_v6_%j.err

# ---- Environment setup -----------------------------------------------------
module purge
source /etc/profile.d/modules.sh
module load 12.8
module load pygvamp/1.0.0

mkdir -p /mnt/hdd/experiments/logs

SEED=0
RUN_DIR=$(printf "/mnt/hdd/experiments/villin_repro_v6/seed_%02d" "${SEED}")

# ---- Job info -------------------------------------------------------------
echo "============================================================"
echo "Villin reproduction v6 (RBF range fix probe)"
echo "============================================================"
echo "Job:        ${SLURM_JOB_ID}    Seed: ${SEED}    Output: ${RUN_DIR}"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Node:       $(hostname)"
echo "Start:      $(date)"
echo "Target:     VAMP-2 = 3.78 ± 0.02 (Ghorbani 2022, Table S1)"
echo "Diff vs v4: --distance_min 0.0 --distance_max 3.0 (Ghorbani RBF range)"
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
    --distance_min          0.0 \
    --distance_max          3.0 \
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
