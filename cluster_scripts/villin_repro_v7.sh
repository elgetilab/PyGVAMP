#!/bin/bash
# ===========================================================================
# PyGVAMP — Villin reproduction v7 (RBF full-reference-match probe)
# ===========================================================================
# Tests whether *full* RBF parity with the Ghorbani 2022 reference closes
# the gap.  v6 pinned the basis range (dmin=0, dmax=3) but left σ at our
# (dmax-dmin)/K formula = 0.1875.  v7 also pins σ = 0.2 to exactly match
# the reference's `var = step`.
#
# Single change vs v6: + --gaussian_var 0.2.
#
#   Tests/test_dataset.py:TestRBFAgainstGhorbaniReference now contains a
#   test_gaussian_var_override_matches_reference case proving that with
#   --distance_min 0.0 --distance_max 3.0 --gaussian_var 0.2 our expansion
#   reproduces the reference's GaussianDistance output bit-for-bit.
#
# Decision rule:
#   v7 seed_00 ≲ 3.72 → RBF is conclusively ruled out (range was tested
#                       in v6, σ in v7, both on top of v4).  Move on.
#   v7 seed_00 ~3.72-3.76 → marginal; consider 3 seeds before committing
#   v7 seed_00 ≳ 3.76 → strong → 10-seed v7 array
#
# Note: v7 inherits v6's range pin (dmin=0, dmax=3), so it's still
# subject to the "2 dead centers below physical Cα-Cα minimum" cost
# diagnosed in v6.  If v7 underperforms by less than v6, the σ formula
# was a partial contributor; if v7 ~ v6, σ is irrelevant.
#
# Single seed (seed 0) — sanity probe before any sweep.
#   v1 seed_00 best = 3.5685
#   v2 seed_00 best = 3.6057
#   v3 seed_00 best = 3.6124
#   v4 seed_00 best = 3.7126  (xavier_normal init — kept)
#   v5 seed_00 best = 3.7074  (encoder v2 — dropped)
#   v6 seed_00 best = 3.6158  (RBF range pin — regressed by 0.097)
#
# v7 is built on **v4 + v6's range pin + new σ pin**.  v5's per-atom
# ReLU stays dropped.
#
# Note: --gaussian_var requires the args.py + base_config.py +
# master_pipeline.py + preparation.py + training.py + analysis.py +
# vampnet_dataset.py changes committed alongside this run.  Module
# rebuild needed before submission.
#
# Submit (no --array; single job):
#   sbatch cluster_scripts/villin_repro_v7.sh
#
# Timestep gotcha: DE Shaw DCD metadata reports 1 ps/frame but the actual
# physical timestep is 200 ps/frame.  --timestep 0.2 is MANDATORY.
# ===========================================================================

#SBATCH --job-name=villin_v7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gputraining
#SBATCH --gres=gpu:batch:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=6:00:00
#SBATCH --output=/mnt/hdd/experiments/logs/villin_repro_v7_%j.out
#SBATCH --error=/mnt/hdd/experiments/logs/villin_repro_v7_%j.err

# ---- Environment setup -----------------------------------------------------
module purge
source /etc/profile.d/modules.sh
module load 12.8
module load pygvamp/1.0.0

mkdir -p /mnt/hdd/experiments/logs

SEED=0
RUN_DIR=$(printf "/mnt/hdd/experiments/villin_repro_v7/seed_%02d" "${SEED}")

# ---- Job info -------------------------------------------------------------
echo "============================================================"
echo "Villin reproduction v7 (RBF full-reference-match probe)"
echo "============================================================"
echo "Job:        ${SLURM_JOB_ID}    Seed: ${SEED}    Output: ${RUN_DIR}"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Node:       $(hostname)"
echo "Start:      $(date)"
echo "Target:     VAMP-2 = 3.78 ± 0.02 (Ghorbani 2022, Table S1)"
echo "Diff vs v6: + --gaussian_var 0.2 (σ = step, full reference match)"
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
    --gaussian_var          0.2 \
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
