#!/bin/bash
# ===========================================================================
# PyGVAMP — Villin reproduction v4 (single-seed init probe)
# ===========================================================================
# Probe whether a tanh-matched weight init closes the residual ~0.12 gap.
# v3 (plain Adam + no jitter) added only +0.007 over v2 — the gap is not
# the optimizer or jitter.  Most v2 seeds settle in a sub-paper basin while
# 2/10 escape to ~3.75 — suggests an init/landscape problem.
#
# Single change vs v2:
#
#   Weight init: kaiming_normal → xavier_normal  (--init_method xavier_normal)
#
#   The current init_for_vamp default calls
#     init_weights(method='kaiming_normal', nonlinearity='relu')
#   (pygv/utils/nn_utils.py:211).  Kaiming was designed for ReLU's
#   half-rectified output and uses scaling factor sqrt(2).  Our pipeline
#   uses tanh as the activation (paper uses tanh post-residual).  Kaiming+
#   ReLU init on a tanh network over-scales by sqrt(2), pushing pre-
#   activations into tanh's saturating tails (|x| large -> tanh -> ±1 ->
#   vanishing gradients).  Xavier (Glorot 2010) is the textbook init for
#   sigmoid/tanh symmetric saturating activations.
#
# Reverts the v3 levers back to v2 settings so we test the init effect in
# isolation (weight_decay=1e-5, training_jitter not overridden -> 1e-6).
#
# Single seed (seed 0) — sanity probe before a 10-seed sweep.
#   v1 seed_00 best = 3.5685
#   v2 seed_00 best = 3.6057
#   v3 seed_00 best = 3.6124
# Decision rule (mirrors v3): <=3.62 abandon / 3.62-3.68 marginal /
# >=3.68 strong -> proceed to 10-seed v4 array.
#
# Note: --init_method requires the args.py + base_config.py + training.py
# changes committed alongside this run.  Module rebuild needed before
# submission.
#
# Submit (no --array; single job):
#   sbatch cluster_scripts/villin_repro_v4.sh
#
# Timestep gotcha: DE Shaw DCD metadata reports 1 ps/frame but the actual
# physical timestep is 200 ps/frame.  --timestep 0.2 is MANDATORY.
# ===========================================================================

#SBATCH --job-name=villin_v4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gputraining
#SBATCH --gres=gpu:batch:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=6:00:00
#SBATCH --output=/mnt/hdd/experiments/logs/villin_repro_v4_%j.out
#SBATCH --error=/mnt/hdd/experiments/logs/villin_repro_v4_%j.err

# ---- Environment setup -----------------------------------------------------
module purge
source /etc/profile.d/modules.sh
module load 12.8
module load pygvamp/1.0.0

mkdir -p /mnt/hdd/experiments/logs

SEED=0
RUN_DIR=$(printf "/mnt/hdd/experiments/villin_repro_v4/seed_%02d" "${SEED}")

# ---- Job info -------------------------------------------------------------
echo "============================================================"
echo "Villin reproduction v4 (single seed init probe)"
echo "============================================================"
echo "Job:        ${SLURM_JOB_ID}    Seed: ${SEED}    Output: ${RUN_DIR}"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Node:       $(hostname)"
echo "Start:      $(date)"
echo "Target:     VAMP-2 = 3.78 ± 0.02 (Ghorbani 2022, Table S1)"
echo "Diff vs v2: weight init kaiming_normal -> xavier_normal"
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
