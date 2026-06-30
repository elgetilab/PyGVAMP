#!/bin/bash
# ===========================================================================
# PyGVAMP — Villin GIN-encoder, NATIVE regime (v1, 10-seed array)
# ===========================================================================
# Villin analog of trpcage_gin_native_v1_array.sh. Companion to
# villin_gin_v1_array.sh (de-tuned). On Trp-cage, de-tuning (esp. stripping
# batch_norm) — not the encoder — was GIN's handicap: native GIN recovered
# 4.5955 → 4.6481 ≈ SchNet. This run gives GIN its OWN recipe on Villin to
# separate "architecture" from "de-tuning".
#
# NATIVE (restored from the gin preset / GINConfig + BaseConfig defaults — all
# DEFAULTED, not passed below):
#   hidden_dim=128, output_dim=64, n_interactions=3, use_embedding=ON,
#   clf_num_layers=2 + clf_norm=batch_norm, init=kaiming_normal, lr=1e-3,
#   weight_decay=1e-4, use_attention=ON. (batch_norm is the expected stabilizer.)
#
# HELD FIXED (benchmark-invariants — so val VAMP-2 is comparable to the Villin
# SchNet 3.6923 baseline): data (2F4K Cα), selection 'name CA', timestep 0.2,
# stride 1, lag 20 ns, n_states 4, no discovery, no retrains, epochs 100,
# val_split 0.3, n_neighbors 10 (identical k-NN graph), seed set.
#
# DELIBERATE DEVIATION — batch_size: gin preset batch=32 is impractical here;
# use batch=1000 (baseline throughput). batch_norm, not tiny batch, is the
# expected stabilizer. val_split held at 0.3 (not native 0.2) to match baseline.
#
#   Villin SchNet v11 : 3.6923 ± 0.0458
#   Villin GIN native : TBD  (does native recover to SchNet parity, as on Trp-cage?)
#
# DATA: single DESRES 2F4K-0 (Villin) Cα trajectory, 0.2 ns/frame.
# MODULE: deployed pygvamp/1.0.0.
#
# Submit ONE seed first (verify before the full sweep):
#   sbatch --array=0 cluster_scripts/villin_gin_native_v1_array.sh
# Then the rest:
#   sbatch --array=1-9%1 cluster_scripts/villin_gin_native_v1_array.sh
#
# Aggregate:
#   python cluster_scripts/aggregate_villin_v11_array.py \
#       --root /mnt/hdd/experiments/villin_gin_native_v1
#
# Timestep gotcha: DESRES DCD metadata reports 1 ps/frame; actual is 200
# ps/frame. --timestep 0.2 is MANDATORY.
# ===========================================================================

#SBATCH --job-name=villin_gin_native
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gputraining
#SBATCH --gres=gpu:batch:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=INFINITE
#SBATCH --output=/mnt/hdd/experiments/logs/villin_gin_native_%A_%a.out
#SBATCH --error=/mnt/hdd/experiments/logs/villin_gin_native_%A_%a.err

# ---- Environment setup -----------------------------------------------------
module purge
source /etc/profile.d/modules.sh
module load cuda/12.8
module load pygvamp/1.0.0

mkdir -p /mnt/hdd/experiments/logs

if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then
    echo "ERROR: submit as an array job, e.g. sbatch --array=0 $0"
    exit 1
fi

SEED=${SLURM_ARRAY_TASK_ID}
RUN_DIR=$(printf "/mnt/hdd/experiments/villin_gin_native_v1/seed_%02d" "${SEED}")

JOB_NAME="villin_gin_native_seed${SEED}"
scontrol update JobId=${SLURM_JOB_ID} Name=${JOB_NAME} 2>/dev/null

# ---- Job info -------------------------------------------------------------
echo "============================================================"
echo "Villin GIN NATIVE-regime run v1 array (task ${SLURM_ARRAY_TASK_ID})"
echo "============================================================"
echo "Job:        ${SLURM_JOB_ID}    Seed: ${SEED}    Output: ${RUN_DIR}"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Node:       $(hostname)"
echo "Start:      $(date)"
echo "Target:     Villin SchNet 3.6923 | GIN native TBD"
echo "Encoder:    GIN native (hidden=128, output=64, embedding ON, clf batch_norm, lr=1e-3)"
echo "Fixed:      lag 20ns, 4 states, n_neighbors 10, val_split 0.3, batch 1000 (native 32 impractical)"
echo "============================================================"

# ---- Run -------------------------------------------------------------------
# Architecture knobs (hidden_dim, output_dim, n_interactions, use_embedding,
# clf_*, init_method, lr, weight_decay, use_attention) are intentionally NOT
# passed → they default to the gin preset (native). Only benchmark-invariants,
# n_neighbors (matched), val_split, and the practical batch are set.
pygvamp \
    --traj_dir /mnt/hdd/data/villin/DESRES-Trajectory_2F4K-0-c-alpha/2F4K-0-c-alpha/ \
    --top      /mnt/hdd/data/villin/DESRES-Trajectory_2F4K-0-c-alpha/topol.pdb \
    --file_pattern '2F4K-0-c-alpha-*.dcd' \
    --protein_name villin \
    --output_dir   "${RUN_DIR}" \
    --timestep     0.2 \
    --seed         "${SEED}" \
    --model        gin \
    --selection    'name CA' \
    --stride       1 \
    --lag_times    20.0 \
    --n_states     4 \
    --no_discover_states \
    --max_retrains 0 \
    --no_warm_start_retrains \
    --n_neighbors  10 \
    --epochs       100 \
    --batch_size   1000 \
    --val_split    0.3 \
    --cache

EXIT_CODE=$?

echo "============================================================"
echo "Finished:   $(date)    Exit: ${EXIT_CODE}"
echo "============================================================"

exit ${EXIT_CODE}
