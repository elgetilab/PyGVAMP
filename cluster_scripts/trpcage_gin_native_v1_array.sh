#!/bin/bash
# ===========================================================================
# PyGVAMP — Trp-cage GIN-encoder, NATIVE regime (v1)
# ===========================================================================
# Companion to trpcage_gin_v1_array.sh. That run DE-TUNED GIN to match the
# SchNet repro (hidden=16, no embedding, no norm, lr=5e-4) for a single-
# variable encoder swap, and GIN came out worse + noisier (4.5955 ± 0.0750 vs
# SchNet 4.6516 ± 0.0175). Diagnosis: that comparison confounds architecture
# with de-tuning — GIN's WL power comes from sum aggregation, which needs the
# normalization/width it was stripped of. This run gives GIN its OWN recipe to
# separate "architecture" from "de-tuning".
#
# NATIVE (restored from the gin preset / GINConfig + BaseConfig defaults):
#   hidden_dim=128, output_dim=64, n_interactions=3, use_embedding=ON,
#   clf_num_layers=2 + clf_norm=batch_norm, init=kaiming_normal, lr=1e-3,
#   weight_decay=1e-4, use_attention=ON.  (All defaulted — NOT passed below.)
#   batch_norm restored is the likely stabilizer for the variance we saw.
#
# HELD FIXED (benchmark-invariants — so the val VAMP-2 is comparable to the
# SchNet 4.6516 baseline; only the encoder + its native regime differ):
#   data (2JOF Cα), selection 'name CA', timestep 0.2, stride 1, lag 20 ns,
#   n_states 5, no discovery, no retrains, epochs 100, val_split 0.3,
#   n_neighbors 7 (identical k-NN graph to the SchNet baseline), seed set.
#
# DELIBERATE DEVIATION FROM NATIVE — batch_size:
#   GIN's preset batch=32 is INFEASIBLE here: 2JOF is ~1.04M frames, so
#   batch 32 → ~23k batches/epoch → ~3 days/seed. We use batch=1000 (matches
#   the baselines' throughput, ~2-4 h/seed). batch_norm — not the tiny batch —
#   is the expected stabilizer, so this should still test the hypothesis.
#   val_split is held at 0.3 (NOT the native 0.2) so the held-out set matches
#   the baseline and the val VAMP-2 is directly comparable.
#
#   SchNet Trp-cage v1 : 4.6516 ± 0.0175
#   GIN (de-tuned)     : 4.5955 ± 0.0750
#   GIN (native)       : 4.6481 ± 0.0343  (n=10) ← RECOVERS to SchNet parity; the
#       de-tuning (esp. stripping batch_norm) was the culprit, not the encoder.
#       Still ~2x SchNet's variance (sum-aggregation conditioning); mean on par.
#       Does NOT beat SchNet despite higher WL-expressiveness — the signal is
#       geometric (distance edge features), so topological power has little to add.
#
# DATA: single DESRES 2JOF (Trp-cage) Cα trajectory, 0.2 ns/frame.
# MODULE: deployed pygvamp/1.0.0.
#
# Submit ONE seed first (verify before the full sweep):
#   sbatch --array=0 cluster_scripts/trpcage_gin_native_v1_array.sh
# Then the rest:
#   sbatch --array=1-9%2 cluster_scripts/trpcage_gin_native_v1_array.sh
#
# Aggregate:
#   python cluster_scripts/aggregate_trpcage_v1_array.py \
#       --root /mnt/hdd/experiments/trpcage_gin_native_v1
#
# Timestep gotcha: DESRES DCD metadata reports 1 ps/frame; actual is 200
# ps/frame. --timestep 0.2 is MANDATORY.
# ===========================================================================

#SBATCH --job-name=trpcage_gin_native
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gputraining
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=INFINITE
#SBATCH --output=/mnt/hdd/experiments/logs/trpcage_gin_native_%A_%a.out
#SBATCH --error=/mnt/hdd/experiments/logs/trpcage_gin_native_%A_%a.err

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
RUN_DIR=$(printf "/mnt/hdd/experiments/trpcage_gin_native_v1/seed_%02d" "${SEED}")

JOB_NAME="trpcage_gin_native_seed${SEED}"
scontrol update JobId=${SLURM_JOB_ID} Name=${JOB_NAME} 2>/dev/null

# ---- Job info -------------------------------------------------------------
echo "============================================================"
echo "Trp-cage GIN NATIVE-regime run v1 array (task ${SLURM_ARRAY_TASK_ID})"
echo "============================================================"
echo "Job:        ${SLURM_JOB_ID}    Seed: ${SEED}    Output: ${RUN_DIR}"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Node:       $(hostname)"
echo "Start:      $(date)"
echo "Target:     SchNet 4.6516 | GIN de-tuned 4.5955 | GIN native TBD"
echo "Encoder:    GIN native (hidden=128, output=64, embedding ON, clf batch_norm, lr=1e-3)"
echo "Fixed:      lag 20ns, 5 states, n_neighbors 7, val_split 0.3, batch 1000 (native 32 infeasible)"
echo "============================================================"

# ---- Run -------------------------------------------------------------------
# Architecture knobs (hidden_dim, output_dim, n_interactions, use_embedding,
# clf_*, init_method, lr, weight_decay, use_attention) are intentionally NOT
# passed → they default to the gin preset (native). Only benchmark-invariants,
# n_neighbors (matched), val_split, and the practical batch are set.
pygvamp \
    --traj_dir /mnt/hdd/data/trpcage/DESRES-Trajectory_2JOF-0-c-alpha/2JOF-0-c-alpha/ \
    --top      /mnt/hdd/data/trpcage/DESRES-Trajectory_2JOF-0-c-alpha/topol.pdb \
    --file_pattern '2JOF-0-c-alpha-*.dcd' \
    --protein_name trpcage \
    --output_dir   "${RUN_DIR}" \
    --timestep     0.2 \
    --seed         "${SEED}" \
    --model        gin \
    --selection    'name CA' \
    --stride       1 \
    --lag_times    20.0 \
    --n_states     5 \
    --no_discover_states \
    --max_retrains 0 \
    --no_warm_start_retrains \
    --n_neighbors  7 \
    --epochs       100 \
    --batch_size   1000 \
    --val_split    0.3 \
    --cache

EXIT_CODE=$?

echo "============================================================"
echo "Finished:   $(date)    Exit: ${EXIT_CODE}"
echo "============================================================"

exit ${EXIT_CODE}
