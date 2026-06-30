#!/bin/bash
# ===========================================================================
# PyGVAMP — NTL9 GIN-encoder, NATIVE regime (v1, 10-seed array)
# ===========================================================================
# NTL9 analog of trpcage_gin_native_v1_array.sh. Companion to
# ntl9_gin_v1_array.sh (de-tuned). Gives GIN its OWN preset recipe to separate
# "architecture" from "de-tuning" (on Trp-cage, native GIN recovered to SchNet).
#
# NATIVE (gin preset / GINConfig + BaseConfig defaults — all DEFAULTED below):
#   hidden_dim=128, output_dim=64, n_interactions=3, use_embedding=ON,
#   clf_num_layers=2 + clf_norm=batch_norm, init=kaiming_normal, lr=1e-3,
#   weight_decay=1e-4, use_attention=ON.
#
# HELD FIXED (benchmark-invariants — val VAMP-2 comparable to SchNet 4.3459):
#   data (NTL9-{0..3} Cα), selection 'name CA', timestep 0.2, stride 1,
#   lag 200 ns, n_states 5, no discovery, no retrains, epochs 100,
#   val_split 0.3, n_neighbors 10 (matched k-NN graph), seed set.
#
# DELIBERATE DEVIATION — batch_size: gin preset batch=32 is infeasible at
# 14.7M frames; use batch=1000. val_split held at 0.3 (not native 0.2).
#
# VRAM NOTE: native GIN (hidden=128, embedding) is heavier than the de-tuned
# hidden=16 run. shard:2 (~8 GB) matched the SchNet repro; if a native seed
# OOMs on GPU, bump to shard:4 (and adjust the %N throttle). Verify seed 0 first.
#
#   NTL9 SchNet v2 : 4.3459 ± 0.0435
#   NTL9 GIN native: TBD
#
# MODULE: deployed pygvamp/1.0.0.
#
# Submit ONE seed first, then the rest:
#   sbatch --array=0 cluster_scripts/ntl9_gin_native_v1_array.sh
#   sbatch --array=1-9%4 cluster_scripts/ntl9_gin_native_v1_array.sh
#
# Aggregate:
#   python cluster_scripts/aggregate_ntl9_v1_array.py \
#       --root /mnt/hdd/experiments/ntl9_gin_native_v1
#
# Timestep gotcha: DESRES DCD metadata reports 1 ps/frame; actual is 200
# ps/frame. --timestep 0.2 is MANDATORY.
# ===========================================================================

#SBATCH --job-name=ntl9_gin_native
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gputraining
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=INFINITE
#SBATCH --output=/mnt/hdd/experiments/logs/ntl9_gin_native_%A_%a.out
#SBATCH --error=/mnt/hdd/experiments/logs/ntl9_gin_native_%A_%a.err

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
RUN_DIR=$(printf "/mnt/hdd/experiments/ntl9_gin_native_v1/seed_%02d" "${SEED}")

JOB_NAME="ntl9_gin_native_seed${SEED}"
scontrol update JobId=${SLURM_JOB_ID} Name=${JOB_NAME} 2>/dev/null

# ---- Job info -------------------------------------------------------------
echo "============================================================"
echo "NTL9 GIN NATIVE-regime run v1 array (task ${SLURM_ARRAY_TASK_ID})"
echo "============================================================"
echo "Job:        ${SLURM_JOB_ID}    Seed: ${SEED}    Output: ${RUN_DIR}"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Node:       $(hostname)"
echo "Start:      $(date)"
echo "Target:     NTL9 SchNet 4.3459 | GIN native TBD"
echo "Encoder:    GIN native (hidden=128, output=64, embedding ON, clf batch_norm, lr=1e-3)"
echo "Fixed:      lag 200ns, 5 states, n_neighbors 10, val_split 0.3, batch 1000 (native 32 infeasible)"
echo "============================================================"

# ---- Run -------------------------------------------------------------------
# Architecture knobs (hidden_dim, output_dim, n_interactions, use_embedding,
# clf_*, init_method, lr, weight_decay, use_attention) are intentionally NOT
# passed → they default to the gin preset (native). Only benchmark-invariants,
# n_neighbors (matched), val_split, and the practical batch are set.
pygvamp \
    --traj_dir /mnt/hdd/data/ntl9/ \
    --top      /mnt/hdd/data/ntl9/topol.pdb \
    --file_pattern 'NTL9-*-c-alpha-*.dcd' \
    --protein_name ntl9 \
    --output_dir   "${RUN_DIR}" \
    --timestep     0.2 \
    --seed         "${SEED}" \
    --model        gin \
    --selection    'name CA' \
    --stride       1 \
    --lag_times    200.0 \
    --n_states     5 \
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
