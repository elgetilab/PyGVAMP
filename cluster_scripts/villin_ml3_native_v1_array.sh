#!/bin/bash
# ===========================================================================
# PyGVAMP — Villin ML3-encoder, NATIVE regime (v1, 10-seed array)
# ===========================================================================
# Villin analog of trpcage_ml3_native_v1_array.sh. Companion to
# villin_ml3_v1_array.sh (de-tuned). Gives ML3 its OWN preset recipe for a
# best-vs-best comparison (on Trp-cage, native did NOT rescue ML3 — 4.5743 ±
# 0.0770, lower mean + high seed instability; its deficit looked intrinsic).
#
# NATIVE (restored from the ml3 preset / ML3Config + BaseConfig defaults — all
# DEFAULTED, not passed below):
#   ml3_hidden_dim=30, ml3_output_dim=32, ml3_num_layers=4, ml3_node_dim=16,
#   ml3_edge_dim=16, ml3_nout1=30, ml3_nout2=2, ml3_use_attention=ON,
#   ml3_edge_mode=gaussian; use_embedding=ON, clf 2-layer + batch_norm,
#   init=kaiming_normal, lr=1e-3, weight_decay=1e-4.
#   ML3 attention is ml3_use_attention (preset True) — do NOT pass --use_attention.
#
# HELD FIXED (benchmark-invariants — val VAMP-2 comparable to SchNet 3.6923):
#   data (2F4K Cα), selection 'name CA', timestep 0.2, stride 1, lag 20 ns,
#   n_states 4, no discovery, no retrains, epochs 100, val_split 0.3,
#   n_neighbors 10 (matched k-NN graph), seed set.
#
# DELIBERATE DEVIATION — batch_size: ml3 preset batch=32 is impractical; use
# batch=1000 (baseline throughput). val_split held at 0.3 (not native 0.2).
#
#   Villin SchNet v11 : 3.6923 ± 0.0458
#   Villin ML3 native : TBD
#
# DATA: single DESRES 2F4K-0 (Villin) Cα trajectory, 0.2 ns/frame.
# MODULE: deployed pygvamp/1.0.0 (ml3 CLI args present).
#
# Submit ONE seed first, then the rest:
#   sbatch --array=0 cluster_scripts/villin_ml3_native_v1_array.sh
#   sbatch --array=1-9%1 cluster_scripts/villin_ml3_native_v1_array.sh
#
# Aggregate:
#   python cluster_scripts/aggregate_villin_v11_array.py \
#       --root /mnt/hdd/experiments/villin_ml3_native_v1
#
# Timestep gotcha: DESRES DCD metadata reports 1 ps/frame; actual is 200
# ps/frame. --timestep 0.2 is MANDATORY.
# ===========================================================================

#SBATCH --job-name=villin_ml3_native
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gputraining
#SBATCH --gres=gpu:batch:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=INFINITE
#SBATCH --output=/mnt/hdd/experiments/logs/villin_ml3_native_%A_%a.out
#SBATCH --error=/mnt/hdd/experiments/logs/villin_ml3_native_%A_%a.err

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
RUN_DIR=$(printf "/mnt/hdd/experiments/villin_ml3_native_v1/seed_%02d" "${SEED}")

JOB_NAME="villin_ml3_native_seed${SEED}"
scontrol update JobId=${SLURM_JOB_ID} Name=${JOB_NAME} 2>/dev/null

# ---- Job info -------------------------------------------------------------
echo "============================================================"
echo "Villin ML3 NATIVE-regime run v1 array (task ${SLURM_ARRAY_TASK_ID})"
echo "============================================================"
echo "Job:        ${SLURM_JOB_ID}    Seed: ${SEED}    Output: ${RUN_DIR}"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Node:       $(hostname)"
echo "Start:      $(date)"
echo "Target:     Villin SchNet 3.6923 | ML3 native TBD"
echo "Encoder:    ML3 native (hidden=30, output=32, 4 layers, embedding ON, clf batch_norm, lr=1e-3)"
echo "Fixed:      lag 20ns, 4 states, n_neighbors 10, val_split 0.3, batch 1000 (native 32 impractical)"
echo "============================================================"

# ---- Run -------------------------------------------------------------------
# ML3 architecture knobs (ml3_*, use_embedding, clf_*, init_method, lr,
# weight_decay) are intentionally NOT passed → they default to the ml3 preset
# (native). Only benchmark-invariants, n_neighbors (matched), val_split, and
# the practical batch are set. No --use_attention (ML3 uses ml3_use_attention).
pygvamp \
    --traj_dir /mnt/hdd/data/villin/DESRES-Trajectory_2F4K-0-c-alpha/2F4K-0-c-alpha/ \
    --top      /mnt/hdd/data/villin/DESRES-Trajectory_2F4K-0-c-alpha/topol.pdb \
    --file_pattern '2F4K-0-c-alpha-*.dcd' \
    --protein_name villin \
    --output_dir   "${RUN_DIR}" \
    --timestep     0.2 \
    --seed         "${SEED}" \
    --model        ml3 \
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
