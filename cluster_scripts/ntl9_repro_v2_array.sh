#!/bin/bash
# ===========================================================================
# PyGVAMP — NTL9 reproduction v2 array (10-seed, 2 shards / 4 CPUs / 32 GB)
# ===========================================================================
# Successor to v1 (job 539), which OOM-killed all 10 seeds at the
# "Computing distance range" step before any epoch ran. Kernel oom_kill
# logs (journalctl on hugin, May 16 00:20-00:21) showed anon-rss of
# 15.0-15.7 GB at kill time against a 16 GB cgroup cap — the working
# set sits right at the boundary, not unboundedly past it.
#
# Root cause: the v1 log estimate ("~5.5M frames") was off. Actual
# data load is 14.7M frames × 39 Cα atoms; the distance-range sampler
# holds the cached coord array (~6.9 GB float32) AND computes pairwise
# distances on top, peaking near 15 GB resident.
#
# WHAT CHANGED FROM v1:
#
#   --gres=shard:2       (was shard:1)   → caps concurrency to 4 on GPU 0
#   --cpus-per-task=4    (was 2)         → ~20-40% per-epoch speedup for
#                                          the vectorized graph build
#   --mem=32G            (was 16G)       → ~16 GB headroom above v1 peak
#
# Each knob independently throttles concurrency to 4 on the current
# partition (8 shards / 2; 16-CPU cap / 4; 128 GB cap / 32), so they're
# consistent. Architecture, hyperparameters, and data are unchanged —
# v2 is purely a resource-budget fix.
#
# WHY 10 SEEDS / ARCHITECTURE / TARGET: same as v1 — see
# claude/NTL9_REPRO_V1_LOG.md and claude/NTL9_REPRO_V2_LOG.md.
#
#   Villin v11  : ours 3.6923 ± 0.0458 vs paper 3.78 ± 0.02 (Δ=-0.088, 1.9σ ours)
#   Trp-cage v1 : ours 4.6516 ± 0.0175 vs paper 4.79 ± 0.01 (Δ=-0.138, 7.9σ ours)
#   NTL9 v2     : TBD                     vs paper 4.59 ± 0.09  (Δ=?)
#
# DATA SCOPE: all four DESHAW NTL9-{0,1,2,3} trajectories combined for
# the strict 1.11 ms total (paper-comparable). 149 DCD files
# (56+54+20+19), 39 Cα atoms each, 200 ps/frame, 14.7M frames total.
#
# WALL-TIME ESTIMATE: ~10-12 min/epoch at 4 CPUs (vs ~2.7 min/epoch on
# Trp-cage v1 at 970 batches/epoch and 2 CPUs; NTL9 has ~3850 train
# batches/epoch and 2× the CPUs). 100 epochs → ~17-20 h per seed.
# %4 concurrency: 10 seeds in 3 waves of (4,4,2) → ~50-60 h end-to-end.
#
# GPU 1's 8 shards are blocked by vLLM 06:00-02:00, so we stay on
# GPU 0 to avoid colliding with vLLM's restart window. Same as v1.
#
# MODULE: requires the rebuilt pygvamp/1.0.0 (with PR #10 vectorization).
# This is what v1 used too — no rebuild needed.
#
# Submit:
#   sbatch --array=0-9%4 cluster_scripts/ntl9_repro_v2_array.sh
#
# Aggregate after completion (v1 aggregator works fine, just point --root):
#   python cluster_scripts/aggregate_ntl9_v1_array.py \
#       --root /mnt/hdd/experiments/ntl9_repro_v2 \
#       --csv /mnt/hdd/experiments/ntl9_repro_v2/summary.csv
#
# Timestep gotcha: DESRES DCD metadata reports 1 ps/frame but the actual
# physical timestep is 200 ps/frame. --timestep 0.2 is MANDATORY.
# ===========================================================================

#SBATCH --job-name=ntl9_v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gputraining
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=INFINITE
#SBATCH --output=/mnt/hdd/experiments/logs/ntl9_v2_%A_%a.out
#SBATCH --error=/mnt/hdd/experiments/logs/ntl9_v2_%A_%a.err

# ---- Environment setup -----------------------------------------------------
module purge
source /etc/profile.d/modules.sh
module load cuda/12.8
module load pygvamp/1.0.0

mkdir -p /mnt/hdd/experiments/logs

if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then
    echo "ERROR: submit as an array job, e.g. sbatch --array=0-9%4 $0"
    exit 1
fi

SEED=${SLURM_ARRAY_TASK_ID}
RUN_DIR=$(printf "/mnt/hdd/experiments/ntl9_repro_v2/seed_%02d" "${SEED}")

JOB_NAME="ntl9_v2_seed${SEED}"
scontrol update JobId=${SLURM_JOB_ID} Name=${JOB_NAME} 2>/dev/null

# ---- Job info -------------------------------------------------------------
echo "============================================================"
echo "NTL9 reproduction v2 array (task ${SLURM_ARRAY_TASK_ID})"
echo "============================================================"
echo "Job:        ${SLURM_JOB_ID}    Seed: ${SEED}    Output: ${RUN_DIR}"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Node:       $(hostname)"
echo "Start:      $(date)"
echo "Target:     VAMP-2 = 4.59 ± 0.09 (Ghorbani 2022, Table S1)"
echo "Arch:       v11 (corrected attention + dual-scoring eval, vectorized)"
echo "Data:       NTL9-{0,1,2,3} combined (1.11 ms, 149 DCDs, 39 Cα atoms)"
echo "Resources:  shard:2 (~8 GB VRAM), cpus=4, mem=32G  (v1 OOM fix)"
echo "============================================================"

# ---- Run -------------------------------------------------------------------
# --traj_dir is the parent containing all 4 trajectory directories;
# default recursive glob walks them. --file_pattern matches any of the
# four trajectory naming conventions (NTL9-0-c-alpha-NNN.dcd through
# NTL9-3-c-alpha-NNN.dcd).
pygvamp \
    --traj_dir /mnt/hdd/data/ntl9/ \
    --top      /mnt/hdd/data/ntl9/topol.pdb \
    --file_pattern 'NTL9-*-c-alpha-*.dcd' \
    --protein_name ntl9 \
    --output_dir   "${RUN_DIR}" \
    --timestep     0.2 \
    --seed         "${SEED}" \
    --model        schnet \
    --selection    'name CA' \
    --stride       1 \
    --lag_times    200.0 \
    --n_states     5 \
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
