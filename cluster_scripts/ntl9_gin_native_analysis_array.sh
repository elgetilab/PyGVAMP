#!/bin/bash
# ===========================================================================
# PyGVAMP — NTL9 GIN-native ANALYSIS-ONLY re-run (10-seed array)
# ===========================================================================
# The ntl9_gin_native_v1 training array (job 723) trained all 10 seeds and
# saved best_model.pt for each, but every seed then died with exit 137 (host-
# RAM OOM) in the in-training post-analysis block before PHASE 3, so all 10
# analysis/ dirs are EMPTY. (Same failure mode as the SchNet v2 recovery — see
# ntl9_repro_v2_analysis_array.sh — the buggy full-trajectory rebuild in
# training.py blows past the mem cap on NTL9's 14.7M frames.)
#
# THIS SCRIPT recovers the missing analysis WITHOUT retraining. --only_analysis
# skips training entirely (so it never touches the buggy in-training block) and
# runs analysis.py:run_analysis, which subsamples to analysis_max_frames (50k)
# and reuses the existing 7 GB prep cache (hash e36088ee) — it will not OOM.
#
# --resume is REQUIRED: plain --only_analysis creates a fresh empty experiment
# dir and finds no models. --resume <exp_ntl9_<ts>> reuses the per-seed dir
# (model under training/, cache under cache/, output into the empty analysis/).
# Each seed has its own exp_ntl9_* timestamp, resolved by glob below.
#
# MODEL CONFIG must match training so the reconstructed architecture loads the
# saved weights: --model gin with NO arch overrides → the native gin preset
# (hidden=128, output=64, embedding on, clf batch_norm), exactly as the
# ntl9_gin_native_v1 training array ran it. n_neighbors 10, lag 200, k 5 held.
#
# NOTE ON COLLAPSED SEEDS: seeds 0 and 7 collapsed to VAMP-2 = 1.0 (degenerate
# single state) during training. Analysis will still run on them but the MSM /
# ITS / CK output will be trivial — the 8 non-collapsed seeds are the ones that
# carry meaning. Training is left AS IS (no retrain); this only recovers analysis.
#
# RESOURCES: shard:2 / 4 CPUs / 120 GB. The analysis path already subsamples to
# 50k frames (won't OOM at 32 GB, as the SchNet recovery showed), but native GIN
# is a heavier model than the hidden=16 SchNet repro, so we give generous mem
# headroom (120 GB, under the 128000M partition cap) to guarantee the full
# analysis completes. SLURM-script knobs, not a new preset class.
#
# MODULE: deployed pygvamp/1.0.0 (same the training array used).
#
# Submit (test one seed first, then the rest):
#   sbatch --array=0    cluster_scripts/ntl9_gin_native_analysis_array.sh
#   sbatch --array=1-9%4 cluster_scripts/ntl9_gin_native_analysis_array.sh
#
# The headline VAMP-2 number still comes from the training logs
# (aggregate_ntl9_v1_array.py); this re-run produces the MSM/state artifacts
# (learned_K, ITS, CK, state structures, interactive report) the OOM prevented.
# ===========================================================================

#SBATCH --job-name=ntl9_gin_native_anly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gputraining
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=INFINITE
#SBATCH --output=/mnt/hdd/experiments/logs/ntl9_gin_native_anly_%A_%a.out
#SBATCH --error=/mnt/hdd/experiments/logs/ntl9_gin_native_anly_%A_%a.err

# ---- Environment setup -----------------------------------------------------
module purge
source /etc/profile.d/modules.sh
module load cuda/12.8
module load pygvamp/1.0.0

if [ -n "${PYGVAMP_SRC_OVERRIDE}" ]; then
    export PYTHONPATH="${PYGVAMP_SRC_OVERRIDE}:${PYTHONPATH}"
    echo "PYTHONPATH override active: ${PYGVAMP_SRC_OVERRIDE}"
fi

mkdir -p /mnt/hdd/experiments/logs

if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then
    echo "ERROR: submit as an array job, e.g. sbatch --array=0 $0"
    exit 1
fi

SEED=${SLURM_ARRAY_TASK_ID}
SEED_DIR=$(printf "/mnt/hdd/experiments/ntl9_gin_native_v1/seed_%02d" "${SEED}")

# Resolve the existing experiment dir for this seed (created by the training
# array). There is exactly one exp_ntl9_* per seed.
EXP_DIR=$(ls -d "${SEED_DIR}"/exp_ntl9_* 2>/dev/null | head -1)
if [ -z "${EXP_DIR}" ]; then
    echo "ERROR: no exp_ntl9_* found under ${SEED_DIR}"
    exit 1
fi
EXP_NAME=$(basename "${EXP_DIR}")

JOB_NAME="ntl9_gin_native_anly_seed${SEED}"
scontrol update JobId=${SLURM_JOB_ID} Name=${JOB_NAME} 2>/dev/null

# ---- Job info -------------------------------------------------------------
echo "============================================================"
echo "NTL9 GIN-native analysis-only re-run (task ${SLURM_ARRAY_TASK_ID})"
echo "============================================================"
echo "Job:        ${SLURM_JOB_ID}    Seed: ${SEED}"
echo "Seed dir:   ${SEED_DIR}"
echo "Resume:     ${EXP_NAME}"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Node:       $(hostname)"
echo "Start:      $(date)"
echo "Mode:       --only_analysis (skips training; 50k-frame analysis subsample)"
echo "Model:      gin native (no arch overrides), n_neighbors 10, lag 200ns, k 5"
echo "Resources:  shard:2, cpus=4, mem=120G"
echo "============================================================"

# ---- Run -------------------------------------------------------------------
# Same data/model/processing args as the training array so analysis matches the
# training cache hash (e36088ee) and hits the cache — no DCD reprocess. --model
# gin with NO arch overrides → native gin preset (matches the trained weights).
pygvamp \
    --only_analysis \
    --resume       "${EXP_NAME}" \
    --output_dir   "${SEED_DIR}" \
    --traj_dir     /mnt/hdd/data/ntl9/ \
    --top          /mnt/hdd/data/ntl9/topol.pdb \
    --file_pattern 'NTL9-*-c-alpha-*.dcd' \
    --protein_name ntl9 \
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
    --batch_size   1000 \
    --cache

EXIT_CODE=$?

echo "============================================================"
echo "Finished:   $(date)    Exit: ${EXIT_CODE}"
echo "Analysis:   ${EXP_DIR}/analysis"
echo "============================================================"

exit ${EXIT_CODE}
