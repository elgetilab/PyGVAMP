#!/bin/bash
# ===========================================================================
# PyGVAMP — NTL9 reproduction v2 ANALYSIS-ONLY re-run (10-seed array)
# ===========================================================================
# The v2 training array (job 552) trained all 10 seeds successfully but every
# seed then died with exit 137 (OOM) before producing any analysis output —
# all 10 analysis/ dirs are empty.
#
# ROOT CAUSE (not the v1 prep/distance-range OOM, and NOT the pipeline's
# PHASE 3 analysis): the OOM is inside run_training itself. After saving the
# model, training.py runs a post-training analysis block
# (analyze_vampnet_outputs + CK + ITS) over a frame loader built from the
# FULL ~11.8M-frame split with NO subsampling (training.py is_frame_loader
# path), unlike analysis.py which caps at analysis_max_frames=50k. For NTL9's
# 14.7M frames that rebuild + full-trajectory inference blows past the 32 GB
# cap, killing the process before PHASE 3 ever runs.
#
# THIS SCRIPT recovers the missing analysis WITHOUT retraining. The pipeline's
# --only_analysis mode skips training entirely (so it never touches the buggy
# in-training block) and runs analysis.py:run_analysis, which subsamples to
# 50k frames and reuses the existing 7 GB prep cache. It will not OOM.
#
# --resume is REQUIRED: plain --only_analysis would create a fresh empty
# experiment dir and discover no models. --resume <exp_<ts>> reuses the
# existing per-seed experiment dir (model under training/, cache under cache/,
# output into the empty analysis/). Each seed has a different exp_<ts>, so we
# resolve it by glob below.
#
# Resources unchanged from the v2 training array: shard:2 / 4 CPUs / 32 GB.
# Analysis loads the 7 GB cache then subsamples → comfortably under 32 GB.
# (These are SLURM-script knobs, not a new preset class.)
#
# MODULE: same deployed pygvamp/1.0.0 the training array used — no rebuild
# needed for this re-run (the fix to the in-training block is a separate change
# that only matters for future FULL runs).
#
# Submit (test one seed first, then the rest):
#   sbatch --array=0    cluster_scripts/ntl9_repro_v2_analysis_array.sh
#   sbatch --array=1-9%4 cluster_scripts/ntl9_repro_v2_analysis_array.sh
#
# Aggregation is unaffected — the headline VAMP-2 number comes from the
# training logs (aggregate_ntl9_v1_array.py), not from analysis. This re-run
# produces the MSM/state artifacts (learned_K, ITS, CK, state structures,
# interactive report) that the OOM prevented.
# ===========================================================================

#SBATCH --job-name=ntl9_v2_anly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gputraining
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=INFINITE
#SBATCH --output=/mnt/hdd/experiments/logs/ntl9_v2_anly_%A_%a.out
#SBATCH --error=/mnt/hdd/experiments/logs/ntl9_v2_anly_%A_%a.err

# ---- Environment setup -----------------------------------------------------
module purge
source /etc/profile.d/modules.sh
module load cuda/12.8
module load pygvamp/1.0.0

# Opt-in: test uncommitted working-tree fixes against the cluster without
# touching the shared module. No effect unless PYGVAMP_SRC_OVERRIDE is set,
# e.g.  sbatch --export=ALL,PYGVAMP_SRC_OVERRIDE=/home/vi/PycharmProjects/PyGVAMP ...
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
SEED_DIR=$(printf "/mnt/hdd/experiments/ntl9_repro_v2/seed_%02d" "${SEED}")

# Resolve the existing experiment dir for this seed (created by the training
# array). There is exactly one exp_ntl9_* per seed.
EXP_DIR=$(ls -d "${SEED_DIR}"/exp_ntl9_* 2>/dev/null | head -1)
if [ -z "${EXP_DIR}" ]; then
    echo "ERROR: no exp_ntl9_* found under ${SEED_DIR}"
    exit 1
fi
EXP_NAME=$(basename "${EXP_DIR}")

JOB_NAME="ntl9_v2_anly_seed${SEED}"
scontrol update JobId=${SLURM_JOB_ID} Name=${JOB_NAME} 2>/dev/null

# ---- Job info -------------------------------------------------------------
echo "============================================================"
echo "NTL9 reproduction v2 analysis-only re-run (task ${SLURM_ARRAY_TASK_ID})"
echo "============================================================"
echo "Job:        ${SLURM_JOB_ID}    Seed: ${SEED}"
echo "Seed dir:   ${SEED_DIR}"
echo "Resume:     ${EXP_NAME}"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Node:       $(hostname)"
echo "Start:      $(date)"
echo "Mode:       --only_analysis (skips training; 50k-frame analysis subsample)"
echo "============================================================"

# ---- Run -------------------------------------------------------------------
# Same data/model/processing args as the training array so analysis matches the
# training cache hash (e36088ee) and hits the cache — no DCD reprocess.
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
    --batch_size   1000 \
    --cache

EXIT_CODE=$?

echo "============================================================"
echo "Finished:   $(date)    Exit: ${EXIT_CODE}"
echo "Analysis:   ${EXP_DIR}/analysis"
echo "============================================================"

exit ${EXIT_CODE}
