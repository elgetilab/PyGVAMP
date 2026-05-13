#!/bin/bash
# ===========================================================================
# PyGVAMP — Trp-cage reproduction v1 array (10-seed, dual-scoring baseline)
# ===========================================================================
# Cross-seed sweep of the v11 architecture on Trp-cage (DESHAW 2JOF), after
# the trpcage_repro_v1_probe.sh single-seed check confirms VRAM fits in
# one shard.
#
# WHY 10 SEEDS:
#
# Ghorbani 2022 reports Trp-cage VAMP-2 = 4.79 ± 0.01 (Table S1), averaged
# across 10 trainings. The ± 0.01 is *cross-seed variability*. A single
# seed cannot be compared to that distribution's mean; this array produces
# the matching cross-seed statistic for both scoring methodologies:
#
#   <concat>_seeds         ± stdev_seeds   (our methodology, drives selection)
#   <perbatch_mean>_seeds  ± stdev_seeds   (paper's methodology — direct
#                                           comparison to 4.79 ± 0.01)
#
# v11 villin array landed perbatch = 3.6923 ± 0.0458 vs paper 3.78 ± 0.02
# (-0.088 gap, 1.9σ in our σ, 4.4σ in theirs). Trp-cage is expected to be
# similarly close.
#
# All settings match cluster_scripts/trpcage_repro_v1_probe.sh except
# --seed comes from SLURM_ARRAY_TASK_ID (0..9) instead of pinned to 0.
#
# PARALLELISM CHOICE: shard:1 per seed, throttle %8 (8 concurrent).
# Constrained by GPU 0 shards (8 available) — GPU 1's 8 shards are
# blocked 06:00-02:00 by vLLM, and Trp-cage training takes ~6-8h, so
# using GPU 1 shards risks overrunning vLLM's 06:00 restart window.
# Staying on GPU 0 keeps the cluster's vLLM service uninterrupted.
#
# CPU/mem split: with the gputraining partition cap raised to 16 CPUs,
# 8 concurrent jobs get 2 CPUs each; 128 GB memory cap gives 16 GB each
# (well above this small model's actual usage).
#
# Wall time per seed at 2 CPUs may be slightly slower than the 8-CPU
# probe (~3.5 min/epoch) — estimate 4-6 min/epoch → ~7-10h per seed.
# 8 in parallel → first wave ~10h. Remaining 2 seeds run as the first
# wave finishes (rolling), total ~14h end-to-end.
#
# Probe (job 517) confirmed shard:1 has no OOM at 8 CPUs; this script
# drops CPUs but the model VRAM footprint is unchanged.
#
# Submit:
#   sbatch --array=0-9%8 cluster_scripts/trpcage_repro_v1_array.sh
#
# Aggregate after completion:
#   python cluster_scripts/aggregate_trpcage_v1_array.py
#
# Timestep gotcha: DESRES DCD metadata reports 1 ps/frame but the actual
# physical timestep is 200 ps/frame. --timestep 0.2 is MANDATORY.
# ===========================================================================

#SBATCH --job-name=trpcage_v1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gputraining
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=INFINITE
#SBATCH --output=/mnt/hdd/experiments/logs/trpcage_v1_%A_%a.out
#SBATCH --error=/mnt/hdd/experiments/logs/trpcage_v1_%A_%a.err

# ---- Environment setup -----------------------------------------------------
module purge
source /etc/profile.d/modules.sh
module load 12.8
module load pygvamp/1.0.0

mkdir -p /mnt/hdd/experiments/logs

if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then
    echo "ERROR: submit as an array job, e.g. sbatch --array=0-9%1 $0"
    exit 1
fi

SEED=${SLURM_ARRAY_TASK_ID}
RUN_DIR=$(printf "/mnt/hdd/experiments/trpcage_repro_v1/seed_%02d" "${SEED}")

JOB_NAME="trpcage_v1_seed${SEED}"
scontrol update JobId=${SLURM_JOB_ID} Name=${JOB_NAME} 2>/dev/null

# ---- Job info -------------------------------------------------------------
echo "============================================================"
echo "Trp-cage reproduction v1 array (task ${SLURM_ARRAY_TASK_ID})"
echo "============================================================"
echo "Job:        ${SLURM_JOB_ID}    Seed: ${SEED}    Output: ${RUN_DIR}"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Node:       $(hostname)"
echo "Start:      $(date)"
echo "Target:     VAMP-2 = 4.79 ± 0.01 (Ghorbani 2022, Table S1)"
echo "Arch:       v11 (corrected attention + dual-scoring eval)"
echo "GRES:       shard:1 (~4 GB VRAM)"
echo "============================================================"

# ---- Run -------------------------------------------------------------------
pygvamp \
    --traj_dir /mnt/hdd/data/trpcage/DESRES-Trajectory_2JOF-0-c-alpha/2JOF-0-c-alpha/ \
    --top      /mnt/hdd/data/trpcage/DESRES-Trajectory_2JOF-0-c-alpha/topol.pdb \
    --file_pattern '2JOF-0-c-alpha-*.dcd' \
    --protein_name trpcage \
    --output_dir   "${RUN_DIR}" \
    --timestep     0.2 \
    --seed         "${SEED}" \
    --model        schnet \
    --selection    'name CA' \
    --stride       1 \
    --lag_times    20.0 \
    --n_states     5 \
    --no_discover_states \
    --max_retrains 0 \
    --no_warm_start_retrains \
    --hidden_dim            16 \
    --output_dim            16 \
    --n_interactions        4 \
    --n_neighbors           7 \
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
