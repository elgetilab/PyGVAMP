#!/bin/bash
# ===========================================================================
# PyGVAMP — NTL9 GIN-encoder run v1 (10-seed array, DE-TUNED regime)
# ===========================================================================
# NTL9 analog of trpcage_gin_v1_array.sh. Apples-to-apples encoder swap:
# IDENTICAL to the NTL9 SchNet reproduction (ntl9_repro_v2_array.sh) in every
# hyperparameter, data, lag, k, and seed set — the ONLY change is the encoder:
#
#     --model schnet   →   --model gin
#
# DE-TUNED: GIN forced into the SchNet repro recipe (hidden=16, output=16,
# n_interactions=4, gaussian=16, attention on, no embedding, clf 1-layer/no-norm,
# init=xavier_normal, lr=5e-4, wd=1e-5). 'gin' preset defaults overridden below.
#
#   NTL9 SchNet v2 : 4.3459 ± 0.0435  vs paper 4.59 ± 0.09  (Δ=-0.244)
#   NTL9 GIN (det) : TBD
#   (cf. Trp-cage: GIN de-tuned 4.5955 ± 0.0750 vs SchNet 4.6516; native recovered.)
#
# DATA SCOPE: all four DESHAW NTL9-{0,1,2,3} trajectories combined (1.11 ms,
# 149 DCDs, 39 Cα atoms, 200 ps/frame, 14.7M frames). Resources mirror the
# NTL9 SchNet repro v2 OOM fix (shard:2 / 4 CPU / 32G).
# MODULE: deployed pygvamp/1.0.0.
#
# Submit ONE seed first (this is the large/slow system — verify before the sweep):
#   sbatch --array=0 cluster_scripts/ntl9_gin_v1_array.sh
# Then the rest:
#   sbatch --array=1-9%4 cluster_scripts/ntl9_gin_v1_array.sh
#
# Aggregate:
#   python cluster_scripts/aggregate_ntl9_v1_array.py \
#       --root /mnt/hdd/experiments/ntl9_gin_v1
#
# Timestep gotcha: DESRES DCD metadata reports 1 ps/frame; actual is 200
# ps/frame. --timestep 0.2 is MANDATORY.
# ===========================================================================

#SBATCH --job-name=ntl9_gin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gputraining
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=INFINITE
#SBATCH --output=/mnt/hdd/experiments/logs/ntl9_gin_%A_%a.out
#SBATCH --error=/mnt/hdd/experiments/logs/ntl9_gin_%A_%a.err

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
RUN_DIR=$(printf "/mnt/hdd/experiments/ntl9_gin_v1/seed_%02d" "${SEED}")

JOB_NAME="ntl9_gin_seed${SEED}"
scontrol update JobId=${SLURM_JOB_ID} Name=${JOB_NAME} 2>/dev/null

# ---- Job info -------------------------------------------------------------
echo "============================================================"
echo "NTL9 GIN-encoder run v1 array (task ${SLURM_ARRAY_TASK_ID})"
echo "============================================================"
echo "Job:        ${SLURM_JOB_ID}    Seed: ${SEED}    Output: ${RUN_DIR}"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Node:       $(hostname)"
echo "Start:      $(date)"
echo "Target:     VAMP-2 = 4.59 ± 0.09 (Ghorbani 2022) | SchNet baseline 4.3459"
echo "Encoder:    GIN de-tuned (sum + attention), repro-matched hyperparams"
echo "Data:       NTL9-{0,1,2,3} combined (1.11 ms, 149 DCDs, 39 Cα)"
echo "Resources:  shard:2 (~8 GB VRAM), cpus=4, mem=32G"
echo "============================================================"

# ---- Run -------------------------------------------------------------------
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
