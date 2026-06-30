#!/bin/bash
# ===========================================================================
# PyGVAMP — Villin GIN-encoder run v1 (10-seed array, DE-TUNED regime)
# ===========================================================================
# Villin analog of trpcage_gin_v1_array.sh. Apples-to-apples encoder swap:
# IDENTICAL to the Villin SchNet reproduction (villin_repro_v11_array.sh) in
# every hyperparameter, data selection, lag, k, and seed set — the ONLY change
# is the encoder:
#
#     --model schnet   →   --model gin
#
# DE-TUNED: GIN forced into the SchNet repro's recipe (hidden=16, output=16,
# n_interactions=4, gaussian=16, attention on, no embedding, clf 1-layer/no-norm,
# init=xavier_normal, lr=5e-4, wd=1e-5) so the cross-seed VAMP-2 is directly
# comparable to the SchNet baseline. The 'gin' preset's own defaults are all
# overridden below (SLURM/CLI knobs, NOT a new preset — feedback_per_experiment_presets).
#
#   Villin SchNet v11 : 3.6923 ± 0.0458  vs paper 3.78 ± 0.02  (Δ=-0.088, 1.9σ ours)
#   Villin GIN  (det) : TBD              vs the SchNet baseline (the point of this run)
#   (cf. Trp-cage: GIN de-tuned 4.5955 ± 0.0750 vs SchNet 4.6516 — worse + noisier
#    when stripped of batch_norm; native GIN recovered. See villin_gin_native_v1.)
#
# DATA: single DESRES 2F4K-0 (Villin) Cα trajectory, 0.2 ns/frame.
# MODULE: deployed pygvamp/1.0.0.
#
# Submit:
#   sbatch --array=0-9%1 cluster_scripts/villin_gin_v1_array.sh
#
# Aggregate after completion:
#   python cluster_scripts/aggregate_villin_v11_array.py \
#       --root /mnt/hdd/experiments/villin_gin_v1
#
# Timestep gotcha: DESRES DCD metadata reports 1 ps/frame; actual is 200
# ps/frame. --timestep 0.2 is MANDATORY.
# ===========================================================================

#SBATCH --job-name=villin_gin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gputraining
#SBATCH --gres=gpu:batch:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=INFINITE
#SBATCH --output=/mnt/hdd/experiments/logs/villin_gin_%A_%a.out
#SBATCH --error=/mnt/hdd/experiments/logs/villin_gin_%A_%a.err

# ---- Environment setup -----------------------------------------------------
module purge
source /etc/profile.d/modules.sh
module load cuda/12.8
module load pygvamp/1.0.0

mkdir -p /mnt/hdd/experiments/logs

if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then
    echo "ERROR: submit as an array job, e.g. sbatch --array=0-9%1 $0"
    exit 1
fi

SEED=${SLURM_ARRAY_TASK_ID}
RUN_DIR=$(printf "/mnt/hdd/experiments/villin_gin_v1/seed_%02d" "${SEED}")

JOB_NAME="villin_gin_seed${SEED}"
scontrol update JobId=${SLURM_JOB_ID} Name=${JOB_NAME} 2>/dev/null

# ---- Job info -------------------------------------------------------------
echo "============================================================"
echo "Villin GIN-encoder run v1 array (task ${SLURM_ARRAY_TASK_ID})"
echo "============================================================"
echo "Job:        ${SLURM_JOB_ID}    Seed: ${SEED}    Output: ${RUN_DIR}"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Node:       $(hostname)"
echo "Start:      $(date)"
echo "Target:     VAMP-2 = 3.78 ± 0.02 (Ghorbani 2022) | SchNet baseline 3.6923"
echo "Encoder:    GIN de-tuned (sum + attention), repro-matched hyperparams"
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
    --model        gin \
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
