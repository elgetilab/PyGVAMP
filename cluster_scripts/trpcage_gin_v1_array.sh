#!/bin/bash
# ===========================================================================
# PyGVAMP — Trp-cage GIN-encoder run v1 (10-seed array)
# ===========================================================================
# Apples-to-apples encoder comparison: IDENTICAL to the Trp-cage SchNet
# reproduction (trpcage_repro_v1_array.sh) in every hyperparameter, data
# selection, lag time, and seed set — the ONLY change is the encoder:
#
#     --model schnet   →   --model gin
#
# Everything else (hidden=16, output=16, n_interactions=4, n_neighbors=7,
# gaussian=16, attention on, no embedding, clf 1-layer/no-norm,
# init=xavier_normal, lr=5e-4, wd=1e-5, 100 epochs, batch=1000,
# val_split=0.3, lag=20ns, 5 states, no state discovery, no retrains) is the
# same, so the cross-seed VAMP-2 is directly comparable to the SchNet
# baseline.
#
#   SchNet Trp-cage v1 : 4.6516 ± 0.0175  vs paper 4.79 ± 0.01  (Δ=-0.138)
#   GIN    Trp-cage v1 : TBD              vs paper 4.79 ± 0.01  (Δ=?)
#   (and TBD vs the SchNet baseline 4.6516 — the point of this run)
#
# These are SLURM-script + CLI knobs overriding the 'gin' config preset,
# NOT a new preset class (per feedback_per_experiment_presets). The 'gin'
# preset's own defaults (hidden=128, output=64, n_neighbors=4, embedding on,
# clf 2-layer+batch_norm, lr=1e-3, batch=32) are all overridden below to
# match the SchNet repro.
#
# DATA: single DESRES 2JOF (Trp-cage) Cα trajectory, 0.2 ns/frame.
# MODULE: deployed pygvamp/1.0.0 (same as the reproductions).
#
# Submit:
#   sbatch --array=0-9%2 cluster_scripts/trpcage_gin_v1_array.sh
#
# Aggregate after completion (VAMP-2 from training logs):
#   python cluster_scripts/aggregate_trpcage_v1_array.py \
#       --root /mnt/hdd/experiments/trpcage_gin_v1
#
# Timestep gotcha: DESRES DCD metadata reports 1 ps/frame but the actual
# physical timestep is 200 ps/frame. --timestep 0.2 is MANDATORY.
# ===========================================================================

#SBATCH --job-name=trpcage_gin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gputraining
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=INFINITE
#SBATCH --output=/mnt/hdd/experiments/logs/trpcage_gin_%A_%a.out
#SBATCH --error=/mnt/hdd/experiments/logs/trpcage_gin_%A_%a.err

# ---- Environment setup -----------------------------------------------------
module purge
source /etc/profile.d/modules.sh
module load cuda/12.8
module load pygvamp/1.0.0

mkdir -p /mnt/hdd/experiments/logs

if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then
    echo "ERROR: submit as an array job, e.g. sbatch --array=0-9%2 $0"
    exit 1
fi

SEED=${SLURM_ARRAY_TASK_ID}
RUN_DIR=$(printf "/mnt/hdd/experiments/trpcage_gin_v1/seed_%02d" "${SEED}")

JOB_NAME="trpcage_gin_seed${SEED}"
scontrol update JobId=${SLURM_JOB_ID} Name=${JOB_NAME} 2>/dev/null

# ---- Job info -------------------------------------------------------------
echo "============================================================"
echo "Trp-cage GIN-encoder run v1 array (task ${SLURM_ARRAY_TASK_ID})"
echo "============================================================"
echo "Job:        ${SLURM_JOB_ID}    Seed: ${SEED}    Output: ${RUN_DIR}"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Node:       $(hostname)"
echo "Start:      $(date)"
echo "Target:     VAMP-2 = 4.79 ± 0.01 (Ghorbani 2022) | SchNet baseline 4.6516"
echo "Encoder:    GIN (sum + attention aggregation), repro-matched hyperparams"
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
    --model        gin \
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
