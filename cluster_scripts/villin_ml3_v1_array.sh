#!/bin/bash
# ===========================================================================
# PyGVAMP — Villin ML3-encoder run v1 (10-seed array, DE-TUNED regime)
# ===========================================================================
# Villin analog of trpcage_ml3_v1_array.sh. ML3 width-matched to the Villin
# SchNet/GIN baselines for a single-variable encoder swap.
#
# WIDTH MATCHING: ML3 reads its OWN dims (ml3_*), not the generic
# --hidden_dim/--output_dim. ml3_output_dim=16 / ml3_hidden_dim=16 /
# ml3_num_layers=4 match the SchNet/GIN repro bottleneck (output=16, 4 layers).
# This does NOT equalise total params (ML3 carries fixed spectral machinery,
# ~46k vs ~7k) — the control is on the embedding dim + depth, not param count.
#   - ML3 attention is ml3_use_attention (preset True) — do NOT pass --use_attention.
#   - ml3_edge_dim=16 matches gaussian_expansion_dim=16 (gaussian edge mode).
#
#   Villin SchNet v11 : 3.6923 ± 0.0458  vs paper 3.78 ± 0.02  (Δ=-0.088, 1.9σ ours)
#   Villin ML3  (det) : TBD
#   (cf. Trp-cage: ML3 de-tuned 4.6209 ± 0.0335 vs SchNet 4.6516.)
#
# DATA: single DESRES 2F4K-0 (Villin) Cα trajectory, 0.2 ns/frame.
# MODULE: deployed pygvamp/1.0.0 (ml3 CLI args present).
#
# Submit ONE seed first, then the rest:
#   sbatch --array=0 cluster_scripts/villin_ml3_v1_array.sh
#   sbatch --array=1-9%1 cluster_scripts/villin_ml3_v1_array.sh
#
# Aggregate:
#   python cluster_scripts/aggregate_villin_v11_array.py \
#       --root /mnt/hdd/experiments/villin_ml3_v1
#
# Timestep gotcha: DESRES DCD metadata reports 1 ps/frame; actual is 200
# ps/frame. --timestep 0.2 is MANDATORY.
# ===========================================================================

#SBATCH --job-name=villin_ml3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gputraining
#SBATCH --gres=gpu:batch:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=INFINITE
#SBATCH --output=/mnt/hdd/experiments/logs/villin_ml3_%A_%a.out
#SBATCH --error=/mnt/hdd/experiments/logs/villin_ml3_%A_%a.err

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
RUN_DIR=$(printf "/mnt/hdd/experiments/villin_ml3_v1/seed_%02d" "${SEED}")

JOB_NAME="villin_ml3_seed${SEED}"
scontrol update JobId=${SLURM_JOB_ID} Name=${JOB_NAME} 2>/dev/null

# ---- Job info -------------------------------------------------------------
echo "============================================================"
echo "Villin ML3-encoder run v1 array (task ${SLURM_ARRAY_TASK_ID})"
echo "============================================================"
echo "Job:        ${SLURM_JOB_ID}    Seed: ${SEED}    Output: ${RUN_DIR}"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Node:       $(hostname)"
echo "Start:      $(date)"
echo "Target:     VAMP-2 = 3.78 ± 0.02 (Ghorbani 2022) | SchNet baseline 3.6923"
echo "Encoder:    ML3 de-tuned (width-matched: output=16, hidden=16, 4 layers)"
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
    --model        ml3 \
    --selection    'name CA' \
    --stride       1 \
    --lag_times    20.0 \
    --n_states     4 \
    --no_discover_states \
    --max_retrains 0 \
    --no_warm_start_retrains \
    --ml3_node_dim   16 \
    --ml3_edge_dim   16 \
    --ml3_hidden_dim 16 \
    --ml3_output_dim 16 \
    --ml3_num_layers 4 \
    --n_neighbors           10 \
    --gaussian_expansion_dim 16 \
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
