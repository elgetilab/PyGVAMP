#!/bin/bash
# ===========================================================================
# PyGVAMP — Villin reproduction v10 (corrected attention baseline)
# ===========================================================================
# CORRECTS A REPRODUCTION ERROR PRESENT IN v1 THROUGH v9.
#
# Audit on 2026-05-06 (claude/ATTENTION_MAPPING_FINDING.md) showed that
# the Ghorbani 2022 reference's "SchNet" branch (--conv_type SchNet in
# their gpu_1.sh) selects an InteractionBlock whose ContinuousFilterConv
# contains softmax-attention over neighbors with a learnable
# self.nbr_filter parameter:
#
#     nbr_filter = torch.matmul(conv_features, self.nbr_filter).view(...)
#     nbr_filter = F.softmax(nbr_filter, -1)
#
# This is the attention they visualize in the paper.  v1-v9 all used
# --no_use_attention, based on the (wrong) mapping in VILLIN_REPRO_LOG.md
# claiming "Original GraphVAMPNet uses classic SchNet, no attention".
# That mapping has been corrected in-place with a note.
#
# Our pygv/encoder/schnet.py:64-68, 129-135 implements the structurally
# equivalent attention when use_attention=True (single learnable vector,
# matmul, softmax over neighbors, element-wise re-weighting of messages).
#
# Single change vs v4: --use_attention (was --no_use_attention).
#
# Decision rule (vs v4 baseline 3.7126, paper 3.78):
#   - v10 ≳ 3.76 → attention was the missing piece; reproduction closes
#                  within paper's reported error bar.  Run a 10-seed v10
#                  array to confirm.
#   - v10 ~3.72-3.76 → partial improvement; attention helps but doesn't
#                      fully close.  Combine with τ-normalization story
#                      from v9.
#   - v10 ≲ 3.72 → attention isn't the missing piece either.  Revert to
#                  v9's conclusion that the gap is τ-normalization +
#                  reporting offset, not architectural.
#
# Note: with --use_attention, the analysis pipeline bug from v6/v9
# (empty edge_indices → step 7 crash → steps 9-14 skipped) does NOT
# fire — _attention_weights gets populated in the forward pass.  So v10
# should produce the full analysis output (PyMOL renders, ITS plot at
# multiple τ, Chapman-Kolmogorov test, interactive HTML report).
#
# Submit (no --array; single seed exploratory):
#   sbatch cluster_scripts/villin_repro_v10.sh
#
# No module rebuild needed — uses existing --use_attention flag.
#
# Timestep gotcha: DESRES DCD metadata reports 1 ps/frame but the actual
# physical timestep is 200 ps/frame.  --timestep 0.2 is MANDATORY.
# ===========================================================================

#SBATCH --job-name=villin_v10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gputraining
#SBATCH --gres=gpu:batch:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=6:00:00
#SBATCH --output=/mnt/hdd/experiments/logs/villin_repro_v10_%j.out
#SBATCH --error=/mnt/hdd/experiments/logs/villin_repro_v10_%j.err

# ---- Environment setup -----------------------------------------------------
module purge
source /etc/profile.d/modules.sh
module load cuda/12.8
module load pygvamp/1.0.0

mkdir -p /mnt/hdd/experiments/logs

SEED=0
RUN_DIR=$(printf "/mnt/hdd/experiments/villin_repro_v10/seed_%02d" "${SEED}")

# ---- Job info -------------------------------------------------------------
echo "============================================================"
echo "Villin reproduction v10 (corrected attention baseline)"
echo "============================================================"
echo "Job:        ${SLURM_JOB_ID}    Seed: ${SEED}    Output: ${RUN_DIR}"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Node:       $(hostname)"
echo "Start:      $(date)"
echo "Target:     VAMP-2 = 3.78 ± 0.02 (Ghorbani 2022, Table I)"
echo "Diff vs v4: --use_attention (was --no_use_attention) — corrects v1-v9 baseline"
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
    --model        schnet \
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