#!/bin/bash
# ===========================================================================
# PyGVAMP — Villin reproduction v11 (dual-scoring baseline)
# ===========================================================================
# Same architecture as v10 (the corrected attention baseline), now with both
# scoring methodologies logged side-by-side per epoch:
#
#   concat       — our default, unbiased; drives model selection
#   perbatch_*   — paper's methodology (Ghorbani 2022 / deeptime); reported
#                  additionally for cross-paper comparison
#
# WHY THIS RUN EXISTS:
#
# v10 corrected the missing-attention bug from v1-v9 (see
# claude/ATTENTION_MAPPING_FINDING.md).  Result: best Val VAMP-2 = 3.7298
# under our concat-then-score evaluation.  Paper reports 3.78 ± 0.02.
#
# Post-hoc audit on saved v10 chi (claude/VILLIN_REPRO_V10_LOG.md
# "Post-v10 audit" section) showed the residual ~0.05 gap was a SCORING
# METHODOLOGY mismatch:
#
#   - Reference's train.py:69-74 averages per-batch VAMP-2 across val
#     batches.
#   - Our vampnet.py:evaluate() concats val chi and scores once
#     (deliberately, the more honest unbiased estimator).
#
# On saved v10 chi:
#   concat val (ours)            = 3.7952
#   perbatch_mean val (theirs)   = 3.7649 ± 0.174
# Paper                          = 3.78 ± 0.02
#
# So the v10 model already matches the paper in their methodology —
# the gap was apples-vs-oranges.  v11 confirms this LIVE during training
# by logging both numbers side-by-side every epoch.
#
# WHAT v11 PRODUCES:
#
#   Every epoch's stdout/log line now reads:
#     Epoch K/100, Train VAMP: X.XXXX,
#     Val VAMP: concat=Y.YYYY, perbatch=Z.ZZZZ±W.WWWW
#
# Both metrics persisted in history dict:
#   - epoch_val_scores            (concat — back-compat with plotting)
#   - epoch_val_perbatch_mean
#   - epoch_val_perbatch_std
#
# Model selection still uses concat (the more conservative estimate).
# Reporting alongside lets us argue cleanly:
#   "Our held-out val VAMP-2 = X (concat-then-score, unbiased).
#    Reported in paper-standard per-batch averaged form = Y, within
#    Ghorbani 2022's reported error bar."
#
# ARCHITECTURALLY IDENTICAL TO v10:
#
#   --use_attention (corrects v1-v9 InteractionBlock attention mismapping)
#   data-derived RBF range (no v6/v7 pins)
#   xavier_normal init (kept from v4)
#   encoder v1 (no per-atom ReLU — v5 ruled out v2)
#   clf_num_layers=1 (full-rank head — v8's bottleneck ruled out)
#
# Decision rule (vs v10 result and paper):
#
#   - v11 concat ~3.73, v11 perbatch ~3.77 → confirms the methodology
#     story; reproduction closed under the corrected baseline.
#   - v11 concat << 3.73 or perbatch << 3.77 → something else changed
#     (rebuild issue, regression introduced); investigate.
#
# REBUILD REQUIRED before submission.  This run depends on the modified
# pygv/vampnet/vampnet.py:evaluate() that returns the dual dict.
#
# Submit (no --array; single exploratory seed):
#   sbatch cluster_scripts/villin_repro_v11.sh
#
# Timestep gotcha: DESRES DCD metadata reports 1 ps/frame but the actual
# physical timestep is 200 ps/frame.  --timestep 0.2 is MANDATORY.
# ===========================================================================

#SBATCH --job-name=villin_v11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gputraining
#SBATCH --gres=gpu:batch:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=6:00:00
#SBATCH --output=/mnt/hdd/experiments/logs/villin_repro_v11_%j.out
#SBATCH --error=/mnt/hdd/experiments/logs/villin_repro_v11_%j.err

# ---- Environment setup -----------------------------------------------------
module purge
source /etc/profile.d/modules.sh
module load cuda/12.8
module load pygvamp/1.0.0

mkdir -p /mnt/hdd/experiments/logs

SEED=0
RUN_DIR=$(printf "/mnt/hdd/experiments/villin_repro_v11/seed_%02d" "${SEED}")

# ---- Job info -------------------------------------------------------------
echo "============================================================"
echo "Villin reproduction v11 (dual-scoring baseline)"
echo "============================================================"
echo "Job:        ${SLURM_JOB_ID}    Seed: ${SEED}    Output: ${RUN_DIR}"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Node:       $(hostname)"
echo "Start:      $(date)"
echo "Target:     VAMP-2 = 3.78 ± 0.02 (Ghorbani 2022, Table I)"
echo "Diff vs v10: vampnet.evaluate() now logs both 'concat' and"
echo "              'perbatch_mean ± std' every epoch (paper-comparable)"
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
