#!/bin/bash
# ===========================================================================
# PyGVAMP — Villin reproduction v8 (h_g rank-bottleneck head probe)
# ===========================================================================
# Tests whether the Ghorbani 2022 reference's rank-2 classifier head
# (h_g=2) closes the residual ~0.10 VAMP-2 gap to the paper's 3.78.
#
# Single change vs v4: --clf_num_layers 2 --clf_hidden_dim 2.
#
# WHY THIS LEVER:
#   Reference's `model.py` has, after global pool:
#     [Linear(h_a=16, h_g) -> Linear(h_g, n_classes)] -> softmax
#   with NO activation between the two Linears (audited via WebFetch
#   2026-05-04, model.py:fc_classes & amino_emb).  Two unactivated linears
#   compose to a single Linear of rank ≤ h_g — a deliberate rank
#   bottleneck.  TrpCage's `gpu_1.sh` uses h_g=2; the paper abstract says
#   embeddings "were transformed into 2D and trained by maximizing
#   VAMP-2", strongly implying h_g=2 is the standard across all systems
#   (incl. villin, even though Table I doesn't tabulate h_g).
#
#   v4–v7 used --clf_num_layers 1, i.e. a single Linear(h_a=16, n_classes=4)
#   of full rank-4 — strictly more expressive than the reference's rank-2
#   head.  Counter-intuitively, the bottleneck can HELP: the slow
#   processes in MD are typically 1–3D, so aligning the model's effective
#   rank with the slow-mode dimensionality acts as strong regularization
#   against overfitting on fast/noisy modes.
#
#   PyG MLP behavior verified (tests/test_classifier.py:TestRankBottleneckHead):
#   SoftmaxMLP(num_layers=2) produces Linear→Linear→Softmax with NO
#   activation between, regardless of the act= argument.  This is
#   because torch_geometric.nn.models.MLP applies activations BETWEEN
#   Linears, and SoftmaxMLP delegates num_layers-1 Linears to MLP and
#   adds the final Linear separately — so num_layers=2 splits to
#   MLP(num_layers=1, just one Linear, no act between) + final Linear.
#   No code change required; we already have the rank-bottleneck head
#   structure exposed via existing CLI flags.
#
# Decision rule:
#   v8 seed_00 ≲ 3.72 → rank bottleneck doesn't help; either the gap
#                       comes from train/val split methodology, or h_g=2
#                       is wrong for villin specifically.
#   v8 seed_00 ~3.72-3.76 → marginal; consider 3 seeds before committing
#   v8 seed_00 ≳ 3.76 → strong → 10-seed v8 array
#
# What v4-v7 history shows:
#   v1 seed_00 best = 3.5685
#   v2 seed_00 best = 3.6057
#   v3 seed_00 best = 3.6124
#   v4 seed_00 best = 3.7126  (xavier_normal init — kept)
#   v5 seed_00 best = 3.7074  (encoder v2 ReLU — dropped)
#   v6 seed_00 best = 3.6158  (RBF range pin alone — regressed)
#   v7 seed_00 best = 3.7005  (RBF range + σ pin — recovered)
#
# v8 is built on **v4** (xavier_normal + encoder v1 + data-derived RBF)
# with the rank-2 head bolted on.  No RBF pins — v6/v7 conclusively
# ruled out RBF as a meaningful lever at the args.py-default values.
#
# Submit (no --array; single job):
#   sbatch cluster_scripts/villin_repro_v8.sh
#
# No module rebuild needed for v8 — uses already-deployed --clf_num_layers
# and --clf_hidden_dim flags.
#
# Timestep gotcha: DE Shaw DCD metadata reports 1 ps/frame but the actual
# physical timestep is 200 ps/frame.  --timestep 0.2 is MANDATORY.
# ===========================================================================

#SBATCH --job-name=villin_v8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gputraining
#SBATCH --gres=gpu:batch:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=6:00:00
#SBATCH --output=/mnt/hdd/experiments/logs/villin_repro_v8_%j.out
#SBATCH --error=/mnt/hdd/experiments/logs/villin_repro_v8_%j.err

# ---- Environment setup -----------------------------------------------------
module purge
source /etc/profile.d/modules.sh
module load cuda/12.8
module load pygvamp/1.0.0

mkdir -p /mnt/hdd/experiments/logs

SEED=0
RUN_DIR=$(printf "/mnt/hdd/experiments/villin_repro_v8/seed_%02d" "${SEED}")

# ---- Job info -------------------------------------------------------------
echo "============================================================"
echo "Villin reproduction v8 (h_g rank-bottleneck head probe)"
echo "============================================================"
echo "Job:        ${SLURM_JOB_ID}    Seed: ${SEED}    Output: ${RUN_DIR}"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Node:       $(hostname)"
echo "Start:      $(date)"
echo "Target:     VAMP-2 = 3.78 ± 0.02 (Ghorbani 2022, Table S1)"
echo "Diff vs v4: --clf_num_layers 2 --clf_hidden_dim 2 (h_g=2 rank bottleneck)"
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
    --no_use_attention \
    --no_use_embedding \
    --clf_num_layers 2 \
    --clf_hidden_dim 2 \
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
