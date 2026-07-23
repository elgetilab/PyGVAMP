#!/bin/bash
# ===========================================================================
# PyGVAMP — Aβ42, RevGraphVAMP 3-phase reproduction (10-seed array)
# ===========================================================================
# Reproduces RevGraphVAMP (Huang 2024) Table 2 Aβ42: VAMP-2 = 3.99 ± 0.002,
# VAMP-E = 3.99 ± 0.003 (k=4, lag=10 ns). Faithful schedule (see
# claude/REVGRAPHVAMP_TODO.md): χ-pretrain (VAMP-2) → algebraic U/S init →
# joint VAMP-E, via the ported VAMPU/VAMPS layer.
#
# DATA — REDUCED ensemble = RevGraphVAMP's Aβ42 (verified: red/ has exactly 5119
# xtc, matching the paper's "5,119 trajectories"; their repo also stores data
# under trajectories/red/). Do NOT combine with ox (a separate ensemble).
#   traj: /mnt/hdd/data/ab42/trajectories/red/  (nested rN/rNcs subdirs; recursive
#         glob is on by default) — 5119 xtc, ~1.26M frames, 250 ps/frame.
#   top:  /mnt/hdd/data/ab42/trajectories/red/topol.pdb  (42 residues → 42 CA).
# Selection 'name CA' = 42 atoms (matches GitHub --num-atoms 42; paper's 40 is the
# outlier). Lag 10 ns = 40 frames at 0.25 ns/frame (--timestep 0.25 MANDATORY).
# Edge Gaussians over [0,8] with 16 bins (their dmin0/dmax8/step0.5).
#
# ⚠️ EPOCHS: RevGraphVAMP GitHub Aβ42 uses pre_train=300 + epochs=1000 → set
# EPOCH_CHI=300 / EPOCH_ALL=1000 below. That is a LONG run on 1.26M frames
# (~2520 batches/epoch at batch 500). Do a GPU smoke (small epochs) first and
# check per-epoch wall time before committing the full array.
#
# MODULE: needs working-tree reversible-3-phase code → run with
# PYGVAMP_SRC_OVERRIDE (opt-in hook below).
#
# Submit ONE seed first (validate + time it), then the rest:
#   sbatch --array=0 --export=ALL,PYGVAMP_SRC_OVERRIDE=/home/vi/PycharmProjects/PyGVAMP \
#       cluster_scripts/ab42_rev_v1_array.sh
#   sbatch --array=1-9%2 --export=ALL,PYGVAMP_SRC_OVERRIDE=/home/vi/PycharmProjects/PyGVAMP \
#       cluster_scripts/ab42_rev_v1_array.sh
#
# Aggregate:
#   python cluster_scripts/aggregate_reversible_array.py \
#       --root /mnt/hdd/experiments/ab42_rev_v1 --paper-vamp2 3.99 --paper-vampe 3.99
# ===========================================================================

#SBATCH --job-name=ab42_rev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gputraining
#SBATCH --gres=gpu:batch:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=INFINITE
#SBATCH --output=/mnt/hdd/experiments/logs/ab42_rev_%A_%a.out
#SBATCH --error=/mnt/hdd/experiments/logs/ab42_rev_%A_%a.err

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
RUN_DIR=$(printf "/mnt/hdd/experiments/ab42_rev_v1/seed_%02d" "${SEED}")

# --- schedule (faithful: chi -> algebraic init -> joint VAMP-E) ---
EPOCH_CHI=300     # pretrain χ with VAMP-2 (their pre_train_epoch)
EPOCH_US=0        # IGNORED in faithful mode (algebraic U/S init)
EPOCH_ALL=1000    # joint VAMP-E training (their epochs)

JOB_NAME="ab42_rev_seed${SEED}"
scontrol update JobId=${SLURM_JOB_ID} Name=${JOB_NAME} 2>/dev/null

echo "============================================================"
echo "Aβ42 RevGraphVAMP 3-phase (task ${SLURM_ARRAY_TASK_ID}, seed ${SEED})"
echo "Target: VAMP-2 3.99±0.002 / VAMP-E 3.99±0.003 (Huang 2024)"
echo "Phases: chi=${EPOCH_CHI}(5e-4) -> algebraic init -> all=${EPOCH_ALL}(1e-4)"
echo "Data:   ab42 RED (5119 trajs, 42 CA), lag 10ns @ 0.25 ns/frame"
echo "Start:  $(date)   Node: $(hostname)"
echo "============================================================"

pygvamp \
    --traj_dir /mnt/hdd/data/ab42/trajectories/red/ \
    --top      /mnt/hdd/data/ab42/trajectories/red/topol.pdb \
    --file_pattern '*.xtc' \
    --protein_name ab42_red \
    --output_dir   "${RUN_DIR}" \
    --timestep     0.25 \
    --seed         "${SEED}" \
    --model        schnet \
    --selection    'name CA' \
    --stride       1 \
    --lag_times    10.0 \
    --n_states     4 \
    --no_discover_states \
    --max_retrains 0 \
    --no_warm_start_retrains \
    --reversible \
    --rev_three_phase \
    --epoch_chi ${EPOCH_CHI} \
    --epoch_us  ${EPOCH_US} \
    --epoch_all ${EPOCH_ALL} \
    --lr_chi 5e-4 --lr_us 5e-4 --lr_all 1e-4 \
    --rev_activation exp \
    --hidden_dim            16 \
    --output_dim            16 \
    --n_interactions        4 \
    --n_neighbors           10 \
    --gaussian_expansion_dim 16 \
    --distance_min 0 \
    --distance_max 8 \
    --no_use_embedding \
    --clf_num_layers 1 \
    --clf_dropout    0 \
    --clf_norm       none \
    --weight_decay 1e-5 \
    --batch_size   500 \
    --val_split    0.3 \
    --cache

EXIT_CODE=$?
echo "============================================================"
echo "Finished: $(date)   Exit: ${EXIT_CODE}"
echo "============================================================"
exit ${EXIT_CODE}
