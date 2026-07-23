#!/bin/bash
# ===========================================================================
# PyGVAMP — Alanine dipeptide, RevGraphVAMP 3-phase reproduction (10-seed array)
# ===========================================================================
# Reproduces RevGraphVAMP (Huang 2024) Table 2 alanine: VAMP-2 = 4.41 ± 0.01,
# VAMP-E = 4.38 ± 0.01 (k=6, lag=20 ps). Uses the ported reversible VAMP-E layer
# (VAMPU/VAMPS) + 3-phase schedule (--rev_three_phase). See
# claude/REVGRAPHVAMP_TODO.md and REVGRAPHVAMP_VAMPE_VERIFICATION.md.
#
# DATA: /mnt/hdd/data/alanine/ (from cluster_scripts/download_alanine.sh):
#   3× alanine-dipeptide-N-250ns-nowater.xtc, 1 ps/frame, 750k frames total,
#   22 atoms / 10 heavy (ACE-ALA-NME). Selection 'not element H' = 10 heavy atoms.
#
# LAG: 20 ps = 0.02 ns at timestep 0.001 ns/frame.
#
# SCHEDULE (faithful to RevGraphVAMP train_ala.py, resolved 2026-07-23):
#   phase chi (VAMP-2, epoch_chi) -> ALGEBRAIC U/S init (closed form, no gradient)
#   -> phase all (joint VAMP-E, epoch_all). --epoch_us is IGNORED in this
#   (default) faithful mode; kept only to satisfy the CLI. Their knobs are
#   pre_train_epoch (=epoch_chi) and epochs (=epoch_all). EPOCH values below are
#   PROVISIONAL — confirm the exact pre_train/epochs from their run command.
#
# MODULE: needs the working-tree reversible 3-phase code (not yet in the deployed
# module) → run with PYGVAMP_SRC_OVERRIDE (set below via the opt-in hook).
#
# Submit ONE seed first (validate), then the rest:
#   sbatch --array=0 --export=ALL,PYGVAMP_SRC_OVERRIDE=/home/vi/PycharmProjects/PyGVAMP \
#       cluster_scripts/alanine_rev_v1_array.sh
#   sbatch --array=1-9%3 --export=ALL,PYGVAMP_SRC_OVERRIDE=/home/vi/PycharmProjects/PyGVAMP \
#       cluster_scripts/alanine_rev_v1_array.sh
# ===========================================================================

#SBATCH --job-name=ala_rev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gputraining
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=INFINITE
#SBATCH --output=/mnt/hdd/experiments/logs/ala_rev_%A_%a.out
#SBATCH --error=/mnt/hdd/experiments/logs/ala_rev_%A_%a.err

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
RUN_DIR=$(printf "/mnt/hdd/experiments/alanine_rev_v1/seed_%02d" "${SEED}")

# --- schedule (PROVISIONAL — see note above) ---
EPOCH_CHI=100     # phase 1: pretrain χ with VAMP-2 (their pre_train_epoch)
EPOCH_US=0        # IGNORED in faithful mode (algebraic U/S init replaces phase 2)
EPOCH_ALL=100     # phase 3: joint VAMP-E training (their epochs)

JOB_NAME="ala_rev_seed${SEED}"
scontrol update JobId=${SLURM_JOB_ID} Name=${JOB_NAME} 2>/dev/null

echo "============================================================"
echo "Alanine RevGraphVAMP 3-phase (task ${SLURM_ARRAY_TASK_ID}, seed ${SEED})"
echo "Target: VAMP-2 4.41±0.01 / VAMP-E 4.38±0.01 (Huang 2024)"
echo "Phases: chi=${EPOCH_CHI}(5e-4) us=${EPOCH_US}(5e-4) all=${EPOCH_ALL}(1e-4)"
echo "Start:  $(date)   Node: $(hostname)"
echo "============================================================"

pygvamp \
    --traj_dir /mnt/hdd/data/alanine/ \
    --top      /mnt/hdd/data/alanine/alanine-dipeptide-nowater.pdb \
    --file_pattern 'alanine-dipeptide-*-250ns-nowater.xtc' \
    --protein_name alanine \
    --output_dir   "${RUN_DIR}" \
    --timestep     0.001 \
    --seed         "${SEED}" \
    --model        schnet \
    --selection    'not element H' \
    --stride       1 \
    --lag_times    0.02 \
    --n_states     6 \
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
    --n_neighbors           5 \
    --gaussian_expansion_dim 16 \
    --no_use_embedding \
    --clf_num_layers 1 \
    --clf_dropout    0 \
    --clf_norm       none \
    --weight_decay 1e-5 \
    --batch_size   1000 \
    --val_split    0.3 \
    --cache

EXIT_CODE=$?
echo "============================================================"
echo "Finished: $(date)   Exit: ${EXIT_CODE}"
echo "============================================================"
exit ${EXIT_CODE}
