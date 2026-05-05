#!/bin/bash
# ===========================================================================
# PyGVAMP — Overnight production validation run
# ===========================================================================
# Exercises Phase 2-4 features (auto-stride, warm-start retrains, new retrain
# policy) against ab42_red at 50 epochs across a ladder of lag times.
#
# Submit as an array:
#   sbatch --array=0-3%1 cluster_scripts/overnight_test.sh
#   # %1 throttles to one task at a time (single GPU).
#
# Wall-time estimate (frame_dt=250ps, prep stride=1):
#   τ=1  -> runtime_stride=1   ≈ 13h  (control; matches job 369 shape)
#   τ=5  -> runtime_stride=2   ≈ 6.5h
#   τ=10 -> runtime_stride=4   ≈ 3.25h
#   τ=20 -> runtime_stride=8   ≈ 1.6h
#   TOTAL  ≈ 24h (plus retrain rounds; rule-(a) convergence should cut these)
# ===========================================================================

#SBATCH --job-name=ab42r_night
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gputraining
#SBATCH --gres=gpu:batch:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=1-12:00:00
#SBATCH --output=/mnt/hdd/experiments/logs/overnight_%A_%a.out
#SBATCH --error=/mnt/hdd/experiments/logs/overnight_%A_%a.err

# ---- Experiment configuration ---------------------------------------------

PROTEIN_NAME="ab42_red"
TOPOLOGY="/mnt/hdd/data/ab42/trajectories/red/topol.pdb"
TRAJ_DIR="/mnt/hdd/data/ab42/trajectories/red/"
SELECTION="name CA"

# Ladder: τ=1 control (no auto-stride subsample) + 3 lags that exercise auto-stride
LAG_TIMES=(1 5 10 20)

PRESET="medium_schnet"
N_STATES=10
EPOCHS=50
STRIDE=1               # preprocessing stride (auto-stride subsamples on top)
BATCH_SIZE=2048
TIMESTEP_NS=0.25       # raw frame_dt = 250 ps — required by --auto_stride

OUTPUT_BASE="/mnt/hdd/experiments/ab42_red_overnight_test"

# ---- Environment setup -----------------------------------------------------

module purge
source /etc/profile.d/modules.sh
module load 12.8
module load pygvamp/1.0.0

mkdir -p /mnt/hdd/experiments/logs

if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then
    echo "ERROR: submit as an array job, e.g. sbatch --array=0-3%1 $0"
    exit 1
fi

if [ ${SLURM_ARRAY_TASK_ID} -ge ${#LAG_TIMES[@]} ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} exceeds ${#LAG_TIMES[@]} lag times"
    exit 1
fi

LAG=${LAG_TIMES[$SLURM_ARRAY_TASK_ID]}
RUN_DIR="${OUTPUT_BASE}/lag${LAG}"

JOB_NAME="ab42r_night_lag${LAG}"
scontrol update JobId=${SLURM_JOB_ID} Name=${JOB_NAME} 2>/dev/null

# ---- Command construction --------------------------------------------------
# New-feature flags:
#   --auto_stride            Phase 2: per-lag runtime subsampling
#   --timestep 0.25          required by --auto_stride (ab42 raw frame_dt = 250 ps)
#   --lr_schedule cosine     anneal LR to 1e-5 over training (smoother late epochs)
#   warm_start_retrains,
#   max_retrains=5,
#   convergence_check        all on by default after Phase 4

CMD="pygvamp"
CMD+=" --traj_dir ${TRAJ_DIR}"
CMD+=" --top ${TOPOLOGY}"
CMD+=" --lag_times ${LAG}"
CMD+=" --protein_name ${PROTEIN_NAME}"
CMD+=" --output_dir ${RUN_DIR}"
CMD+=" --preset ${PRESET}"
CMD+=" --n_states ${N_STATES}"
CMD+=" --no_discover_states"
CMD+=" --epochs ${EPOCHS}"
CMD+=" --batch_size ${BATCH_SIZE}"
CMD+=" --stride ${STRIDE}"
CMD+=" --selection '${SELECTION}'"
CMD+=" --timestep ${TIMESTEP_NS}"
CMD+=" --auto_stride"
CMD+=" --lr_schedule cosine --lr_min 1e-5"
# Plateau-based early stopping (suggested defaults, validated on ab42_red):
#   patience=8        stop after 8 consecutive sub-threshold epochs
#   tol=5e-4          relative; an improvement >0.05% per epoch resets counter
#                     (1e-4 was too tight — asymptotic noise kept resetting the
#                     counter, so initial training ran near full 50 epochs)
#   min_epochs=10     warmup — no early-stop before epoch 10
CMD+=" --early_stopping_patience 8"
CMD+=" --early_stopping_tol 5e-4"
CMD+=" --early_stopping_min_epochs 10"
CMD+=" --cache"

# ---- Job info -------------------------------------------------------------
echo "============================================================"
echo "PyGVAMP — Overnight validation run (array task ${SLURM_ARRAY_TASK_ID})"
echo "============================================================"
echo "Job:          ${SLURM_JOB_ID}"
echo "Protein:      ${PROTEIN_NAME}"
echo "Lag time:     ${LAG} ns"
echo "Output:       ${RUN_DIR}"
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Node:         $(hostname)"
echo "Start:        $(date)"
echo "Command:      ${CMD}"
echo "Features on:  auto_stride, warm_start_retrains(default), max_retrains=5,"
echo "              convergence_check(default rule-a), lr_schedule=cosine,"
echo "              early_stopping(patience=8, tol=1e-4, min_epochs=10)"
echo "============================================================"

eval ${CMD}
EXIT_CODE=$?

echo "============================================================"
echo "Finished:     $(date)"
echo "Exit code:    ${EXIT_CODE}"
echo "============================================================"

exit ${EXIT_CODE}
