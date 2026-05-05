#!/bin/bash
# ===========================================================================
# PyGVAMP — Ab42 Oxidized: Reversible (RevGraphVAMP)  |  SLURM Array Job
# ===========================================================================
# Usage:
#   sbatch --array=0-24 ab42_ox_reversible.sh
#   sbatch --array=0-24%2 ab42_ox_reversible.sh
# ===========================================================================

#SBATCH --job-name=ab42o_rev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gputraining
#SBATCH --gres=gpu:batch:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/ab42o_rev_%A_%a.out
#SBATCH --error=logs/ab42o_rev_%A_%a.err

# ---- EXPERIMENT CONFIGURATION ---------------------------------------------

PROTEIN_NAME="ab42_ox"
TOPOLOGY="/mnt/hdd/data/ab42/trajectories/ox/topol.gro"
TRAJ_DIR="/mnt/hdd/data/ab42/trajectories/ox/"
FILE_PATTERN="*.xtc"
SELECTION="name CA"

LAG_TIMES=(1 5 10 20 50)
N_RUNS=10

PRESET="medium_schnet"
N_STATES=10                             # TODO: update from discovery job
EPOCHS=50
STRIDE=1
BATCH_SIZE=2048

OUTPUT_BASE="/mnt/hdd/experiments"

# ---- Environment setup -----------------------------------------------------
module purge
source /etc/profile.d/modules.sh
module load 12.8
module load pygvamp/1.0.0

mkdir -p logs

# ---- Resolve (lag_time, run_index) from array task ID ----------------------
if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then
    TOTAL=$((${#LAG_TIMES[@]} * N_RUNS))
    echo "ERROR: Submit as array job:  sbatch --array=0-$((TOTAL-1)) $0"
    exit 1
fi

N_LAGS=${#LAG_TIMES[@]}
LAG_IDX=$((SLURM_ARRAY_TASK_ID / N_RUNS))
RUN_IDX=$((SLURM_ARRAY_TASK_ID % N_RUNS))

if [ ${LAG_IDX} -ge ${N_LAGS} ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} exceeds ${N_LAGS} lag times x ${N_RUNS} runs"
    exit 1
fi

LAG=${LAG_TIMES[$LAG_IDX]}
RUN_DIR=$(printf "%s/%s_rev/lag%s/run_%02d" "${OUTPUT_BASE}" "${PROTEIN_NAME}" "${LAG}" "${RUN_IDX}")

JOB_NAME="${PROTEIN_NAME}_rev_lag${LAG}_run${RUN_IDX}"
scontrol update JobId=${SLURM_JOB_ID} Name=${JOB_NAME} 2>/dev/null

# ---- Build command ---------------------------------------------------------
CMD="pygvamp"
CMD+=" --traj_dir ${TRAJ_DIR}"
CMD+=" --top ${TOPOLOGY}"
CMD+=" --lag_times ${LAG}"
CMD+=" --protein_name ${PROTEIN_NAME}"
CMD+=" --output_dir ${RUN_DIR}"
CMD+=" --reversible"
CMD+=" --preset ${PRESET}"
CMD+=" --n_states ${N_STATES}"
CMD+=" --no_discover_states"
CMD+=" --epochs ${EPOCHS}"
CMD+=" --batch_size ${BATCH_SIZE}"
CMD+=" --stride ${STRIDE}"
CMD+=" --selection '${SELECTION}'"
CMD+=" --cache"

# ---- Print job info --------------------------------------------------------
echo "============================================================"
echo "PyGVAMP — Ab42 Oxidized, Reversible (RevGraphVAMP)"
echo "============================================================"
echo "Job:          ${SLURM_JOB_ID} (array task ${SLURM_ARRAY_TASK_ID})"
echo "Protein:      ${PROTEIN_NAME}"
echo "Lag time:     ${LAG}  (index ${LAG_IDX}/${N_LAGS})"
echo "Run:          ${RUN_IDX}/${N_RUNS}"
echo "Output:       ${RUN_DIR}"
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Node:         $(hostname)"
echo "Start:        $(date)"
echo "Command:      ${CMD}"
echo "============================================================"

eval ${CMD}
EXIT_CODE=$?

echo "============================================================"
echo "Finished:     $(date)"
echo "Exit code:    ${EXIT_CODE}"
echo "============================================================"

exit ${EXIT_CODE}
