#!/bin/bash
# ===========================================================================
# PyGVAMP — Smoke test for main()-return-bug fix
# ===========================================================================
# Re-runs the same args as job 350 (ab42_red, lag=1, n_states=10, epochs=5)
# but with PYTHONPATH pointed at the dev repo so the fixed master_pipeline.py
# is used instead of the module's snapshot at /opt/software/pygvamp/1.0.0/source.
#
# Success criterion: Exit code 0 at the end of the pipeline.
# (Previously 350 printed "PIPELINE COMPLETED" but exited 1.)
# ===========================================================================

#SBATCH --job-name=pygv_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gputraining
#SBATCH --gres=gpu:batch:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/hdd/experiments/logs/pygv_test_%j.out
#SBATCH --error=/mnt/hdd/experiments/logs/pygv_test_%j.err

# ---- Config (matches pygv_350) --------------------------------------------
PROTEIN_NAME="ab42_red"
TOPOLOGY="/mnt/hdd/data/ab42/trajectories/red/topol.pdb"
TRAJ_DIR="/mnt/hdd/data/ab42/trajectories/red/"
SELECTION="name CA"
LAG=1
N_STATES=10
RUN_IDX=97
PRESET="medium_schnet"
EPOCHS=5
STRIDE=1
BATCH_SIZE=2048
OUTPUT_BASE="/mnt/hdd/experiments"

REPO="/home/vi/PycharmProjects/PyGVAMP"

# ---- Environment -----------------------------------------------------------
module purge
source /etc/profile.d/modules.sh
module load cuda/12.8
module load pygvamp/1.0.0

# Override: prepend dev repo so our fixed master_pipeline.py wins at import.
export PYTHONPATH="${REPO}:${PYTHONPATH}"

mkdir -p /mnt/hdd/experiments/logs

RUN_DIR=$(printf "%s/%s_std/lag%s/run_%02d" "${OUTPUT_BASE}" "${PROTEIN_NAME}" "${LAG}" "${RUN_IDX}")

# ---- Print job info --------------------------------------------------------
echo "============================================================"
echo "PyGVAMP SMOKE TEST — main() return fix"
echo "============================================================"
echo "PYTHONPATH:   ${PYTHONPATH}"
echo "pygvamp from: $(command -v pygvamp)"
echo "Python:       $(command -v python)"
echo "Protein:      ${PROTEIN_NAME}"
echo "Lag:          ${LAG}  N_states: ${N_STATES}  Epochs: ${EPOCHS}"
echo "Run:          ${RUN_IDX}"
echo "Output:       ${RUN_DIR}"
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Node:         $(hostname)"
echo "Start:        $(date)"
echo "============================================================"

# Verify our patched file is the one being imported
python -c "import pygv.pipe.master_pipeline as m; print('master_pipeline.__file__ =', m.__file__)"

pygvamp \
    --traj_dir "${TRAJ_DIR}" \
    --top "${TOPOLOGY}" \
    --lag_times "${LAG}" \
    --protein_name "${PROTEIN_NAME}" \
    --output_dir "${RUN_DIR}" \
    --preset "${PRESET}" \
    --n_states "${N_STATES}" \
    --no_discover_states \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --stride "${STRIDE}" \
    --selection "${SELECTION}" \
    --cache

EXIT_CODE=$?

echo "============================================================"
echo "Finished:     $(date)"
echo "Exit code:    ${EXIT_CODE}"
echo "Expected:     0 (was 1 before fix)"
echo "============================================================"

exit ${EXIT_CODE}
