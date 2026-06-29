#!/bin/bash
# ===========================================================================
# PyGVAMP — AT1R (d125d) Standard VAMP-2 Training (single run)
# ===========================================================================
# One standard (non-reversible) VAMPNet run on the AT1R / d125d system.
#   - preset:    large_schnet
#   - lag time:  20 ns  (cutted_dt_1ns.xtc -> 1 ns/frame)
#   - resources: full 5090 (gpu:batch:1) + 16 CPUs + 200G
#
# Run atr_d_discovery.sh FIRST to get the recommended n_states, then:
#   sbatch cluster_scripts/atr_d_experiment.sh --n_states <N>
#   # optional: --run <IDX>   (default 0)
#
# Data:      /mnt/hdd/data/julia_ATR   (recursive, d125d/cutted_dt_1ns.xtc -> 40 files)
# Topology:  .../gmm0/r1/d125d/prot_chains.pdb
# Selection: "chainid 0 and name CA"   (receptor CA only; peptide chain excluded)
# Output:    /mnt/hdd/experiments/atr_d_std/lag20/run_<IDX>
# ===========================================================================

# ---- SLURM directives ------------------------------------------------------
#SBATCH --job-name=atr_d_std
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gputraining
#SBATCH --gres=gpu:batch:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128000M
#SBATCH --time=2-00:00:00
#SBATCH --output=/mnt/hdd/experiments/logs/pygv_%j.out
#SBATCH --error=/mnt/hdd/experiments/logs/pygv_%j.err

# ---- Args: n_states (required) and run index (optional) --------------------
N_STATES=""
RUN_IDX=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --n_states) N_STATES="$2"; shift 2;;
        --run)      RUN_IDX="$2"; shift 2;;
        *)          echo "Unknown option: $1"; exit 1;;
    esac
done

if [[ -z "$N_STATES" ]]; then
    echo "ERROR: --n_states is required (read it from atr_d_discovery.sh's log)."
    echo "  sbatch $0 --n_states <N>"
    exit 1
fi

# ---- Environment setup -----------------------------------------------------
module purge
source /etc/profile.d/modules.sh
module load cuda/12.8
module load pygvamp/1.0.0

# Opt-in: run uncommitted working-tree code (e.g. the per-cluster full-atom
# structure feature) without rebuilding the shared module. No effect unless
# PYGVAMP_SRC_OVERRIDE is set.
if [ -n "${PYGVAMP_SRC_OVERRIDE}" ]; then
    export PYTHONPATH="${PYGVAMP_SRC_OVERRIDE}:${PYTHONPATH}"
    echo "PYTHONPATH override active: ${PYGVAMP_SRC_OVERRIDE}"
fi

mkdir -p /mnt/hdd/experiments/logs

# ---- Hardcoded ATR / d125d parameters --------------------------------------
PROTEIN_NAME="atr_d"
TRAJ_DIR="/mnt/hdd/data/julia_ATR"
TOPOLOGY="/mnt/hdd/data/julia_ATR/gmm0/r1/d125d/prot_chains.pdb"
FILE_PATTERN="d125d/cutted_dt_1ns.xtc"
SELECTION="chainid 0 and name CA"
PRESET="medium_schnet"
LAG=20
N_NEIGHBORS=10
EPOCHS=50
BATCH_SIZE=256   # 319-node graphs OOM'd at 2048 (large_schnet) on the 32G 5090; medium_schnet + nn10 is far lighter, 256 is safe
STRIDE=1         # full data (~20k frames). Stride 5 (runs 0-1, ~4k frames) overfit; stride 1 is the real run
RUN_DIR=$(printf "/mnt/hdd/experiments/atr_d_std/lag%s/run_%02d" "${LAG}" "${RUN_IDX}")

JOB_NAME="atr_d_std_lag${LAG}_run${RUN_IDX}"
scontrol update JobId=${SLURM_JOB_ID} Name=${JOB_NAME} 2>/dev/null

# ---- Print job info --------------------------------------------------------
echo "============================================================"
echo "PyGVAMP — AT1R (d125d) Standard VAMP-2"
echo "============================================================"
echo "Preset:       ${PRESET}"
echo "Lag time:     ${LAG} ns"
echo "N states:     ${N_STATES}"
echo "N neighbors:  ${N_NEIGHBORS}"
echo "Run:          ${RUN_IDX}"
echo "Epochs:       ${EPOCHS}"
echo "Batch size:   ${BATCH_SIZE}"
echo "Trajectories: ${TRAJ_DIR}  (pattern: ${FILE_PATTERN}, recursive)"
echo "Topology:     ${TOPOLOGY}"
echo "Selection:    ${SELECTION}"
echo "Output:       ${RUN_DIR}"
echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Node:         $(hostname)"
echo "Start:        $(date)"
echo "============================================================"

# Standard (non-reversible) VAMP-2: no --reversible flag.
# In-run state discovery is ENABLED (no --no_discover_states) so the
# preparation phase writes the Graph2Vec/clustering artifacts the interactive
# report's "prep -> VAMP state mapping" panel needs. Training still uses the
# CLI --n_states (the discovered recommendation is ignored for training because
# --n_states sets _n_states_from_cli; see master_pipeline.py:730).
pygvamp \
    --traj_dir "${TRAJ_DIR}" \
    --top "${TOPOLOGY}" \
    --file_pattern "${FILE_PATTERN}" \
    --selection "${SELECTION}" \
    --lag_times ${LAG} \
    --n_states ${N_STATES} \
    --protein_name "${PROTEIN_NAME}" \
    --output_dir "${RUN_DIR}" \
    --preset "${PRESET}" \
    --n_neighbors ${N_NEIGHBORS} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --stride ${STRIDE} \
    --cache

EXIT_CODE=$?

echo "============================================================"
echo "Finished:     $(date)"
echo "Exit code:    ${EXIT_CODE}"
echo "============================================================"

exit ${EXIT_CODE}