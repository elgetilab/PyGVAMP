#!/bin/bash
# ===========================================================================
# PyGVAMP — WW domain FiP35 (GTT), multi-lag exploration  [Category 3]
# ===========================================================================
# Goal: k*(τ) and ITS(τ) across ~3 orders of magnitude in lag time, on a system
# we have NOT published numbers for. Not a reproduction — a framework
# capability demonstration (EXPERIMENT_CHECKLIST.md Category 3).
#
# DATA (extracted + verified 2026-07-29):
#   /mnt/hdd/data/gtt/  — DESRES GTT-0 (25 dcd) + GTT-1 (33 dcd) = 58 chunks.
#   Each chunk is 100,000 frames @ 200 ps = 20 µs; last chunk of each run is
#   partial (GTT-0 tail = 32,586 frames). ~5.6M frames, ~1130 µs aggregate.
#   35 CA, sequence GSKLPPGWEKRMSRDGRVYYFNHITGTTQFERPSG.
#   topol.pdb built by cluster_scripts/mae_to_topol_pdb.py (validated by
#   regenerating trpcage's hand-built topology byte-for-byte).
#   GTT-0 and GTT-1 are SEPARATE runs — correctly loaded as independent
#   trajectories, never concatenated.
#
# ⚠️ WHY ONE JOB PER (SEED, LAG) rather than one job with --lag_times a b c:
#   state discovery runs ONCE during preparation (master_pipeline.py:107-123)
#   and yields a single recommended n_states for the whole invocation. Passing
#   all lags to one call would therefore give ONE k shared across every lag —
#   which is exactly the quantity this experiment is trying to measure. Each
#   (seed, lag) gets its own preparation + discovery so k* is free to vary.
#
# LAG LADDER: every value is an integer multiple of the 0.2 ns frame spacing.
#   Pair retention is >=90% even at 2000 ns (10,000 of 100,000 frames per
#   chunk), because lagged pairs must lie inside ONE chunk.
#
# ⚠️ COST: measure before committing. Villin (628k frames, 35 CA) took 3h53m on
#   a WHOLE gpu + 16 cpu; trpcage (1.04M frames, 20 CA) took ~30h on a shard +
#   2 cpu. Resources dominate, not dataset size. Run the PROBE first:
#     sbatch --array=5 cluster_scripts/gtt_lagsweep_v1_array.sh   # seed 0, τ=50ns
#   read the wall time, THEN size the full sweep.
#
# Full sweep (33 tasks = 3 seeds × 11 lags), 2 concurrent:
#   sbatch --array=0-32%2 cluster_scripts/gtt_lagsweep_v1_array.sh
# ===========================================================================

#SBATCH --job-name=gtt_lag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gputraining
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=INFINITE
#SBATCH --output=/mnt/hdd/experiments/logs/gtt_lag_%A_%a.out
#SBATCH --error=/mnt/hdd/experiments/logs/gtt_lag_%A_%a.err

module purge
source /etc/profile.d/modules.sh
module load cuda/12.8
module load pygvamp/1.0.0

# The deployed module lags the working tree; harmless here (this path needs no
# reversible code) but keep the hook so runs are pinned to reviewed code.
if [ -n "${PYGVAMP_SRC_OVERRIDE}" ]; then
    export PYTHONPATH="${PYGVAMP_SRC_OVERRIDE}:${PYTHONPATH}"
    echo "PYTHONPATH override active: ${PYGVAMP_SRC_OVERRIDE}"
fi

mkdir -p /mnt/hdd/experiments/logs

if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then
    echo "ERROR: submit as an array job, e.g. sbatch --array=5 $0"
    exit 1
fi

# --- (seed, lag) grid ------------------------------------------------------
LAGS=(1.0 2.0 5.0 10.0 20.0 50.0 100.0 200.0 500.0 1000.0 2000.0)
N_LAGS=${#LAGS[@]}

SEED=$(( SLURM_ARRAY_TASK_ID / N_LAGS ))
LAG=${LAGS[$(( SLURM_ARRAY_TASK_ID % N_LAGS ))]}

RUN_DIR=$(printf "/mnt/hdd/experiments/gtt_lagsweep_v1/seed_%02d/lag_%sns" "${SEED}" "${LAG}")

JOB_NAME="gtt_s${SEED}_lag${LAG}"
scontrol update JobId=${SLURM_JOB_ID} Name=${JOB_NAME} 2>/dev/null

echo "============================================================"
echo "GTT (WW domain FiP35) lag sweep — task ${SLURM_ARRAY_TASK_ID}"
echo "Seed: ${SEED}   Lag: ${LAG} ns   (ladder: ${LAGS[*]})"
echo "Out:  ${RUN_DIR}"
echo "Data: 58 chunks x 20 us, 35 CA, 200 ps/frame, ~5.6M frames"
echo "State discovery: ON (per-lag, so k* may vary with tau)"
echo "Code: $(cd /home/vi/PycharmProjects/PyGVAMP 2>/dev/null && git rev-parse --short HEAD 2>/dev/null || echo module)"
echo "Start: $(date)   Node: $(hostname)"
echo "============================================================"

# NOTE: keep this arg list free of inline '#' comments — a bare comment line
# inside a '\'-continued command TERMINATES it and silently drops every later
# flag (this bit us on ab42, commit e8dfe97).
pygvamp \
    --traj_dir /mnt/hdd/data/gtt/ \
    --top      /mnt/hdd/data/gtt/topol.pdb \
    --file_pattern '*.dcd' \
    --protein_name gtt \
    --output_dir   "${RUN_DIR}" \
    --timestep     0.2 \
    --seed         "${SEED}" \
    --model        schnet \
    --selection    'name CA' \
    --lag_times    "${LAG}" \
    --auto_stride \
    --max_retrains 0 \
    --no_warm_start_retrains \
    --hidden_dim            16 \
    --output_dim            16 \
    --n_interactions        4 \
    --n_neighbors           10 \
    --gaussian_expansion_dim 16 \
    --lr           5e-4 \
    --epochs       100 \
    --batch_size   1000 \
    --val_split    0.3 \
    --cache

EXIT_CODE=$?
echo "============================================================"
echo "Finished: $(date)   Exit: ${EXIT_CODE}"
echo "============================================================"
exit ${EXIT_CODE}
