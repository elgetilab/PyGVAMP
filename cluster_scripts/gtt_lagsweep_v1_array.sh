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
# ⚠️ HOW k IS DETERMINED — start at 10, let the retrain loop reduce it.
#   Corrected 2026-07-30 after two wrong turns, both recorded so they are not
#   repeated:
#
#   (1) `--max_states 25` was WRONG. It was raised from 10 to stop the discovery
#       METRICS pinning at their ceiling — but `recommended_n_states` also feeds
#       the TRAINED model, so every run trained k=24/25 on a 35-residue protein,
#       which is not physically defensible. Raising the cap did not fix the
#       censoring either: BIC/AIC are monotone in k and simply re-pinned at 25.
#       Discovery's max-across-metrics rule is degenerate here regardless of cap.
#
#   (2) `--max_retrains 0 --no_warm_start_retrains` was WRONG for this
#       experiment. Those flags were copied from the REPRODUCTION scripts, where
#       pinning k is required because the paper fixes it. This is Category 3
#       exploration, where `_run_retrain_loop` is precisely the capability being
#       demonstrated: train at k, run the JSD diagnostic (jsd_threshold 0.05) to
#       find redundant `merge_groups` + underpopulated states, retrain at the
#       recommended `effective_n_states`, and repeat until the diagnostic points
#       at the current model. Warm-start reuses the encoder across rounds.
#
#   So: start every lag at k=10 with discovery OFF, and let the diagnostic walk
#   k down. The converged k per lag IS k*(tau), measured from trained models
#   rather than from a clustering heuristic — which is what this sweep is for.
#
# ⚠️ WHY ONE JOB PER (SEED, LAG) rather than one job with --lag_times a b c:
#   state discovery runs ONCE during preparation (master_pipeline.py:107-123)
#   and yields a single recommended n_states for the whole invocation. Passing
#   all lags to one call would therefore give ONE k shared across every lag —
#   which is exactly the quantity this experiment is trying to measure. Each
#   (seed, lag) gets its own preparation + discovery so k* is free to vary.
#
# LAG LADDER: every value is an integer multiple of the 0.2 ns frame spacing.
#   Pair retention is >=90% at every rung, because lagged pairs must lie inside
#   ONE 20 us chunk.
#
# ⚠️ LADDER CAPPED AT 500 ns (decision 2026-07-29). The cap is NOT about pair
#   retention or cost — it is about TRAINING-SET SIZE. Auto-stride targets ~10
#   frames per lag, so the dataset shrinks as tau grows:
#     tau      prep_stride  runtime_stride  training frames
#     1 ns          5             1           1,137,344
#     10 ns        10             1             568,672
#     50 ns        10             2             284,336   <-- probe, confirmed
#     100 ns       10             5             113,734
#     500 ns       10            25              22,747
#    (1000 ns      10            50              11,373)  dropped
#    (2000 ns      10           100               5,687)  dropped
#   At 1000-2000 ns only ~4k-8k samples survive the 70/30 split, which is too
#   thin to fit a 35-CA graph net AND run state discovery — k* there would be
#   noise, at exactly the end of the ladder a k*(tau) curve is read from.
#   500 ns keeps >=22k training frames everywhere while still spanning nearly
#   three orders of magnitude.
#
# ⚠️ COST: measure before committing. Villin (628k frames, 35 CA) took 3h53m on
#   a WHOLE gpu + 16 cpu; trpcage (1.04M frames, 20 CA) took ~30h on a shard +
#   2 cpu. Resources dominate, not dataset size. Run the PROBE first:
#     sbatch --array=5 cluster_scripts/gtt_lagsweep_v1_array.sh   # seed 0, τ=50ns
#   read the wall time, THEN size the full sweep.
#
# ⚠️ REQUEUE SURVIVAL (added 2026-08-03 after the job 877 post-mortem).
#   hugin hard-crashed 4x between Jul 30 15:30 and Jul 31 07:25 (no shutdown
#   sequence in the journal, no I/O / MCE / OOM / Xid signature), then was powered
#   off Fri->Mon. It is NOT a time limit: the partition is MaxTime=UNLIMITED with
#   preemption off. SLURM requeues the tasks, but before this change each attempt
#   started a fresh exp_gtt_<timestamp> directory and retrained from epoch 0, so
#   11 attempts across 4 lags produced zero usable results.
#
#   Three flags make a requeued attempt continue instead of restart:
#     --exp_name        deterministic per-(seed,lag) directory -> reuses the prep
#                       cache, and skips training/analysis that already FINISHED
#     --resume_training continues an interrupted model from resume_state.pt
#                       (optimizer moments, scheduler, epoch, RNG), not epoch 0
#     --save_every 10   how often resume_state.pt is refreshed; a crash now costs
#                       at most 10 epochs
#
#   "Finished" is decided by marker files (training_complete.json /
#   analysis_complete.json), NOT by the presence of best_model.pt — that file is
#   written on every validation improvement from epoch 1, so an interrupted run
#   leaves one behind and would otherwise be accepted as a finished model.
#
# Full sweep (27 tasks = 3 seeds × 9 lags), 2 concurrent:
#   sbatch --array=0-26%2 cluster_scripts/gtt_lagsweep_v1_array.sh
# ===========================================================================

#SBATCH --job-name=gtt_lag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gputraining
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=INFINITE
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/mnt/hdd/experiments/logs/gtt_lag_%A_%a.out
#SBATCH --error=/mnt/hdd/experiments/logs/gtt_lag_%A_%a.err

# ⚠️ --open-mode=append is REQUIRED, not cosmetic. Without it every requeue
#   TRUNCATES the .out/.err, so the crash evidence from earlier attempts is gone.
#   Job 877 task 0 was requeued 4 times and left exactly one "Start:" banner —
#   which is why the Jul 30/31 crashes had to be reconstructed from journalctl.
# ⚠️ --requeue is explicit so a node crash re-queues the task rather than failing it.

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

# PREFLIGHT: the resume flags below were added 2026-08-03 and do NOT exist in the
# deployed pygvamp/1.0.0 module. Without them argparse rejects the whole command in
# ~2 s and every array task dies instantly. Fail here with a readable message rather
# than as an unrecognized-arguments dump 27 times.
if ! pygvamp --help 2>&1 | grep -q -- '--exp_name'; then
    echo "ERROR: the pygvamp on PATH does not support --exp_name/--resume_training."
    echo "       Either redeploy the module, or rerun with:"
    echo "         PYGVAMP_SRC_OVERRIDE=/home/vi/PycharmProjects/PyGVAMP sbatch ... $0"
    echo "       (without these flags a requeued task restarts from epoch 0 — the"
    echo "        exact failure this script was changed to prevent)"
    exit 1
fi

if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then
    echo "ERROR: submit as an array job, e.g. sbatch --array=5 $0"
    exit 1
fi

# --- (seed, lag) grid ------------------------------------------------------
# STRIDE IS EXPLICIT PER LAG, and must be (corrected 2026-07-30, job 861).
# The pipeline does NOT adapt the prep stride to the lag: it applies the preset
# stride (10) and then REJECTS any lag that is not a multiple of the resulting
# effective timestep (0.2 ns x 10 = 2 ns), erroring out in ~2 s:
#     "Invalid lag times: 1.0 ns -> closest valid: 0.0 ns"
# So tau=1 and tau=5 failed on every seed. For each lag we pass the LARGEST
# stride <= 10 whose effective timestep divides it:
#     tau      stride  eff dt   lag in frames   cache frames
#     1 ns        5     1.0 ns        1          1,137,344
#     2 ns       10     2.0 ns        1            568,672
#     5 ns        5     1.0 ns        5          1,137,344
#     10-500 ns  10     2.0 ns      5..250         568,672
# (An earlier note here predicted the pipeline would pick these strides itself
#  via _select_compatible_stride. It does not on this path — hence the explicit
#  table. The two 1.13M-frame rungs cost roughly 2x the others.)
LAGS=(5.0   10.0 20.0 50.0)
STRIDES=(5   10   10   10)
N_LAGS=${#LAGS[@]}

SEED=$(( SLURM_ARRAY_TASK_ID / N_LAGS ))
LAG=${LAGS[$(( SLURM_ARRAY_TASK_ID % N_LAGS ))]}
STRIDE=${STRIDES[$(( SLURM_ARRAY_TASK_ID % N_LAGS ))]}

RUN_DIR=$(printf "/mnt/hdd/experiments/gtt_lagsweep_v1/seed_%02d/lag_%sns" "${SEED}" "${LAG}")

# Deterministic experiment directory. Every requeue of this (seed, lag) lands in
# the SAME directory, so the prep cache (0.2-0.5 GB), any finished training and any
# finished analysis are reused instead of rebuilt. Without --exp_name the pipeline
# mints exp_gtt_<timestamp> per attempt: that is why job 877 burned 11 attempts
# across 4 lags and produced zero usable results.
EXP_NAME=$(printf "exp_gtt_s%02d_lag%s" "${SEED}" "${LAG}")

JOB_NAME="gtt_s${SEED}_lag${LAG}"
scontrol update JobId=${SLURM_JOB_ID} Name=${JOB_NAME} 2>/dev/null

echo "============================================================"
echo "GTT (WW domain FiP35) lag sweep — task ${SLURM_ARRAY_TASK_ID}"
echo "Seed: ${SEED}   Lag: ${LAG} ns   Stride: ${STRIDE}   (ladder: ${LAGS[*]})"
echo "Out:  ${RUN_DIR}/${EXP_NAME}"
echo "Data: 58 chunks x 20 us, 35 CA, 200 ps/frame, ~5.6M frames"
echo "k: discovery ON, start 10 -> JSD retrain loop reduces it (max 5 rounds, warm-started)"
echo "Code: $(cd /home/vi/PycharmProjects/PyGVAMP 2>/dev/null && git rev-parse --short HEAD 2>/dev/null || echo module)"
echo "Attempt: ${SLURM_RESTART_COUNT:-0}   Start: $(date)   Node: $(hostname)"
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
    --stride       "${STRIDE}" \
    --auto_stride \
    --n_states     10 \
    --max_retrains 5 \
    --hidden_dim            16 \
    --output_dim            16 \
    --n_interactions        4 \
    --n_neighbors           10 \
    --gaussian_expansion_dim 16 \
    --lr           5e-4 \
    --epochs       100 \
    --batch_size   1000 \
    --val_split    0.3 \
    --exp_name     "${EXP_NAME}" \
    --resume_training \
    --save_every   10 \
    --cache

EXIT_CODE=$?
echo "============================================================"
echo "Finished: $(date)   Exit: ${EXIT_CODE}"
echo "============================================================"
exit ${EXIT_CODE}
