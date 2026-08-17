#!/bin/bash
# ===========================================================================
# PyGVAMP — Trp-cage ANGULAR PRE-TEST (single-variable swap vs the v1 repro)
# ===========================================================================
# PURPOSE (claude/PAINN_SCOPE.md §7). Before spending ~30 runs on PaiNN, ask the
# cheap version of the same question: does ANGULAR information move VAMP-2 at all?
#
#   GIN and ML3 changed the AGGREGATION over the same two-body Cα distance inputs.
#   Neither beat SchNet on any system (Trp-cage tie, Villin -0.10, NTL9 unstable).
#   That is consistent with the binding constraint being the INFORMATION in the
#   descriptor, not the expressiveness of the network. PaiNN is the first
#   candidate that changes the inputs — it carries directional structure a
#   distance-only representation does not contain.
#
#   This arm adds explicit angular features to the EXISTING SchNet path:
#     chain -> Cα pseudo bond angle + SIGNED pseudo dihedral. The sign carries
#              handedness, which a pairwise distance matrix provably cannot
#              represent (a structure and its mirror image have identical
#              distance matrices — pinned by tests/test_angular_features.py).
#     knn   -> Gaussian-expanded neighbour-centre-neighbour angle distribution;
#              reflection-EVEN, so angular resolution but not chirality.
#     both  -> used here, to maximise sensitivity. A null result with `both` is
#              the strongest cheap evidence that angular information does not
#              help; attribution between the two groups is only worth doing if
#              the result is positive.
#
# CONTROL ARM — DO NOT RE-RUN IT.
#   cluster_scripts/trpcage_repro_v1_array.sh, seeds 0-9, already gives
#     SchNet perbatch_mean VAMP-2 = 4.6516 ± 0.0175  (experiments/trpcage_encoders.md)
#   THIS SCRIPT IS THAT SCRIPT with exactly one line added: --angular_features both.
#   Every other flag is byte-identical, deliberately. If you edit any of them, the
#   comparison to 4.6516 is void.
#
# DETECTION THRESHOLD — know this before reading the result.
#   Baseline std 0.0175 over 10 seeds -> 95% CI half-width ~0.011. By the project's
#   standard (non-overlapping 95% CIs), this test can only resolve a shift of
#   roughly >0.022 in VAMP-2. A smaller true effect will read as null. Say so when
#   reporting: "no effect larger than ~0.02", not "no effect".
#
# KNOWN CONFOUND — MEASURED 2026-08-17, and much larger than first estimated.
#   `both` widens node features 20 -> 32. Measured parameter counts (job 905 probe
#   vs control job 522), same 56 tensors, same architecture:
#       control  node_dim=20   7,685 params
#       angular  node_dim=32  12,821 params      -> +5,136 (+67%)
#   An earlier note here guessed "~192 weights (~3%)". That was wrong by ~27x:
#   node_dim feeds more than the single first-layer projection. So this arm changes
#   information AND substantially more capacity.
#
# PRE-REGISTERED DECISION RULE (fixed before any result is read, so a positive
# cannot be rationalised after the fact):
#   * NULL (angular CI overlaps 4.6516 +- 0.011, or is lower)
#       -> conclusive for the cheap test. Even with +67% parameters AND explicit
#          angular information, nothing improved. Do not run the control arm.
#          Report as "no effect larger than ~0.02" and treat PaiNN as unmotivated.
#   * POSITIVE (angular CI clears the baseline CI upward)
#       -> NOT attributable to angular information yet. +67% params is a live
#          alternative explanation. Run the capacity control (12 columns of fixed
#          per-node random features: identical width, identical parameter count,
#          zero geometric content) for 10 seeds and compare angular vs control
#          before making any claim about angles.
#
# Caches are NOT invalidated: angular features are computed per frame at
# graph-build time, and the cache stores raw frames.
#
# Submit:
#   sbatch --array=0-9%8 cluster_scripts/trpcage_angular_pretest_array.sh
# Probe one seed first and read the wall time before committing all 10:
#   sbatch --array=0 cluster_scripts/trpcage_angular_pretest_array.sh
#
# Timestep gotcha: DESRES DCD metadata reports 1 ps/frame but the actual
# physical timestep is 200 ps/frame. --timestep 0.2 is MANDATORY.
# ===========================================================================

#SBATCH --job-name=trpcage_ang
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gputraining
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=INFINITE
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/mnt/hdd/experiments/logs/trpcage_ang_%A_%a.out
#SBATCH --error=/mnt/hdd/experiments/logs/trpcage_ang_%A_%a.err

# ---- Environment setup -----------------------------------------------------
module purge
source /etc/profile.d/modules.sh
module load cuda/12.8
module load pygvamp/1.0.0

if [ -n "${PYGVAMP_SRC_OVERRIDE}" ]; then
    export PYTHONPATH="${PYGVAMP_SRC_OVERRIDE}:${PYTHONPATH}"
    echo "PYTHONPATH override active: ${PYGVAMP_SRC_OVERRIDE}"
fi

mkdir -p /mnt/hdd/experiments/logs

# PREFLIGHT: --angular_features was added 2026-08-17 and does not exist in an
# unrefreshed module. Fail readably instead of an argparse dump x10.
if ! pygvamp --help 2>&1 | grep -q -- '--angular_features'; then
    echo "ERROR: the pygvamp on PATH has no --angular_features."
    echo "       Redeploy the module, or rerun with:"
    echo "         PYGVAMP_SRC_OVERRIDE=/home/vi/PycharmProjects/PyGVAMP sbatch ... $0"
    exit 1
fi

if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then
    echo "ERROR: submit as an array job, e.g. sbatch --array=0-9%8 $0"
    exit 1
fi

SEED=${SLURM_ARRAY_TASK_ID}
ANG_MODE="${TRPCAGE_ANGULAR_MODE:-both}"
RUN_DIR=$(printf "/mnt/hdd/experiments/trpcage_angular_pretest/seed_%02d" "${SEED}")
EXP_NAME=$(printf "exp_trpcage_ang%s_s%02d" "${ANG_MODE}" "${SEED}")

JOB_NAME="trpcage_ang_seed${SEED}"
scontrol update JobId=${SLURM_JOB_ID} Name=${JOB_NAME} 2>/dev/null

echo "============================================================"
echo "Trp-cage ANGULAR pre-test (task ${SLURM_ARRAY_TASK_ID})"
echo "============================================================"
echo "Job:        ${SLURM_JOB_ID}    Seed: ${SEED}    Output: ${RUN_DIR}/${EXP_NAME}"
echo "Angular:    ${ANG_MODE}   (control arm: trpcage_repro_v1, 4.6516 +- 0.0175)"
echo "Resolves:   shifts >~0.022 in VAMP-2 only (non-overlapping 95% CI, n=10)"
echo "Code:       $(cd /home/vi/PycharmProjects/PyGVAMP 2>/dev/null && git rev-parse --short HEAD 2>/dev/null || echo module)"
echo "Attempt:    ${SLURM_RESTART_COUNT:-0}   Start: $(date)   Node: $(hostname)"
echo "============================================================"

# ---- Run -------------------------------------------------------------------
# Identical to trpcage_repro_v1_array.sh except --angular_features (+ the resume
# flags, which do not affect the science). Keep it that way.
pygvamp \
    --traj_dir /mnt/hdd/data/trpcage/DESRES-Trajectory_2JOF-0-c-alpha/2JOF-0-c-alpha/ \
    --top      /mnt/hdd/data/trpcage/DESRES-Trajectory_2JOF-0-c-alpha/topol.pdb \
    --file_pattern '2JOF-0-c-alpha-*.dcd' \
    --protein_name trpcage \
    --output_dir   "${RUN_DIR}" \
    --exp_name     "${EXP_NAME}" \
    --timestep     0.2 \
    --seed         "${SEED}" \
    --model        schnet \
    --selection    'name CA' \
    --stride       1 \
    --lag_times    20.0 \
    --n_states     5 \
    --no_discover_states \
    --max_retrains 0 \
    --no_warm_start_retrains \
    --angular_features "${ANG_MODE}" \
    --hidden_dim            16 \
    --output_dim            16 \
    --n_interactions        4 \
    --n_neighbors           7 \
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
    --resume_training \
    --save_every   10 \
    --cache

EXIT_CODE=$?

echo "============================================================"
echo "Finished:   $(date)    Exit: ${EXIT_CODE}"
echo "============================================================"

exit ${EXIT_CODE}
