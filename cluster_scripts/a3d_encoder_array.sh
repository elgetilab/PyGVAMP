#!/bin/bash
# ===========================================================================
# PyGVAMP — alpha3D encoder comparison: SchNet vs PaiNN at fixed k=3
# ===========================================================================
# Category 2 (encoder comparison). SINGLE-VARIABLE SWAP: the two arms differ
# only in the encoder and its width. Every other flag is identical, which fixes
# the known weakness of the GIN/ML3 table (whole-recipe vs paper-SchNet, not
# encoder-in-isolation — see experiments/trpcage_encoders.md).
#
# WHY alpha3D. Every previously benchmarked system is effectively two-state
# (GTT converged at k*=2 across a 50x range in tau). The probe (job 921,
# 2026-08-23) descended 10->9->8->7->6->5->4->3 and CONVERGED at k*=3 with
# populations 0.469 / 0.362 / 0.169 — three genuinely occupied states, well
# inside the retrain cap. First non-two-state system in the campaign, so there
# is finally something for an encoder to resolve beyond folded/unfolded.
#
# ⚠️ k IS PINNED AT 3 HERE, and that is correct for this experiment. The GTT
#   sweep deliberately enabled the retrain loop because k*(tau) was the quantity
#   being measured. Here k must be IDENTICAL across arms or the comparison is
#   confounded — so discovery off, max_retrains 0, exactly as in the Category 1
#   reproduction scripts. k=3 comes from the probe.
#
# ⚠️ CAPACITY IS PARAMETER-MATCHED, NOT WIDTH-MATCHED, and this matters more
#   than it sounds. Encoder-only parameter counts at edge_dim 16, output_dim 16,
#   n_interactions 4:
#       node_dim=20 (trpcage):  SchNet  7,600   PaiNN@16 15,952  -> 2.10x
#       node_dim=73 (alpha3D):  SchNet 38,976   PaiNN@16 16,800  -> 0.43x
#   SchNet's count scales with node_dim (one-hot over atoms); PaiNN projects to
#   hidden_dim immediately. So "equal width" would INFLATE PaiNN by 2x on
#   trpcage and HANDICAP it by 2.3x here — a 5x swing across systems.
#   Matched setting for alpha3D: --painn_hidden_dim 26 -> 38,990 (1.000x).
#   RECOMPUTE THIS PER SYSTEM. Do not reuse 26 elsewhere.
#
# ⚠️ PaiNN HAS NO ATTENTION. --use_attention is passed to both arms so the
#   command lines stay identical; PaiNN warns and ignores it. PaiNN runs
#   therefore produce a smaller artifact set (no attention maps) — the analysis
#   phase skips them with a "SKIPPED:" line rather than dying. Do not treat the
#   missing attention PNGs as a failure.
#
# DATA: /mnt/hdd/data/a3d/ -> A3D-0 (18 dcd) + A3D-1 (19 dcd), 73 CA,
#   200 ps/frame, ~3.47M frames. SEPARATE runs, never concatenated.
#   --timestep 0.2 MANDATORY (DESRES metadata reports 1 ps; 200x error).
#   stride 10 -> eff dt 2.0 ns; tau=50 ns = 25 frames exactly (divisibility).
#
# MEASURED (probe job 921): ~12 s/epoch for SchNet, 7.7 GB GPU, analysis ~1 h.
#   At fixed k=3 each seed is ONE model + ONE analysis, so ~1.5-2 h/seed.
#   PaiNN's per-epoch cost is UNMEASURED — run one seed and read it before
#   committing the rest.
#
# Submit ONE PaiNN seed first to measure the rate:
#   A3D_ENCODER=painn sbatch --array=0 cluster_scripts/a3d_encoder_array.sh
# Then the arms:
#   A3D_ENCODER=painn  sbatch --array=1-9%4 cluster_scripts/a3d_encoder_array.sh
#   A3D_ENCODER=schnet sbatch --array=0-9%4 cluster_scripts/a3d_encoder_array.sh
# ===========================================================================

#SBATCH --job-name=a3d_enc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gputraining
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=INFINITE
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/mnt/hdd/experiments/logs/a3d_enc_%A_%a.out
#SBATCH --error=/mnt/hdd/experiments/logs/a3d_enc_%A_%a.err

module purge
source /etc/profile.d/modules.sh
module load cuda/12.8
module load pygvamp/1.0.0

if [ -n "${PYGVAMP_SRC_OVERRIDE}" ]; then
    export PYTHONPATH="${PYGVAMP_SRC_OVERRIDE}:${PYTHONPATH}"
    echo "PYTHONPATH override active: ${PYGVAMP_SRC_OVERRIDE}"
fi

mkdir -p /mnt/hdd/experiments/logs

ENCODER="${A3D_ENCODER:-schnet}"

# PREFLIGHT: --model painn and the resume flags postdate the Aug 3 module.
if ! pygvamp --help 2>&1 | grep -q -- '--exp_name'; then
    echo "ERROR: pygvamp on PATH lacks --exp_name. Redeploy the module or set PYGVAMP_SRC_OVERRIDE."
    exit 1
fi
if [ "${ENCODER}" = "painn" ] && ! pygvamp --help 2>&1 | grep -q -- '--painn_hidden_dim'; then
    echo "ERROR: pygvamp on PATH lacks --painn_hidden_dim. Redeploy the module."
    exit 1
fi

if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then
    echo "ERROR: submit as an array job, e.g. sbatch --array=0-9%4 $0"
    exit 1
fi

SEED=${SLURM_ARRAY_TASK_ID}
LAG=50.0
NSTATES=3
RUN_DIR=/mnt/hdd/experiments/a3d_encoder_v1/${ENCODER}
EXP_NAME=$(printf "exp_a3d_%s_s%02d" "${ENCODER}" "${SEED}")

# Encoder-specific flags — THE ONLY DIFFERENCE BETWEEN THE ARMS.
if [ "${ENCODER}" = "painn" ]; then
    ENC_FLAGS="--model painn --painn_hidden_dim 26 --hidden_dim 16"
    EXPECTED_PARAMS="38,990 encoder params (1.000x SchNet)"
else
    ENC_FLAGS="--model schnet --hidden_dim 16"
    EXPECTED_PARAMS="38,976 encoder params"
fi

scontrol update JobId=${SLURM_JOB_ID} Name="a3d_${ENCODER}_s${SEED}" 2>/dev/null

echo "============================================================"
echo "alpha3D encoder comparison — ${ENCODER}, seed ${SEED}"
echo "============================================================"
echo "Out:      ${RUN_DIR}/${EXP_NAME}"
echo "Fixed:    k=${NSTATES} (from probe job 921), tau=${LAG} ns, stride 10"
echo "Capacity: ${EXPECTED_PARAMS}"
echo "Encoder:  ${ENC_FLAGS}"
echo "Code:     $(cd /home/vi/PycharmProjects/PyGVAMP 2>/dev/null && git rev-parse --short HEAD 2>/dev/null || echo module)"
echo "Attempt:  ${SLURM_RESTART_COUNT:-0}   Start: $(date)   Node: $(hostname)"
echo "============================================================"

pygvamp \
    --traj_dir /mnt/hdd/data/a3d/ \
    --top      /mnt/hdd/data/a3d/topol.pdb \
    --file_pattern '*.dcd' \
    --protein_name a3d \
    --output_dir   "${RUN_DIR}" \
    --exp_name     "${EXP_NAME}" \
    --timestep     0.2 \
    --seed         "${SEED}" \
    ${ENC_FLAGS} \
    --selection    'name CA' \
    --lag_times    "${LAG}" \
    --stride       10 \
    --auto_stride \
    --n_states     "${NSTATES}" \
    --no_discover_states \
    --max_retrains 0 \
    --no_warm_start_retrains \
    --output_dim            16 \
    --n_interactions        4 \
    --n_neighbors           10 \
    --gaussian_expansion_dim 16 \
    --use_attention \
    --no_use_embedding \
    --lr           5e-4 \
    --epochs       100 \
    --batch_size   1000 \
    --val_split    0.3 \
    --resume_training \
    --save_every   10 \
    --cache

EXIT_CODE=$?

echo "============================================================"
echo "Finished: $(date)   Exit: ${EXIT_CODE}"
echo "⚠️  Exit 0 does NOT mean results exist — check analysis_completed in"
echo "    ${RUN_DIR}/${EXP_NAME}/pipeline_summary.json"
echo "============================================================"
exit ${EXIT_CODE}
