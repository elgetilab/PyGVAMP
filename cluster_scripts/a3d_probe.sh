#!/bin/bash
# ===========================================================================
# PyGVAMP — alpha3D (A3D) PROBE: one seed, SchNet, discovery + retrain loop
# ===========================================================================
# PURPOSE. Before committing a two-encoder arm on a system 3.6x larger than
# anything we have run, measure three things that have each burned us before:
#   1. k*  — is alpha3D actually richer than two-state? Every system so far
#            (trpcage/villin/ntl9/GTT) collapsed toward a two-state picture, and
#            GTT converged at k*=2 across a 50x range in tau. If alpha3D also
#            gives k*=2 there is little for ANY encoder to resolve and the
#            comparison arm should be reconsidered.
#   2. wall time — three GTT estimates were wrong before one was measured.
#   3. memory / stability at 73 CA — NTL9 (39 CA) is where GIN went unstable and
#            ML3 CUDA-OOM'd 9/10. This is nearly twice that size.
#
# DATA (extracted + verified 2026-08-23):
#   /mnt/hdd/data/a3d/  -> symlinks to A3D-0 (18 dcd) and A3D-1 (19 dcd),
#   37 chunks, 73 CA, 200 ps/frame, 1,732,710 frames in A3D-0 alone (~346 us);
#   ~3.47M frames total (~694 us).
#   Topology from the REDUCED structure file inside each DCD directory
#   (A3D-0-c-alpha.mae), NOT system.mae — the latter is the full solvated system
#   (25,921 atoms) and the converter happily produces garbage from it.
#   A3D-0 and A3D-1 topologies differ only in REMARK lines and reference
#   coordinates; the residue sequence is identical, so one topol.pdb serves both.
#   They are SEPARATE runs and must never be concatenated (the loader keeps
#   per-file trajectory boundaries, same as GTT-0/GTT-1).
#
# ⚠️ --timestep 0.2 IS MANDATORY. times.csv shows chunk starts at 200.0 and
#   20,000,200.0 ps => 200 ps/frame. DESRES DCD metadata reports 1 ps and would
#   put every lag off by 200x.
#
# LAG / STRIDE. stride 10 -> effective dt 2.0 ns; tau=50 ns = 25 frames, exactly
#   divisible (the divisibility trap that killed job 861 and, on the analysis
#   path, tau=50 in job 877). ~347k frames after stride, comparable to GTT's 568k.
#
# WHY max_retrains 10, NOT the default 5: GTT tau=1 EXHAUSTED a cap of 5 while
#   still descending, so k* was never measured and the run had to be repeated.
#   alpha3D starts at k=10 and may need a long descent.
#
# Submit:
#   sbatch cluster_scripts/a3d_probe.sh
# ===========================================================================

#SBATCH --job-name=a3d_probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gputraining
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=INFINITE
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/mnt/hdd/experiments/logs/a3d_probe_%j.out
#SBATCH --error=/mnt/hdd/experiments/logs/a3d_probe_%j.err

# shard:2 and 96G are deliberately generous for a probe: this is 73 CA vs NTL9's
# 39, and NTL9 is where ML3 CUDA-OOM'd at shard:2. Measure the real footprint
# here, then right-size the arms rather than guessing.

module purge
source /etc/profile.d/modules.sh
module load cuda/12.8
module load pygvamp/1.0.0

if [ -n "${PYGVAMP_SRC_OVERRIDE}" ]; then
    export PYTHONPATH="${PYGVAMP_SRC_OVERRIDE}:${PYTHONPATH}"
    echo "PYTHONPATH override active: ${PYGVAMP_SRC_OVERRIDE}"
fi

mkdir -p /mnt/hdd/experiments/logs

if ! pygvamp --help 2>&1 | grep -q -- '--exp_name'; then
    echo "ERROR: the pygvamp on PATH lacks --exp_name (resume support)."
    echo "       Redeploy the module or set PYGVAMP_SRC_OVERRIDE."
    exit 1
fi

SEED=0
LAG=50.0
RUN_DIR=/mnt/hdd/experiments/a3d_probe
EXP_NAME=$(printf "exp_a3d_schnet_s%02d_lag%s" "${SEED}" "${LAG}")

echo "============================================================"
echo "alpha3D PROBE — SchNet, seed ${SEED}, tau ${LAG} ns"
echo "============================================================"
echo "Out:       ${RUN_DIR}/${EXP_NAME}"
echo "Data:      37 dcd (A3D-0 + A3D-1), 73 CA, 200 ps/frame, ~3.47M frames"
echo "k:         discovery ON, start 10 -> JSD retrain loop (max 10, warm-start)"
echo "Measuring: k*, wall time, peak memory at 73 CA"
echo "Code:      $(cd /home/vi/PycharmProjects/PyGVAMP 2>/dev/null && git rev-parse --short HEAD 2>/dev/null || echo module)"
echo "Attempt:   ${SLURM_RESTART_COUNT:-0}   Start: $(date)   Node: $(hostname)"
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
    --model        schnet \
    --selection    'name CA' \
    --lag_times    "${LAG}" \
    --stride       10 \
    --auto_stride \
    --n_states     10 \
    --max_retrains 10 \
    --hidden_dim            16 \
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
echo "⚠️  Exit 0 does NOT mean results exist. Verify:"
echo "    analysis_completed non-empty in ${RUN_DIR}/${EXP_NAME}/pipeline_summary.json"
echo "    and grep the log for 'retrain loop exhausted'"
echo "============================================================"
exit ${EXIT_CODE}
