#!/bin/bash
# ===========================================================================
# PyGVAMP — Trp-cage ML3-encoder run v1 (10-seed array)
# ===========================================================================
# Third encoder in the apples-to-apples Trp-cage comparison (after SchNet and
# GIN). Same data, lag (20 ns), seed set, and training schedule as the SchNet
# reproduction; encoder = ML3, width matched as closely as the architecture
# allows.
#
# WIDTH MATCHING — IMPORTANT:
#   ML3 reads its OWN dims (ml3_*), not the generic --hidden_dim/--output_dim.
#   We set ml3_output_dim=16 / ml3_hidden_dim=16 / ml3_num_layers=4 so the
#   embedding bottleneck feeding the VAMP classifier and the depth match the
#   SchNet/GIN baselines (output=16, 4 layers). NOTE this does NOT equalise
#   total params: ML3 still carries fixed spectral machinery (nout1, recfield,
#   …) → ~46k params vs ~7k for SchNet/GIN. The control is on the embedding
#   dimension + depth (the VAMP bottleneck), not parameter count.
#
#   Exposing ml3_* on the pipeline CLI is a NEW capability (parse_pipeline_args
#   previously only wired the generic schnet/gin encoder args). REQUIRES the
#   module redeployed from the commit that adds:
#     - pygv/pipe/args.py            (--ml3_node/edge/hidden/output_dim, --ml3_num_layers)
#     - pygv/pipe/master_pipeline.py (apply those overrides to config)
#   Regression test: tests/test_config.py::TestML3PipelineCLI.
#   Until redeployed, run via the PYGVAMP_SRC_OVERRIDE hook below.
#
# ENCODER NOTES:
#   - ML3 attention is controlled by ml3_use_attention (preset default True),
#     not the generic --use_attention, so we don't pass --use_attention here.
#   - ml3_edge_dim=16 matches gaussian_expansion_dim=16 (gaussian edge mode).
#   - Other ML3-specific knobs (nout1, nout2, edge_mode, nfreq, recfield) stay
#     at the ml3 preset — no SchNet analog to match.
#
#   SchNet Trp-cage v1 : 4.6516 ± 0.0175  vs paper 4.79 ± 0.01  (Δ=-0.138)
#   GIN    Trp-cage v1 : 4.5955 ± 0.0750  vs paper 4.79 ± 0.01  (Δ=-0.195)
#   ML3    Trp-cage v1 : TBD              vs paper 4.79 ± 0.01  (Δ=?)
#
# DATA: single DESRES 2JOF (Trp-cage) Cα trajectory, 0.2 ns/frame.
#
# Submit (AFTER the module is redeployed with the ml3 CLI args):
#   sbatch --array=0-9%2 cluster_scripts/trpcage_ml3_v1_array.sh
#
# Smoke-test ONE seed on uncommitted working-tree code (before redeploy):
#   sbatch --array=0 \
#     --export=ALL,PYGVAMP_SRC_OVERRIDE=/home/vi/PycharmProjects/PyGVAMP \
#     cluster_scripts/trpcage_ml3_v1_array.sh
#
# Aggregate after completion:
#   python cluster_scripts/aggregate_trpcage_v1_array.py \
#       --root /mnt/hdd/experiments/trpcage_ml3_v1
#
# Timestep gotcha: DESRES DCD metadata reports 1 ps/frame; actual is 200
# ps/frame. --timestep 0.2 is MANDATORY.
# ===========================================================================

#SBATCH --job-name=trpcage_ml3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gputraining
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=INFINITE
#SBATCH --output=/mnt/hdd/experiments/logs/trpcage_ml3_%A_%a.out
#SBATCH --error=/mnt/hdd/experiments/logs/trpcage_ml3_%A_%a.err

# ---- Environment setup -----------------------------------------------------
module purge
source /etc/profile.d/modules.sh
module load cuda/12.8
module load pygvamp/1.0.0

# Opt-in: test uncommitted working-tree code (the ml3 CLI args) without
# touching the shared module. No effect unless PYGVAMP_SRC_OVERRIDE is set.
if [ -n "${PYGVAMP_SRC_OVERRIDE}" ]; then
    export PYTHONPATH="${PYGVAMP_SRC_OVERRIDE}:${PYTHONPATH}"
    echo "PYTHONPATH override active: ${PYGVAMP_SRC_OVERRIDE}"
fi

mkdir -p /mnt/hdd/experiments/logs

if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then
    echo "ERROR: submit as an array job, e.g. sbatch --array=0-9%2 $0"
    exit 1
fi

SEED=${SLURM_ARRAY_TASK_ID}
RUN_DIR=$(printf "/mnt/hdd/experiments/trpcage_ml3_v1/seed_%02d" "${SEED}")

JOB_NAME="trpcage_ml3_seed${SEED}"
scontrol update JobId=${SLURM_JOB_ID} Name=${JOB_NAME} 2>/dev/null

# ---- Job info -------------------------------------------------------------
echo "============================================================"
echo "Trp-cage ML3-encoder run v1 array (task ${SLURM_ARRAY_TASK_ID})"
echo "============================================================"
echo "Job:        ${SLURM_JOB_ID}    Seed: ${SEED}    Output: ${RUN_DIR}"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Node:       $(hostname)"
echo "Start:      $(date)"
echo "Target:     VAMP-2 = 4.79 ± 0.01 (Ghorbani 2022) | SchNet 4.6516 | GIN 4.5955"
echo "Encoder:    ML3 (width-matched: output=16, hidden=16, 4 layers)"
echo "GRES:       shard:1 (~4 GB VRAM)"
echo "============================================================"

# ---- Run -------------------------------------------------------------------
pygvamp \
    --traj_dir /mnt/hdd/data/trpcage/DESRES-Trajectory_2JOF-0-c-alpha/2JOF-0-c-alpha/ \
    --top      /mnt/hdd/data/trpcage/DESRES-Trajectory_2JOF-0-c-alpha/topol.pdb \
    --file_pattern '2JOF-0-c-alpha-*.dcd' \
    --protein_name trpcage \
    --output_dir   "${RUN_DIR}" \
    --timestep     0.2 \
    --seed         "${SEED}" \
    --model        ml3 \
    --selection    'name CA' \
    --stride       1 \
    --lag_times    20.0 \
    --n_states     5 \
    --no_discover_states \
    --max_retrains 0 \
    --no_warm_start_retrains \
    --ml3_node_dim   16 \
    --ml3_edge_dim   16 \
    --ml3_hidden_dim 16 \
    --ml3_output_dim 16 \
    --ml3_num_layers 4 \
    --n_neighbors           7 \
    --gaussian_expansion_dim 16 \
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
    --cache

EXIT_CODE=$?

echo "============================================================"
echo "Finished:   $(date)    Exit: ${EXIT_CODE}"
echo "============================================================"

exit ${EXIT_CODE}
