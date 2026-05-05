#!/bin/bash
# ===========================================================================
# PyGVAMP — Villin reproduction v5 (SchNet v2 encoder probe)
# ===========================================================================
# Probe whether the per-atom ReLU before global pooling — the only
# documented architectural delta still on the table after v4 — closes more
# of the residual gap to Ghorbani 2022's reported VAMP-2 = 3.78.
#
# Single change vs v4: --encoder_variant v2.
#
#   v4 used pygv/encoder/schnet.py:SchNetEncoderNoEmbed.  The reference
#   (github.com/ghorbanimahdi73/GraphVampNet, src/model.py:337) applies
#   nn.ReLU() to per-atom features between the residual conv loop and
#   global mean pool.  ReLU and mean-pool do not commute, so this is a
#   genuine architectural difference, not a no-op rearrangement.
#
#   v2 (pygv/encoder/schnet_v2.py:SchNetEncoderNoEmbedV2) inserts that
#   per-atom ReLU at the same position.
#
# Single seed (seed 0) — sanity probe before a 10-seed sweep.
#   v1 seed_00 best = 3.5685
#   v2 seed_00 best = 3.6057
#   v3 seed_00 best = 3.6124  (plain Adam + no jitter — ineffective)
#   v4 seed_00 best = 3.7308  (xavier_normal init)
# Decision rule:
#   v5 seed_00 ≲ 3.72 → Delta A doesn't help, abandon
#   v5 seed_00 ~3.72-3.76 → marginal; consider 3 seeds before committing
#   v5 seed_00 ≳ 3.76 → strong → proceed to 10-seed v5 array
#
# Note: --encoder_variant requires the args.py + base_config.py +
# training.py + schnet_v2.py changes committed alongside this run.
# Module rebuild needed before submission.
#
# Submit (no --array; single job):
#   sbatch cluster_scripts/villin_repro_v5.sh
#
# Timestep gotcha: DE Shaw DCD metadata reports 1 ps/frame but the actual
# physical timestep is 200 ps/frame.  --timestep 0.2 is MANDATORY.
# ===========================================================================

#SBATCH --job-name=villin_v5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gputraining
#SBATCH --gres=gpu:batch:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=6:00:00
#SBATCH --output=/mnt/hdd/experiments/logs/villin_repro_v5_%j.out
#SBATCH --error=/mnt/hdd/experiments/logs/villin_repro_v5_%j.err

# ---- Environment setup -----------------------------------------------------
module purge
source /etc/profile.d/modules.sh
module load 12.8
module load pygvamp/1.0.0

mkdir -p /mnt/hdd/experiments/logs

SEED=0
RUN_DIR=$(printf "/mnt/hdd/experiments/villin_repro_v5/seed_%02d" "${SEED}")

# ---- Job info -------------------------------------------------------------
echo "============================================================"
echo "Villin reproduction v5 (SchNet v2 encoder probe)"
echo "============================================================"
echo "Job:        ${SLURM_JOB_ID}    Seed: ${SEED}    Output: ${RUN_DIR}"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Node:       $(hostname)"
echo "Start:      $(date)"
echo "Target:     VAMP-2 = 3.78 ± 0.02 (Ghorbani 2022, Table S1)"
echo "Diff vs v4: --encoder_variant v2 (per-atom ReLU before pool)"
echo "============================================================"

# ---- Run -------------------------------------------------------------------
pygvamp \
    --traj_dir /mnt/hdd/data/villin/DESRES-Trajectory_2F4K-0-c-alpha/2F4K-0-c-alpha/ \
    --top      /mnt/hdd/data/villin/DESRES-Trajectory_2F4K-0-c-alpha/topol.pdb \
    --file_pattern '2F4K-0-c-alpha-*.dcd' \
    --protein_name villin \
    --output_dir   "${RUN_DIR}" \
    --timestep     0.2 \
    --seed         "${SEED}" \
    --model        schnet \
    --selection    'name CA' \
    --stride       1 \
    --lag_times    20.0 \
    --n_states     4 \
    --no_discover_states \
    --max_retrains 0 \
    --no_warm_start_retrains \
    --hidden_dim            16 \
    --output_dim            16 \
    --n_interactions        4 \
    --n_neighbors           10 \
    --gaussian_expansion_dim 16 \
    --no_use_attention \
    --no_use_embedding \
    --clf_num_layers 1 \
    --clf_dropout    0 \
    --clf_norm       none \
    --init_method    xavier_normal \
    --encoder_variant v2 \
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
