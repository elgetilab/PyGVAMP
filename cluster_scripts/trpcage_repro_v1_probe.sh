#!/bin/bash
# ===========================================================================
# PyGVAMP — Trp-cage reproduction v1 single-seed PROBE (1 shard, OOM check)
# ===========================================================================
# First Trp-cage (DESHAW 2JOF) run after the villin v11 array closed.
# Reuses the v11 corrected-attention architecture + dual-scoring eval
# (committed already, no module rebuild needed if the v11-era module
# is still installed).
#
# WHY A PROBE FIRST:
#
# Hugin's GPU sharding is new: --gres=shard:1 gives ~4 GB VRAM of any
# free GPU. The v11 model is small (hidden_dim=16, n_interactions=4)
# and almost certainly fits in 4 GB, but Trp-cage has different data
# shape (n_atoms=20 vs villin's 35, n_neighbors=7 vs 10) and we have
# no measurement yet. This probe runs ONE seed on shard:1 to confirm
# no OOM before the 10-seed array launches.
#
# Expected outcome:
#   - Job completes in ~2-3 h (Trp-cage is smaller than villin).
#   - Best Val concat in the neighborhood of 4.5-4.8 (paper: 4.79 ± 0.01,
#     v11 hit 3.6-3.8 on villin which paper reports as 3.78 ± 0.02 —
#     so our perbatch should land near the paper for Trp-cage too).
#   - If OOM: bump to shard:2 in the array.
#   - If far off paper: re-check k, n_neighbors, lag time, timestep
#     before launching the array.
#
# TRP-CAGE TARGET (Ghorbani et al. 2022, Table S1):
#   VAMP-2 = 4.79 ± 0.01
#
# DESHAW 2JOF DATA PREP (one-time setup before this runs):
#   cd /mnt/hdd/data/DESHAW
#   mkdir -p /mnt/hdd/data/trpcage
#   tar -xJf DESRES-Trajectory_2JOF-0-c-alpha.tar.xz -C /mnt/hdd/data/trpcage
#   # Generate topol.pdb from the .mae (mirrors what was done for villin):
#   # e.g. via Schrodinger maestro export, or:
#   #   python -c "import mdtraj as md;
#   #              t = md.load('/mnt/hdd/data/trpcage/DESRES-Trajectory_2JOF-0-c-alpha/2JOF-0-c-alpha/2JOF-0-c-alpha-000.dcd',
#   #                          top='/mnt/hdd/data/trpcage/DESRES-Trajectory_2JOF-0-c-alpha/2JOF-0-c-alpha/2JOF-0-c-alpha.mae');
#   #              t[:1].save_pdb('/mnt/hdd/data/trpcage/DESRES-Trajectory_2JOF-0-c-alpha/topol.pdb')"
#   # (Whatever produces a CA-only PDB matching the 20 atoms in the DCDs.)
#
# Hyperparameters from claude/EXPERIMENT_CHECKLIST.md (Trp-cage row):
#   k=5, lag=20 ns, n_neighbors=7, n_atoms=20, batch=1000, lr=5e-4, 10 seeds
# Architecture from v11 (corrected attention, dual-scoring eval):
#   hidden_dim=16, n_interactions=4, gauss_expansion_dim=16, use_attention,
#   no_use_embedding, clf_num_layers=1, clf_dropout=0, clf_norm=none,
#   xavier_normal init, val_split=0.3, weight_decay=1e-5, epochs=100
#
# Timestep gotcha: DESRES DCD metadata reports 1 ps/frame but the actual
# physical timestep is 200 ps/frame. --timestep 0.2 is MANDATORY.
#
# Submit:
#   sbatch cluster_scripts/trpcage_repro_v1_probe.sh
# ===========================================================================

#SBATCH --job-name=trpcage_v1_probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gputraining
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/mnt/hdd/experiments/logs/trpcage_v1_probe_%j.out
#SBATCH --error=/mnt/hdd/experiments/logs/trpcage_v1_probe_%j.err

# ---- Environment setup -----------------------------------------------------
module purge
source /etc/profile.d/modules.sh
module load 12.8
module load pygvamp/1.0.0

mkdir -p /mnt/hdd/experiments/logs

SEED=0
RUN_DIR=$(printf "/mnt/hdd/experiments/trpcage_repro_v1/seed_%02d" "${SEED}")

# ---- Job info -------------------------------------------------------------
echo "============================================================"
echo "Trp-cage reproduction v1 PROBE (1 shard, OOM check)"
echo "============================================================"
echo "Job:        ${SLURM_JOB_ID}    Seed: ${SEED}    Output: ${RUN_DIR}"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Node:       $(hostname)"
echo "Start:      $(date)"
echo "Target:     VAMP-2 = 4.79 ± 0.01 (Ghorbani 2022, Table S1)"
echo "Arch:       v11 (corrected attention + dual-scoring eval)"
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
    --model        schnet \
    --selection    'name CA' \
    --stride       1 \
    --lag_times    20.0 \
    --n_states     5 \
    --no_discover_states \
    --max_retrains 0 \
    --no_warm_start_retrains \
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
    --cache

EXIT_CODE=$?

echo "============================================================"
echo "Finished:   $(date)    Exit: ${EXIT_CODE}"
echo "============================================================"

exit ${EXIT_CODE}
