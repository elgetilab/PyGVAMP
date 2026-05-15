#!/bin/bash
# ===========================================================================
# PyGVAMP — NTL9 reproduction v1 array (10-seed, vectorized + dual-scoring)
# ===========================================================================
# Third reproduction system after villin v11 and Trp-cage v1. Same v11
# architecture (corrected attention + dual-scoring eval). First array
# that benefits from the graph-build vectorization merged in PR #10 —
# expected ~5x per-epoch speedup vs the Trp-cage array baseline.
#
# WHY 10 SEEDS:
#
# Ghorbani 2022 reports NTL9 VAMP-2 = 4.59 ± 0.09 (Table S1), averaged
# across 10 trainings. The paper's σ on NTL9 is noticeably looser than
# on villin (± 0.02) and Trp-cage (± 0.01), which gives us more room
# before we'd call this "not parity". Across-system pattern so far:
#
#   Villin v11  : ours 3.6923 ± 0.0458 vs paper 3.78 ± 0.02 (Δ=-0.088, 1.9σ ours)
#   Trp-cage v1 : ours 4.6516 ± 0.0175 vs paper 4.79 ± 0.01 (Δ=-0.138, 7.9σ ours)
#   NTL9 v1     : TBD                     vs paper 4.59 ± 0.09  (Δ=?)
#
# Both prior systems undershoot the paper in the same direction. NTL9's
# wider paper σ means a similar Δ could still land inside paper's bar,
# which would be informative.
#
# DATA SCOPE: all four DESHAW NTL9-{0,1,2,3} trajectories combined for
# the strict 1.11 ms total (paper-comparable). 149 DCD files total
# (56+54+20+19), 39 Cα atoms each, 200 ps/frame.
#
# Wall time per seed: ~5.5M frames × val_split=0.3 → ~3850 train
# batches/epoch at batch=1000. With the vectorized graph build at
# ~2.7 min/epoch on Trp-cage's 970 batches/epoch (2 CPUs), NTL9 scales
# linearly with batch count → ~10-12 min/epoch → ~17-20h per seed.
#
# PARALLELISM: shard:1 per seed, throttle %8 (8 concurrent on GPU 0
# shards). Wall time estimate: first wave ~17-20h, seeds 8-9 roll
# forward as wave 1 finishes → total ~35-40h end-to-end.
#
# GPU 1's 8 shards are blocked by vLLM 06:00-02:00, so we stay on
# GPU 0 to avoid colliding with vLLM's restart window. Same call as
# Trp-cage array.
#
# CPU/MEM: 2 CPUs/job, 16 GB/job. With the new 16-CPU gputraining cap
# (raised from 8 in HuginSLURM/config/slurm.conf), 8 jobs × 2 CPUs = 16.
# 128 GB partition cap / 8 = 16 GB each. NTL9 data is ~5× bigger than
# Trp-cage's (5.5M vs 1M frames × 39 vs 20 atoms) but the cached frame
# coords are still only ~2.6 GB — should fit comfortably.
#
# MODULE: requires the rebuilt pygvamp/1.0.0 (with PR #10 vectorization).
# Verify with:
#   /opt/software/pygvamp/1.0.0/conda_env/bin/python -c \
#       "import inspect; from pygv.dataset.vampnet_dataset import VAMPNetDataset; \
#        print('vectorized' if 'for i in range(self.n_atoms)' not in \
#        inspect.getsource(VAMPNetDataset._create_graph_from_frame) else 'OLD CODE')"
#
# Submit:
#   sbatch --array=0-9%8 cluster_scripts/ntl9_repro_v1_array.sh
#
# Aggregate after completion:
#   python cluster_scripts/aggregate_ntl9_v1_array.py \
#       --csv /mnt/hdd/experiments/ntl9_repro_v1/summary.csv
#
# Timestep gotcha: DESRES DCD metadata reports 1 ps/frame but the actual
# physical timestep is 200 ps/frame. --timestep 0.2 is MANDATORY.
# ===========================================================================

#SBATCH --job-name=ntl9_v1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gputraining
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=INFINITE
#SBATCH --output=/mnt/hdd/experiments/logs/ntl9_v1_%A_%a.out
#SBATCH --error=/mnt/hdd/experiments/logs/ntl9_v1_%A_%a.err

# ---- Environment setup -----------------------------------------------------
module purge
source /etc/profile.d/modules.sh
module load cuda/12.8
module load pygvamp/1.0.0

mkdir -p /mnt/hdd/experiments/logs

if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then
    echo "ERROR: submit as an array job, e.g. sbatch --array=0-9%8 $0"
    exit 1
fi

SEED=${SLURM_ARRAY_TASK_ID}
RUN_DIR=$(printf "/mnt/hdd/experiments/ntl9_repro_v1/seed_%02d" "${SEED}")

JOB_NAME="ntl9_v1_seed${SEED}"
scontrol update JobId=${SLURM_JOB_ID} Name=${JOB_NAME} 2>/dev/null

# ---- Job info -------------------------------------------------------------
echo "============================================================"
echo "NTL9 reproduction v1 array (task ${SLURM_ARRAY_TASK_ID})"
echo "============================================================"
echo "Job:        ${SLURM_JOB_ID}    Seed: ${SEED}    Output: ${RUN_DIR}"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Node:       $(hostname)"
echo "Start:      $(date)"
echo "Target:     VAMP-2 = 4.59 ± 0.09 (Ghorbani 2022, Table S1)"
echo "Arch:       v11 (corrected attention + dual-scoring eval, vectorized)"
echo "Data:       NTL9-{0,1,2,3} combined (1.11 ms, 149 DCDs, 39 Cα atoms)"
echo "GRES:       shard:1 (~4 GB VRAM)"
echo "============================================================"

# ---- Run -------------------------------------------------------------------
# --traj_dir is the parent containing all 4 trajectory directories;
# default recursive glob walks them. --file_pattern matches any of the
# four trajectory naming conventions (NTL9-0-c-alpha-NNN.dcd through
# NTL9-3-c-alpha-NNN.dcd).
pygvamp \
    --traj_dir /mnt/hdd/data/ntl9/ \
    --top      /mnt/hdd/data/ntl9/topol.pdb \
    --file_pattern 'NTL9-*-c-alpha-*.dcd' \
    --protein_name ntl9 \
    --output_dir   "${RUN_DIR}" \
    --timestep     0.2 \
    --seed         "${SEED}" \
    --model        schnet \
    --selection    'name CA' \
    --stride       1 \
    --lag_times    200.0 \
    --n_states     5 \
    --no_discover_states \
    --max_retrains 0 \
    --no_warm_start_retrains \
    --hidden_dim            16 \
    --output_dim            16 \
    --n_interactions        4 \
    --n_neighbors           10 \
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
