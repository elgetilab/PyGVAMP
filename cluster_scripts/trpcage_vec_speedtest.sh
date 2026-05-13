#!/bin/bash
# ===========================================================================
# PyGVAMP — Trp-cage vectorization speed test (GPU 1 / vLLM shard)
# ===========================================================================
# Validates the graph-build-vectorize branch (Tier 1 + Tier 2) against the
# 15 min/epoch baseline observed in trpcage_repro_v1_array (2 CPUs/job).
# Acceptance bar from claude/PYG_GRAPH_PREBUILD_PLAN.md:
#
#     <= 5 min/epoch at 2 CPUs (3x speedup or better)
#
# Local microbench (local_checks/bench_graph_build.py) projected
# ~4.3 min/epoch at 1 worker / ~2.2 min/epoch at 2 workers. This script
# validates that projection under realistic loader + GPU + pin_memory
# conditions.
#
# WHY ON THE vLLM GPU:
#
# The 10-seed trpcage_repro_v1 array occupies all 8 shards on GPU 0.
# GPU 1's shards are normally blocked by vLLM but become available when
# vLLM is stopped. With GPU 0 saturated, --gres=shard:1 should land on
# GPU 1 — the GPU placement check below logs the UUID for confirmation.
#
# RUNBOOK:
#   1. Stop vLLM (frees GPU 1's 8 shards):
#         systemctl stop vllm           # or whatever the service name is
#   2. Submit this job:
#         sbatch cluster_scripts/trpcage_vec_speedtest.sh
#   3. Tail the output until per-epoch times are stable (usually epoch 2+):
#         tail -F /mnt/hdd/experiments/logs/trpcage_vec_speedtest_<JOBID>.out
#   4. After 3-5 epochs, kill the job (post-training analysis isn't needed):
#         scancel <JOBID>
#   5. Restart vLLM.
#
# DESIGN CHOICES:
#   - epochs=5: enough for 3-4 stable per-epoch measurements after warmup
#   - cpus-per-task=2: matches the array exactly. This is the slow regime
#     we're trying to fix; matching it makes the speedup directly comparable.
#   - PYTHONPATH override: pygv module at /opt/software/pygvamp/1.0.0 is the
#     OLD code. We prepend the local repo so 'import pygv' resolves there.
#   - --cache: reuses the cache the running array built. The Tier 1 changes
#     preserve cache format, so old caches load fine.
#   - Unique output_dir suffix: avoids any collision with the running array.
#
# OUT OF SCOPE:
#   - Training quality. 5 epochs is too few to reach the paper VAMP-2 = 4.79.
#     This run is purely a wall-time measurement.
# ===========================================================================

#SBATCH --job-name=trpcage_vec_speedtest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=standard
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/mnt/hdd/experiments/logs/trpcage_vec_speedtest_%j.out
#SBATCH --error=/mnt/hdd/experiments/logs/trpcage_vec_speedtest_%j.err

# ---- Environment setup -----------------------------------------------------
module purge
source /etc/profile.d/modules.sh
module load 12.8
module load pygvamp/1.0.0

# Use the LOCAL repo (graph-build-vectorize branch) instead of the installed
# pygv. PYTHONPATH is honored before site-packages, so 'import pygv' resolves
# here. Verified by the import-path check below.
LOCAL_PYGV=/home/vi/PycharmProjects/PyGVAMP
export PYTHONPATH="${LOCAL_PYGV}:${PYTHONPATH}"

mkdir -p /mnt/hdd/experiments/logs

SEED=0
RUN_DIR=/mnt/hdd/experiments/trpcage_vec_speedtest/$(date +%Y%m%dT%H%M%S)_job${SLURM_JOB_ID}

# ---- Job info / sanity checks ---------------------------------------------
echo "============================================================"
echo "Trp-cage vectorization speed test (5 epochs, 2 CPUs)"
echo "============================================================"
echo "Job:        ${SLURM_JOB_ID}    Seed: ${SEED}"
echo "Output:     ${RUN_DIR}"
echo "Node:       $(hostname)"
echo "Start:      $(date)"
echo "Baseline:   ~15 min/epoch (trpcage_repro_v1_array, 2 CPUs, old code)"
echo "Target:     <= 5 min/epoch (3x speedup acceptance bar)"
echo "============================================================"
echo "GPU placement (want GPU 1 / vLLM card):"
nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.free --format=csv,noheader \
    || echo "  nvidia-smi failed"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "============================================================"
echo "Python import resolution (must point at LOCAL repo):"
python - <<'PY'
import inspect
import pygv
from pygv.dataset.vampnet_dataset import VAMPNetDataset
print("  pygv package:        ", pygv.__file__)
print("  VAMPNetDataset from: ", inspect.getsourcefile(VAMPNetDataset))
# Sanity check: the vectorized code uses _frames_t.  If we hit the old
# module, this attribute doesn't exist on the class.
has_vec = '_frames_t' in inspect.getsource(VAMPNetDataset.__init__)
print(f"  vectorized markers:  {'PRESENT' if has_vec else 'MISSING (BAD)'}")
PY
echo "============================================================"

# ---- Run -------------------------------------------------------------------
# Hyperparameters match cluster_scripts/trpcage_repro_v1_array.sh exactly
# except --epochs (5 instead of 100) so we measure per-epoch wall time
# without paying for a full training run.
time pygvamp \
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
    --epochs       5 \
    --batch_size   1000 \
    --val_split    0.3 \
    --cache

EXIT_CODE=$?

echo "============================================================"
echo "Finished:   $(date)    Exit: ${EXIT_CODE}"
echo "============================================================"
echo "How to read the result:"
echo "  - Look for 'Epoch N/5' lines in the log."
echo "  - Each epoch should print a tqdm bar and total time."
echo "  - Ignore epoch 1 (warmup, RBF cache build, kernel JIT)."
echo "  - Average epochs 2-5 for the per-epoch wall time."
echo "  - Compare to ~15 min/epoch baseline; bar is <= 5 min/epoch."
echo "============================================================"

exit ${EXIT_CODE}
