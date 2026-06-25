#!/bin/bash
# ===========================================================================
# PyGVAMP — Trp-cage ML3-encoder, NATIVE regime (v1)
# ===========================================================================
# ML3 analog of trpcage_gin_native_v1_array.sh. The matched run
# (trpcage_ml3_v1_array.sh) width-matched ML3 to the SchNet/GIN baselines
# (ml3_hidden=16, ml3_output=16, no embedding, clf 1-layer/no-norm, lr=5e-4)
# and only seed_00 ever ran (4.6431, smoke-test). Having learned that GIN's
# deficit was DE-TUNING (native GIN recovered 4.5955 → 4.6481 ≈ SchNet), this
# run gives ML3 its OWN recipe for a fair best-vs-best comparison.
#
# NATIVE (restored from the ml3 preset / ML3Config + BaseConfig defaults —
# all DEFAULTED, not passed below):
#   ml3_hidden_dim=30, ml3_output_dim=32, ml3_num_layers=4, ml3_node_dim=16,
#   ml3_edge_dim=16, ml3_nout1=30, ml3_nout2=2, ml3_use_attention=ON,
#   ml3_edge_mode=gaussian; use_embedding=ON, clf 2-layer + batch_norm,
#   init=kaiming_normal, lr=1e-3, weight_decay=1e-4.
#   (ml3_edge_dim=16 == gaussian_expansion_dim=16 by default — coupling OK.)
#   ML3 attention is ml3_use_attention (preset True) — do NOT pass --use_attention.
#
# HELD FIXED (benchmark-invariants — val VAMP-2 comparable to SchNet 4.6516):
#   data (2JOF Cα), selection 'name CA', timestep 0.2, stride 1, lag 20 ns,
#   n_states 5, no discovery, no retrains, epochs 100, val_split 0.3,
#   n_neighbors 7 (matched k-NN graph), seed set.
#
# DELIBERATE DEVIATION — batch_size:
#   ml3 preset batch=32 is INFEASIBLE at ~1.04M frames (~3 days/seed). Use
#   batch=1000 (baseline throughput). val_split held at 0.3 (not native 0.2)
#   so the held-out set matches the baselines.
#
#   SchNet Trp-cage v1 : 4.6516 ± 0.0175
#   GIN (de-tuned)     : 4.5955 ± 0.0750     GIN (native): 4.6481 ± 0.0343
#   ML3 (de-tuned)     : 4.6431 (seed 0 only)
#   ML3 (native)       : 4.5743 ± 0.0770  (n=10) ← does NOT recover to SchNet/GIN
#       parity. Lower mean AND high seed instability (worst seeds ~4.44) — about
#       where de-tuned GIN sat. Unlike GIN, native regime doesn't rescue ML3:
#       its deficit looks intrinsic (spectral/multi-layer machinery is harder to
#       optimize reliably), not just de-tuning. NB: seed 0 alone (4.6483) was a
#       lucky seed — the cross-seed sweep was needed to see the instability.
#
# DATA: single DESRES 2JOF (Trp-cage) Cα trajectory, 0.2 ns/frame.
# MODULE: deployed pygvamp/1.0.0 (ml3 CLI args confirmed present).
#
# Submit ONE seed first (verify before the full sweep):
#   sbatch --array=0 cluster_scripts/trpcage_ml3_native_v1_array.sh
# Then the rest:
#   sbatch --array=1-9%4 cluster_scripts/trpcage_ml3_native_v1_array.sh
#
# Aggregate:
#   python cluster_scripts/aggregate_trpcage_v1_array.py \
#       --root /mnt/hdd/experiments/trpcage_ml3_native_v1
#
# Timestep gotcha: DESRES DCD metadata reports 1 ps/frame; actual is 200
# ps/frame. --timestep 0.2 is MANDATORY.
# ===========================================================================

#SBATCH --job-name=trpcage_ml3_native
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gputraining
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=INFINITE
#SBATCH --output=/mnt/hdd/experiments/logs/trpcage_ml3_native_%A_%a.out
#SBATCH --error=/mnt/hdd/experiments/logs/trpcage_ml3_native_%A_%a.err

# ---- Environment setup -----------------------------------------------------
module purge
source /etc/profile.d/modules.sh
module load cuda/12.8
module load pygvamp/1.0.0

mkdir -p /mnt/hdd/experiments/logs

if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then
    echo "ERROR: submit as an array job, e.g. sbatch --array=0 $0"
    exit 1
fi

SEED=${SLURM_ARRAY_TASK_ID}
RUN_DIR=$(printf "/mnt/hdd/experiments/trpcage_ml3_native_v1/seed_%02d" "${SEED}")

JOB_NAME="trpcage_ml3_native_seed${SEED}"
scontrol update JobId=${SLURM_JOB_ID} Name=${JOB_NAME} 2>/dev/null

# ---- Job info -------------------------------------------------------------
echo "============================================================"
echo "Trp-cage ML3 NATIVE-regime run v1 array (task ${SLURM_ARRAY_TASK_ID})"
echo "============================================================"
echo "Job:        ${SLURM_JOB_ID}    Seed: ${SEED}    Output: ${RUN_DIR}"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Node:       $(hostname)"
echo "Start:      $(date)"
echo "Target:     SchNet 4.6516 | GIN native 4.6481 | ML3 native TBD"
echo "Encoder:    ML3 native (hidden=30, output=32, 4 layers, embedding ON, clf batch_norm, lr=1e-3)"
echo "Fixed:      lag 20ns, 5 states, n_neighbors 7, val_split 0.3, batch 1000 (native 32 infeasible)"
echo "============================================================"

# ---- Run -------------------------------------------------------------------
# ML3 architecture knobs (ml3_*, use_embedding, clf_*, init_method, lr,
# weight_decay) are intentionally NOT passed → they default to the ml3 preset
# (native). Only benchmark-invariants, n_neighbors (matched), val_split, and
# the practical batch are set. No --use_attention (ML3 uses ml3_use_attention).
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
    --n_neighbors  7 \
    --epochs       100 \
    --batch_size   1000 \
    --val_split    0.3 \
    --cache

EXIT_CODE=$?

echo "============================================================"
echo "Finished:   $(date)    Exit: ${EXIT_CODE}"
echo "============================================================"

exit ${EXIT_CODE}
