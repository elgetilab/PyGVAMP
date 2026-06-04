# NTL9_REPRO_V2_LOG.md — NTL9 reproduction v2 (resource-budget fix for v1 OOM)

Pure resource-bump successor to v1. Architecture, hyperparameters, and
data unchanged — see [NTL9_REPRO_V1_LOG.md](NTL9_REPRO_V1_LOG.md) for
those.

## Why v2 exists

v1 (job 539, submitted 2026-05-16 00:11) **OOM-killed all 10 seeds**
at the "Computing distance range" step, before any epoch ran. Each
seed wrote a partial 876 MB cache and was killed seconds into the
distance-range sampler.

Kernel `oom_kill` logs on hugin (May 16 00:20–00:21) show the
working set sitting right at the 16 GB cgroup boundary, not blowing
miles past it:

| seed | anon-rss at kill | total RSS | cgroup limit |
|---|---|---|---|
| 539_0 | 15.08 GB | 15.40 GB | 16 GB |
| 539_1 | 14.98 GB | 15.30 GB | 16 GB |
| 539_5 | 14.98 GB | 15.30 GB | 16 GB |

(`total-vm` of ~32 GB is mostly mdtraj's file-backed mmaps of the DCDs
— not resident.)

**Root cause:** the v1 log estimate ("~5.5M frames") was off. Actual
data load is **14.7M frames × 39 Cα atoms**; the distance-range
sampler holds the cached coord array (~6.9 GB float32) AND computes
pairwise distances on top, peaking near 15 GB resident.

No `seff` available and SLURM accounting storage is off, so post-hoc
`sacct MaxRSS` won't recover the peak — the kernel oom-killer log was
the only source.

## What changed from v1

Only the SLURM resource directives:

| | v1 | v2 |
|---|---|---|
| `--gres` | `shard:1` (~4 GB VRAM) | `shard:2` (~8 GB VRAM) |
| `--cpus-per-task` | 2 | 4 |
| `--mem` | 16 G | 32 G |
| `--array` throttle | `%8` | `%4` |

Each knob independently throttles concurrency to 4 on the current
partition (8 shards / 2; 16-CPU cap / 4; 128 GB cap / 32), so they're
consistent. The duplication is intentional — explicit and survives
future partition-limit changes.

**Per `feedback_per_experiment_presets`:** these are SLURM-script
knobs, not a new preset class. Model architecture and training
hyperparameters are unchanged from v1.

**VRAM note:** shard:2 ≈ 8 GB VRAM. The 16/16/4-layer SchNet at
batch=1000 fit comfortably in shard:1's ~4 GB, so the extra VRAM is
pure overhead — no benefit, no problem.

## Expected wall time

- Per-epoch: ~10–12 min at 4 CPUs (vs Trp-cage v1's ~2.7 min/epoch at
  970 batches/epoch and 2 CPUs; NTL9 has ~3850 train batches/epoch
  and 2× the CPUs).
- Per-seed: 100 epochs → ~17–20 h.
- 10 seeds × 3 waves at %4 (4, 4, 2) → **~50–60 h end-to-end**
  (vs v1's projected ~35–40 h at %8 — the concurrency drop costs us
  ~15–25 h).

## Submission

```bash
sbatch --array=0-9%4 cluster_scripts/ntl9_repro_v2_array.sh
```

## Aggregation

The v1 aggregator works fine, just point `--root`:

```bash
python cluster_scripts/aggregate_ntl9_v1_array.py \
    --root /mnt/hdd/experiments/ntl9_repro_v2 \
    --csv  /mnt/hdd/experiments/ntl9_repro_v2/summary.csv
```

## Outstanding questions to watch

1. **Did the 32 GB cap hold?** The 15 GB peak was from the
   distance-range step. If anything *later* in the pipeline (graph
   building, validation epoch over the full cached set) has a higher
   peak, we may see another OOM further in. The kernel oom log will
   say so.
2. **Did 4 CPUs actually help per-epoch time?** Vectorized graph build
   releases the GIL for numpy/PyG hot paths, so a 20–40% speedup is
   plausible. >2× would be surprising.
3. **Cleanup:** v1 left ~9 GB of partial caches across
   `/mnt/hdd/experiments/ntl9_repro_v1/seed_*/`. Not deleting until v2
   completes — if v2 fails for an unrelated reason, that's still our
   only record of the v1 load behavior.

## Result

All 10 seeds trained to 100 epochs (job 552, 2026-05-18 → 05-28).
Numbers parsed from the training logs by
`aggregate_ntl9_v1_array.py --root /mnt/hdd/experiments/ntl9_repro_v2`.
The headline VAMP-2 number does **not** depend on the analysis phase, so
the v2 analysis OOM (below) does not affect it.

### Headline numbers (cross-seed, n=10)

- Best concat: **4.4644 ± 0.0756**
- perbatch_mean @ best-concat epoch: **4.3459 ± 0.0435**
- Δ vs paper (4.59 ± 0.09): **−0.2441**  (2.7σ in paper's σ, 5.6σ in ours)

### Per-seed table

| seed | epoch | best concat | perbatch_mean | perbatch_std | status |
|---|---|---|---|---|---|
| 0 | 78 | 4.4594 | 4.3574 | 0.3722 | ok |
| 1 | 99 | 4.5121 | 4.3673 | 0.4266 | ok |
| 2 | 98 | 4.5098 | 4.3797 | 0.4140 | ok |
| 3 | 94 | 4.5147 | 4.3810 | 0.4116 | ok |
| 4 | 84 | 4.5056 | 4.3727 | 0.4074 | ok |
| 5 | 59 | 4.3263 | 4.2727 | 0.3133 | ok |
| 6 | 92 | 4.4846 | 4.3249 | 0.4511 | ok |
| 7 | 82 | 4.4944 | 4.3611 | 0.4179 | ok |
| 8 | 86 | 4.5139 | 4.3762 | 0.4010 | ok |
| 9 | 96 | 4.3229 | 4.2660 | 0.3178 | ok |

best/worst concat: seed 3 = 4.5147 / seed 9 = 4.3229.
best/worst perbatch_mean: seed 3 = 4.3810 / seed 9 = 4.2660.

### Verdict

**Outside both σ — clear miss** (like Trp-cage, and further off:
Δ=−0.244 vs Trp-cage's −0.138). NTL9 joins the gap-investigation lane;
do not treat as reproduced.

---

## OOM correction (the v2 32 GB bump worked, but exposed a *different* OOM)

The "Outstanding questions" above guessed the next OOM risk was a later
*prep/graph-build/validation* step. That guess was wrong. What actually
happened (all 10 seeds, `exit 137` *after* training completed, empty
`analysis/` dirs):

- v2's 32 GB / 4 CPU bump **did** fix the v1 prep/distance-range OOM —
  prep + all 100 training epochs ran fine on every seed.
- The kill came **inside `run_training` itself**, in its post-training
  analysis block (`pygv/pipe/training.py`): it rebuilt a frame loader over
  the **full ~11.8M-frame split with no subsampling** and ran inference
  over all of it, peaking past 32 GB — *before* the pipeline's PHASE 3
  analysis (`analysis.py`, which already caps at `analysis_max_frames=50k`)
  ever ran. This is **not** the prep OOM and **not** the analysis-phase
  loader; it is a separate latent bug that only large systems trip
  (villin/Trp-cage were small enough to survive it).

### Recovering the analysis (2026-06-04)

- **Fix 1** — `training.py` frame loader now subsamples to
  `analysis_max_frames` (mirrors `analysis.py`); regression test in
  `tests/test_training.py::TestFrameLoaderSubsampling`.
- **Fix 2** — found during the re-run: `generate_state_structures`
  (`pygv/utils/analysis.py`) discovered trajectories with **non-recursive**
  globs, so NTL9's nested DESRES layout was missed → analysis aborted at
  "Generating state structures" (`No trajectory files found`). Now uses the
  recursive, pattern-aware `find_trajectory_files`; regression test in
  `tests/test_analysis.py::TestStateStructureTrajectoryDiscovery`.
- **Re-run** — `cluster_scripts/ntl9_repro_v2_analysis_array.sh` runs the
  pipeline `--only_analysis --resume` per seed over the saved models
  (skips training, reuses the 7 GB prep cache, no OOM). Seed_00 dry-run
  (job 591) confirmed the approach before Fix 2; full 10-seed re-run after
  the module is redeployed with both fixes.
