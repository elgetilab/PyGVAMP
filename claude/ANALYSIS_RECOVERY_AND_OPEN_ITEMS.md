# Analysis OOM recovery + open items (2026-06-04)

Handoff note. Captures what changed in the analysis pipeline this session,
the recovered NTL9 v2 results, and — most importantly — **what still needs
to be addressed properly** before/after the next run. See also
[NTL9_REPRO_V2_LOG.md](NTL9_REPRO_V2_LOG.md).

---

## TL;DR

- NTL9 v2 trained fine (10 seeds) but **all 10 OOM-killed in analysis**. Two
  distinct bugs caused it; both are now **fixed, tested, committed
  (`4a280d8`), and deployed** to `/opt/software/pygvamp/1.0.0/source`
  (verified byte-identical to HEAD).
- The OOMed analysis was **re-run for all 10 seeds** (`--only_analysis`) and
  is now complete (transition matrices, diagnostics, state attention,
  populations, ITS, CK, state structures — ~100k files/seed, all exit 0).
- Headline reproduction result (from training logs, independent of analysis):
  **VAMP-2 = 4.346 ± 0.044 vs paper 4.59 → clear miss, Δ = −0.244.**
- Cross-seed analysis aggregated (n=10): see numbers below.
- **One real methodological issue remains open** (ITS/CK on a non-contiguous
  subsample — not time-calibrated). Details in "Open items".

---

## What changed (commit `4a280d8`)

Two latent bugs that only large / nested-layout systems trip:

1. **In-training analysis OOM** — `pygv/pipe/training.py`. After saving the
   model, `run_training` rebuilt a frame loader over the *full* split and ran
   inference over all frames with no cap, OOM-killing NTL9 (14.7M frames)
   *before* the pipeline's PHASE 3 analysis. **Fix:** the `is_frame_loader`
   path now subsamples to `analysis_max_frames` (deterministic, seed 42),
   mirroring `analysis.py`. Added `--analysis_max_frames` CLI knob
   (`pygv/args/args_train.py`).
   Test: `tests/test_training.py::TestFrameLoaderSubsampling`.

2. **Non-recursive trajectory lookup** — `pygv/utils/analysis.py`.
   `generate_state_structures` globbed `traj_dir/*.dcd` etc. non-recursively,
   missing NTL9's nested DESRES layout (`DESRES-Trajectory_*/*/*.dcd`) and
   aborting analysis at "Generating state structures" with "No trajectory
   files found". **Fix:** new `_discover_trajectory_files` uses the recursive,
   pattern-aware `find_trajectory_files`; `file_pattern` threaded through from
   `pygv/pipe/analysis.py:399`.
   Test: `tests/test_analysis.py::TestStateStructureTrajectoryDiscovery`.

Test status: 145 passed (the one failure, `test_training_with_meta`, is a
pre-existing `torch_scatter` env issue, also fails on the deployed module).

### How the re-run was done (no retraining)

`cluster_scripts/ntl9_repro_v2_analysis_array.sh` runs the pipeline
`--only_analysis --resume <exp_<ts>>` per seed over the saved models, reusing
the 7 GB prep cache. Jobs: 591 (seed_00 dry-run, hit bug 2), 592 (seed_00
full, via `PYGVAMP_SRC_OVERRIDE` before deploy), 593 (seeds 1–9, post-deploy).
All exit 0; analysis dirs uniform (~100k files each, CK + ITS + 5 state dirs).

Re-running the full pipeline from scratch is **not** needed — the headline
VAMP-2 number comes from the training logs and is already recovered.

---

## Cross-seed analysis results (n=10, lag 200 ns, 5 states)

Produced by `cluster_scripts/aggregate_ntl9_v2_analysis.py` (a thin driver
that reuses `for_publication/paper_analysis.py` — Hungarian state-matching +
mean/CI aggregation — over the `seed_NN/` layout). Outputs in
`/mnt/hdd/experiments/ntl9_repro_v2/cross_seed_analysis/`.

### State populations (mean ± SD)

| state | population |
|---|---|
| 4 | 83.10% ± 0.04% |
| 5 | 15.96% ± 0.16% |
| 3 | 0.60% ± 0.09% |
| 2 | 0.20% ± 0.06% |
| 1 | 0.14% ± 0.01% |

**Finding:** the 5-state model collapses to ~2 populated states (≈83% + ≈16%)
on every seed; states 1–3 hold <1% combined. Matches the per-seed diagnostic
("RETRAIN with ~2 states") and is consistent with the weak VAMP-2 score.

### Implied timescales (mean ± SD, ns)

| mode | ITS @ τ=1000 ns |
|---|---|
| t₂ | 315.9 ± 5.5 |
| t₃ | 255.0 ± 8.8 |
| t₄ | 178.0 ± 48.8 |
| t₅ | 136.1 ± 26.0 |

⚠️ Absolute ITS values are **not reliable** — see open item #1.

---

## Open items — to address properly

### 1. ITS/CK are computed on a non-contiguous subsample (MAIN issue)

Analysis runs on a 50k-frame *random* subsample of 14.7M frames (the
`analysis_max_frames` cap that fixes the OOM). The subsample is time-ordered
but **not contiguous** (~295-frame / ~59 ns average gap). ITS/CK nonetheless
treat consecutive rows as 0.2 ns apart, so the lag→frame→time mapping is
broken.

- **Populations are fine** (uniform sample → unbiased) — trust those.
- **ITS/CK absolute timescales are not time-calibrated.** Cross-seed
  *consistency* (small SD) is real; the nanosecond magnitudes are not.
- This affects **every** system the pipeline analyzes (villin/Trp-cage too),
  not just NTL9 — it predates this session.
- **Proper fix options:** (a) compute ITS/CK by running inference over
  *contiguous* trajectory segments (chunked), not the random subsample;
  (b) subsample as contiguous blocks; or (c) at minimum, label ITS/CK plots
  as subsample-derived and not use them quantitatively. Pin behavior with a
  test first.

### 2. `paper_analysis.py:load_run_data` can't read the transition-matrix CSV

The pipeline writes `*_transition_matrix_all_lag*ns.csv` pandas-style (header
row + row labels); `np.loadtxt` chokes on it. The field is **loaded but never
used**, so it's only an incidental crash. Worked around with a tolerant-load
shim inside `aggregate_ntl9_v2_analysis.py`. Proper fix: make `load_run_data`
parse the labeled CSV (skip header + label column), or drop the unused load.
Test-first.

### 3. `analysis_max_frames` default (50k)

`base_config.py:98` defaults to 50k (comment says the paper "report uses 5K").
Worth deciding the right cap for production analysis once #1 is resolved — the
contiguity fix may change what's appropriate.

### 4. Uncommitted / cleanup

- `cluster_scripts/aggregate_ntl9_v2_analysis.py` — **uncommitted** (new
  driver). Commit alongside the others if keeping.
- `cluster_scripts/ntl9_repro_v2_analysis_array.sh` carries an opt-in
  `PYGVAMP_SRC_OVERRIDE` hook (committed; harmless, inactive unless the env
  var is set). Used to test uncommitted code on the cluster before deploy.
  Strip it if you don't want it shipped.

---

## Pointers

- Reproduction status: [NTL9_REPRO_V2_LOG.md](NTL9_REPRO_V2_LOG.md)
- Re-run analysis: `cluster_scripts/ntl9_repro_v2_analysis_array.sh`
  (`sbatch --array=0-9%4 ...`, no override needed — module is deployed)
- Aggregate analysis: `cluster_scripts/aggregate_ntl9_v2_analysis.py`
- Aggregate VAMP score (training logs): `cluster_scripts/aggregate_ntl9_v1_array.py --root /mnt/hdd/experiments/ntl9_repro_v2`
- Experiment data: `/mnt/hdd/experiments/ntl9_repro_v2/seed_*/exp_*/`
- Deployed module source: `/opt/software/pygvamp/1.0.0/source` (redeploy via
  `sudo git archive HEAD | tar -x -C ...` after committing)
