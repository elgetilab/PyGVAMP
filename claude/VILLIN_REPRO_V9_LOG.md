# VILLIN_REPRO_V9_LOG.md — Villin reproduction v9 (lag-time off-by-10 probe)

Companion to v1–v8 logs.  Same target — Ghorbani et al. 2022 Table I
VAMP-2 = 3.78 ± 0.02 on DE Shaw 2F4K-0.

## Why this probe

After v5–v8 ruled out every architectural lever exposed by the reference's
run scripts and Table I, the residual ~0.10 VAMP-2 gap had to come from
something not in the paper's published architecture spec.  My initial
hypotheses (deeptime equivalence, float32 precision, autograd through
eigh) were all checked and ruled out — the maximal divergence in the
+ε·I covariance regularization is ~0.01, an order of magnitude too small.

The user proposed an off-script hypothesis: **the paper itself may have an
off-by-10 lag-time error.**

Supporting circumstantial evidence:

- The user observed similar confusion training ab42 at τ=20 ns (poor
  convergence) vs τ=2 ns (better results).  If the same pattern holds
  on villin, the paper's "3.78 at τ=20 ns" might actually have come from
  a τ=2 ns run mislabeled.
- DESRES DCD trajectories have a documented timestep mismatch: file
  metadata reports 1 ps/frame but physical timestep is 200 ps/frame
  (200× factor — see header of every villin_repro_v*.sh).  An off-by-10
  in lag conversion is a similar class of unit-conversion error; it's
  easy to make.
- Mechanically, lower lag → less decorrelation → eigenvalues of the
  transition matrix closer to identity → VAMP-2 ≈ Σ λᵢ² mechanically
  higher.  So *of course* lag=2 ns will give a higher number than
  lag=20 ns; the question is whether the magnitude matches the paper's
  reported 3.78.

## What v9 changes

| # | Change | v4 setting | v9 setting | Reason |
|---|---|---|---|---|
| 1 | Training lag time | `--lag_times 20.0` (ns) | `--lag_times 2.0` (ns) | Test off-by-10 hypothesis |

Everything else identical to v4: data, k=4, n_neighbors=10, hidden_dim=16,
n_interactions=4, gaussian_expansion_dim=16, **--timestep 0.2** (kept —
DESRES physical timestep), no attention, no pre-encoder MLP, linear
softmax head, batch=1000, lr=5e-4, val_split=0.3, weight_decay=1e-5,
xavier_normal init, encoder v1, 100 epochs.

v6, v7, v8 modifications are NOT carried over (RBF range, σ pin, h_g
bottleneck were all conclusively ruled out).

## Probe scope: single exploratory run, single seed

Decision rule (vs v4 baseline 3.7126):

- v9 ≈ v4 (~3.71) → lag-time isn't the explanation; something else
  drives the gap.
- v9 ~3.78 → strong evidence the paper's reported number actually came
  from a τ=2 ns regime mislabeled as τ=20 ns.  Would warrant a 10-seed
  array at τ=2 ns to confirm and an ITS plot to characterize.
- v9 >> 3.78 (e.g. 3.9+) → at τ=2 ns the score is in a "too-easy"
  regime where 4 metastable states haven't separated; high VAMP-2
  reflects sub-equilibrium correlation, not real metastable structure.
  The high number doesn't validate the paper's 3.78 — it shows the
  paper's number couldn't have come from a realistic τ.

The v9 result is **only interpretable in concert with an ITS plot** at
several τ values from the same model.  This run produces only the
training-time best Val VAMP; if the hypothesis holds we'd want to
follow up with multi-τ analysis.

## What v9 does NOT change

- `--timestep 0.2` stays — that's the DESRES physical step in ns/frame
  and is independent of the choice of lag-time.
- All architecture / optimizer / data settings.

## CLI plumbing

**No new code, no rebuild.**  v9 only changes the value passed to the
existing `--lag_times` flag.

## Submission

```
sbatch cluster_scripts/villin_repro_v9.sh
```

Single job (no `--array`).  Output:
`/mnt/hdd/experiments/villin_repro_v9/seed_00/`.  Estimated wall time:
~3.5 h on RTX 5090, but lag=2 ns gives 10× more time-lagged pairs than
lag=20 ns (because pair count = n_frames - lag_frames; with lag_frames
= 10 vs 100, more pairs are valid), so the actual wall time may be
slightly longer due to larger per-epoch batch count.

## Outputs to compare against

Best Val VAMP-2 in:
  `/mnt/hdd/experiments/villin_repro_v9/seed_00/exp_villin_<TIMESTAMP>/logs/log_*.txt`
look for the last `New best model with score: <X.XXXX>` line.
