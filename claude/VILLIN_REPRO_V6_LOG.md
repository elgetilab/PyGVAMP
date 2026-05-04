# VILLIN_REPRO_V6_LOG.md — Villin reproduction v6 (RBF range fix probe)

Companion to `VILLIN_REPRO_LOG.md`, `VILLIN_REPRO_V2_LOG.md`,
`VILLIN_REPRO_V3_LOG.md`, `VILLIN_REPRO_V4_LOG.md`, `VILLIN_REPRO_V5_LOG.md`.
Same target — Ghorbani et al. 2022 Table S1 VAMP-2 = 3.78 ± 0.02 on DE Shaw 2F4K-0.

## Where v5 (encoder v2 per-atom ReLU) landed (single seed)

| Run | seed_00 best | Δ vs v4 |
|---|---|---|
| v1 | 3.5685 | — |
| v2 | 3.6057 | — |
| v3 | 3.6124 | — |
| v4 | 3.7126 | — |
| v5 | **3.7074** | **−0.005** |

v5 fell below v4's seed_00 baseline (3.7126) and below the v5 decision
threshold of 3.72.  Verdict noted in v5 log: per-atom ReLU before mean-pool
is **not** the missing piece.  Architectural levers around the encoder body
are exhausted — the next suspects flagged in the v5 log were **RBF range**
and **pooling-head structure**.  v6 tests the cheaper of the two.

v6 is built on the **v4** baseline (xavier_normal init, encoder v1).  v5's
per-atom ReLU is dropped.

## Hypothesis under test in v6

A code-level diff against the Ghorbani 2022 reference
(`github.com/ghorbanimahdi73/GraphVampNet`) audited the Gaussian RBF basis:

| Param | Reference | PyGVAMP v1–v5 |
|---|---|---|
| Formula | `exp(-(d-μ)²/var²)` | `exp(-(d-μ)²/σ²)` |
| K | 16 | 16 |
| dmin | **0.0 nm fixed** (`args.py:--dmin 0.`) | **data-derived** (~0.38 nm villin) |
| dmax | **3.0 nm fixed** (`args.py:--dmax 3.`) | **data-derived** (~3.0–3.5 nm villin) |
| Center placement | `arange(0, 3.2, 0.2)` | `linspace(dmin, dmax, K)` |
| σ (var) | **= step = 0.2** | **= (dmax-dmin)/K** ≈ 0.16–0.22 |

Findings, pinned in `tests/test_dataset.py:TestRBFAgainstGhorbaniReference`:

1. **Centers match exactly** when dmin=0, dmax=3, K=16:
   `linspace(0, 3, 16)` ≡ `arange(0, 3.2, 0.2)`.  No deviation here.
2. **σ formula differs by K/(K-1) = 16/15 = 1.067**: with the same range,
   ours σ = 0.1875 vs reference σ = 0.2.  Ours' Gaussians are ~6.7%
   narrower → less overlap between adjacent centers → coarser smoothness
   in the distance encoding.  This is **not** fixable with the v6 CLI
   knobs — would require exposing σ separately.
3. **Range** is data-derived in v1–v5 but fixed in the reference.  For
   villin Cα-only: data-derived dmin ≈ 0.38 nm (excludes the bottom 2
   reference centers from any meaningful contribution); data-derived dmax
   tracks the populated max distance.

Of the three deltas, only the **range** is exposed via the CLI overrides
(`--distance_min`, `--distance_max`) committed for v6.  v6 tests whether
matching the reference's *fixed* basis range — including the 2 "dead"
centers below the physical Cα-Cα minimum — closes the residual gap.

## What v6 changes

| # | Change | v4 setting | v6 setting | Reason |
|---|---|---|---|---|
| 1 | RBF basis range | data-derived per-protein | `--distance_min 0.0 --distance_max 3.0` (nm) | Match Ghorbani 2022 hardcoded basis |

Everything else identical to v4: data, k=4, lag=20 ns, n_neighbors=10,
hidden_dim=16, n_interactions=4, gaussian_expansion_dim=16, no attention,
no pre-encoder MLP, linear softmax head, no early stopping, batch=1000,
lr=5e-4, val_split=0.3, weight_decay=1e-5, **xavier_normal init**, encoder
**v1** (no per-atom ReLU), 100 epochs.

## What v6 does NOT change

- σ formula (still `(dmax-dmin)/K`, not `step`).  Saved as a follow-up
  lever if v6 underperforms — would need a `--gaussian_var` override.
- Pooling-head structure (single linear → softmax).  The other named
  next-suspect from the v5 log; deferred until RBF is conclusively ruled
  in or out.
- n_neighbors=10 (reference default is 5; we keep our value because v1–v5
  all used 10 and it's an unrelated lever).

## Probe scope: single seed (mirrors v3, v4, v5)

Only seed 0.  Decision rule:

- v6 seed_00 ≲ 3.72 → RBF range alone doesn't help; abandon this lever
  and look at pooling-head structure (or follow up with σ override if the
  drop is small).
- v6 seed_00 ~3.72–3.76 → marginal; consider 3 seeds before deciding.
- v6 seed_00 ≳ 3.76 → strong signal that the RBF range was the remaining
  piece (modulo σ), proceed to 10-seed v6 array.

## CLI plumbing committed alongside v6

- `pygv/config/base_config.py`: new fields `distance_min: Optional[float] = None`,
  `distance_max: Optional[float] = None` (default preserves current
  data-derived behavior).
- `pygv/pipe/args.py`: new `--distance_min` and `--distance_max` flags
  (units: nm).
- `pygv/pipe/master_pipeline.main`: wired `args.distance_{min,max}` →
  `config.distance_{min,max}`.  Validation: raises if only one is set
  (the dataset's override path requires both — silent fallback otherwise),
  and if `min >= max`.
- `pygv/pipe/master_pipeline._create_prep_args` and
  `_create_analysis_args`: forward both fields onto the namespaces.
  `_create_train_args` builds from `config.to_dict()` and picks them up
  automatically.
- `pygv/pipe/preparation.py`, `pygv/pipe/training.py`,
  `pygv/pipe/analysis.py`: `VAMPNetDataset(distance_min=..., distance_max=...)`
  forwarded.  Constructor already accepted these (vampnet_dataset.py:86-87).
- `tests/test_dataset.py`: new `TestRBFAgainstGhorbaniReference` class
  (5 tests) pinning the centers-match / σ-differs / negative-sentinel /
  end-to-end-divergence properties against the reference's defaults.

## Submission

Module rebuild needed first (new CLI flags + new config fields).  Then:

```
sbatch cluster_scripts/villin_repro_v6.sh
```

Single job (no `--array`).  Output:
`/mnt/hdd/experiments/villin_repro_v6/seed_00/`.  Estimated wall time:
~3.5 h on RTX 5090 (same architecture size as v4).

## Outputs to compare against

Best Val VAMP-2 in:
  `/mnt/hdd/experiments/villin_repro_v6/seed_00/exp_villin_<TIMESTAMP>/logs/log_*.txt`
look for the last `New best model with score: <X.XXXX>` line.

## Result (job 450, 2026-05-04 18:22 → 22:08 CEST, exit 0)

**v6 seed_00 best Val VAMP-2 = 3.6158** — set around epoch 22, then 78
consecutive epochs of no improvement (final epoch 100: train 3.6471,
val 3.5433).

| Run | seed_00 best | Δ vs v4 |
|---|---|---|
| v1 | 3.5685 | — |
| v2 | 3.6057 | — |
| v3 | 3.6124 | — |
| v4 | 3.7126 | — |
| v5 | 3.7074 | −0.005 |
| **v6** | **3.6158** | **−0.097** |

### Verdict

≲ 3.72 → **RBF range fix doesn't help, abandon this lever.**

Pinning the basis to Ghorbani's hardcoded `dmin=0, dmax=3` made things
substantially worse than v4 (data-derived range), not better.  The most
likely mechanism: the first ~2 reference centers (μ=0.0, 0.2 nm) sit
below the physical Cα-Cα minimum (~0.38 nm) and produce near-zero
features — wasting 2/16 = 12.5% of the basis dimensionality.  v4 packs
all 16 centers densely into the populated distance range and gets a
denser, more informative encoding.

This **does not rule out the σ formula deviation** (ours K/(K-1)
narrower than reference) — that lever is still untested in isolation
because v6 changed both range and σ relative-to-spacing simultaneously.

### Next probes (queued for v7 / v8)

- **v7 (RBF — full reference match)**: range 0.0/3.0 (as v6) **+** new
  `--gaussian_var 0.2` override to also fix σ to the reference's literal
  value.  If still ≲ 3.72, RBF is conclusively ruled out and we move on.
- **v8 (pooling-head structure)**: switch the classifier head to
  multi-layer MLP (the other named suspect from the v5 log).  Independent
  of RBF; safe to queue alongside v7.
