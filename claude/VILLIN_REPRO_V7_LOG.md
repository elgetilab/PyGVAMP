# VILLIN_REPRO_V7_LOG.md — Villin reproduction v7 (RBF full-reference-match probe)

Companion to `VILLIN_REPRO_LOG.md`, `VILLIN_REPRO_V2_LOG.md`,
`VILLIN_REPRO_V3_LOG.md`, `VILLIN_REPRO_V4_LOG.md`, `VILLIN_REPRO_V5_LOG.md`,
`VILLIN_REPRO_V6_LOG.md`.  Same target — Ghorbani et al. 2022 Table S1
VAMP-2 = 3.78 ± 0.02 on DE Shaw 2F4K-0.

## Where v6 (RBF range pin) landed (single seed)

| Run | seed_00 best | Δ vs v4 |
|---|---|---|
| v1 | 3.5685 | — |
| v2 | 3.6057 | — |
| v3 | 3.6124 | — |
| v4 | 3.7126 | — |
| v5 | 3.7074 | −0.005 |
| v6 | **3.6158** | **−0.097** |

v6 fell well below v4 baseline.  Pinning the RBF range to Ghorbani's
fixed `dmin=0, dmax=3` made things noticeably worse — probably because
the first ~2 reference centers (μ=0.0, 0.2 nm) sit below the physical
Cα-Cα minimum (~0.38 nm) and contribute near-zero signal, wasting 12.5%
of the basis dimensionality.

But v6 didn't fully isolate the σ formula: ours σ=(dmax-dmin)/K=3/16=0.1875
vs reference σ=step=0.2.  v7 closes that final RBF gap.

## Hypothesis under test in v7

The Ghorbani 2022 reference's `GaussianDistance` (`src/layers.py`) has:

    self.filter = torch.arange(dmin, dmax+step, step)   # 16 centers, 0.0 to 3.0
    self.var    = step                                  # = 0.2
    expand(d)   = exp(-(d - filter)^2 / var^2)

With v6's `--distance_min 0.0 --distance_max 3.0`, our centers exactly
match the reference's `arange(0, 3.2, 0.2)`.  The only remaining numerical
deviation is σ:

| Quantity | Reference | PyGVAMP v6 | PyGVAMP v7 |
|---|---|---|---|
| Centers | `arange(0, 3.2, 0.2)` | `linspace(0, 3, 16)` (≡) | `linspace(0, 3, 16)` (≡) |
| σ formula | `step` = 0.2 | `(dmax-dmin)/K` = 0.1875 | **`gaussian_var` override = 0.2** |
| Ratio σ/(center spacing) | 1.0 | 0.9375 | **1.0** |

Test `tests/test_dataset.py:test_gaussian_var_override_matches_reference`
verifies that the v7 override path reproduces the reference's expansion
output bit-for-bit (allclose, atol=1e-6).

## What v7 changes

| # | Change | v6 setting | v7 setting | Reason |
|---|---|---|---|---|
| 1 | RBF Gaussian width | σ = (dmax-dmin)/K = 0.1875 | `--gaussian_var 0.2` | Match Ghorbani 2022 σ = step |

Everything else identical to v6: data, k=4, lag=20 ns, n_neighbors=10,
hidden_dim=16, n_interactions=4, gaussian_expansion_dim=16,
**`--distance_min 0.0 --distance_max 3.0`** (kept from v6),
no attention, no pre-encoder MLP, linear softmax head, no early stopping,
batch=1000, lr=5e-4, val_split=0.3, weight_decay=1e-5, **xavier_normal
init**, encoder **v1** (no per-atom ReLU), 100 epochs.

## Probe scope: single seed (mirrors v3, v4, v5, v6)

Only seed 0.  Decision rule:

- v7 seed_00 ≲ 3.72 → **RBF is conclusively ruled out**.  Range tested
  in v6, σ in v7, both on top of v4 — no remaining RBF-side levers
  except switching to a hand-tuned partial reference match (e.g. data-
  derived range + σ=0.2, but this isn't what the reference does).
- v7 seed_00 ~3.72–3.76 → marginal; consider 3 seeds before deciding.
- v7 seed_00 ≳ 3.76 → strong → 10-seed v7 array.

If v7 < v6: σ pin actively hurts on top of the range pin (would suggest
the wider σ overshoots smoothness, blurring distance discrimination).
If v7 > v6 but < v4: σ pin partially compensates the range cost —
RBF formula matters less than range densification.
If v7 ≈ v6: σ is irrelevant; range alone explains the v6 regression.

## What v7 does NOT change

- Range pin (still 0.0 / 3.0) — v6's range cost remains baked in.
- n_neighbors=10 (reference default 5, but unrelated lever).
- Pooling head — code-level audit of reference (model.py:Linear(h_a,h_g)
  + Linear(h_g, n_classes), no activation between) confirms it is
  mathematically equivalent to a single Linear when h_g ≥ n_states.  Our
  v4-v6 head (clf_num_layers=1) already matches the reference's effective
  head structure.  Originally flagged in the v5 log as "Delta B" — the
  flag was based on layer count, not the equivalence-after-no-activation.
  No probe needed.

## CLI plumbing committed alongside v7

- `pygv/config/base_config.py`: new field `gaussian_var: Optional[float] = None`
  (default preserves current behavior — σ = (distance_max-distance_min)/K).
- `pygv/pipe/args.py`: new `--gaussian_var FLOAT` flag (units: nm).
- `pygv/pipe/master_pipeline.main`: wired `args.gaussian_var` →
  `config.gaussian_var`.  Validates `gaussian_var > 0`.
- `pygv/pipe/master_pipeline._create_prep_args` / `_create_analysis_args`:
  forward `gaussian_var` onto namespaces.  `_create_train_args` builds
  from `config.to_dict()` and picks it up automatically.
- `pygv/pipe/preparation.py`, `pygv/pipe/training.py`,
  `pygv/pipe/analysis.py`: `VAMPNetDataset(gaussian_var=...)` forwarded.
- `pygv/dataset/vampnet_dataset.py`: new constructor parameter
  `gaussian_var: Optional[float] = None`.  In
  `_compute_gaussian_expanded_distances`, σ is now
  `self.gaussian_var if self.gaussian_var is not None else (dmax-dmin)/K`.
- `tests/test_dataset.py`: extended `TestRBFAgainstGhorbaniReference`
  with `test_gaussian_var_override_matches_reference` (proves the override
  reproduces the reference exactly) and `test_gaussian_var_default_unchanged`
  (back-compat path).  7/7 passing.

## Submission

Module rebuild needed first (new CLI flag + new config field +
constructor signature change).  Then:

```
sbatch cluster_scripts/villin_repro_v7.sh
```

Single job (no `--array`).  Output:
`/mnt/hdd/experiments/villin_repro_v7/seed_00/`.  Estimated wall time:
~3.5 h on RTX 5090 (same architecture size as v4-v6).

## Outputs to compare against

Best Val VAMP-2 in:
  `/mnt/hdd/experiments/villin_repro_v7/seed_00/exp_villin_<TIMESTAMP>/logs/log_*.txt`
look for the last `New best model with score: <X.XXXX>` line.
