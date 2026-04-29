# VILLIN_REPRO_V4_LOG.md — Villin reproduction v4 (single-seed init probe)

Companion to `VILLIN_REPRO_LOG.md`, `VILLIN_REPRO_V2_LOG.md`,
`VILLIN_REPRO_V3_LOG.md`.  Same target — Ghorbani et al. 2022 Table S1
VAMP-2 = 3.78 ± 0.02 on DE Shaw 2F4K-0.

## Where v3 left us (single-seed probe of plain Adam + no jitter)

| Run | seed_00 best | Δ vs prior |
|---|---|---|
| v1 | 3.5685 | — |
| v2 | 3.6057 | +0.037 |
| v3 | 3.6124 | +0.007 |

v3's +0.007 is within run-to-run noise (v2 stdev 0.053). Optimizer and
input-jitter levers are **not** the explanation for the residual ~0.12 gap.
Verdict noted in v3 log: abandon those levers.

## Hypothesis under test in v4

Most v2 seeds settle at ~3.60–3.66 while 2/10 (seeds 02, 07) escape to ~3.75
— suggests an **init/landscape problem**, not a capacity problem.

Current init at `pygv/utils/nn_utils.py:211` calls
`init_weights(method='kaiming_normal', nonlinearity='relu')`.  Kaiming was
designed for ReLU's half-rectified output (scaling factor √2).  Our pipeline
uses **tanh** as the activation (and the paper specifies tanh post-residual).
Kaiming+ReLU init on a tanh network over-scales by √2, pushing pre-
activations into tanh's saturating tails (|x| large → tanh→±1 → vanishing
gradients).  Xavier (Glorot 2010) is the textbook init for sigmoid/tanh
symmetric saturating activations.

## What v4 changes

| # | Change | v2 setting | v4 setting | Reason |
|---|---|---|---|---|
| 1 | Weight init | `kaiming_normal` (with `nonlinearity='relu'`) | `xavier_normal` (with gain=0.8) | Activation is tanh; Kaiming over-scales by √2 → saturating-tail trap |

v3's reverts (plain Adam, no jitter) are **rolled back** to v2 settings so
the init effect is tested in isolation:
- `--weight_decay 1e-5` (back to v2/v3 default)
- `--training_jitter` not overridden → 1e-6 (v2 default)

Everything else identical to v2: data, k=4, lag=20 ns, n_neighbors=10,
hidden_dim=16, n_interactions=4, gaussian_expansion_dim=16, no attention,
no pre-encoder MLP, linear softmax head, no early stopping, batch=1000,
lr=5e-4, val_split=0.3, 100 epochs.

## Probe scope: single seed (mirrors v3)

Only seed 0.  Decision rule:

- v4 seed_00 ≲ 3.62 → init isn't the issue, abandon and look elsewhere
  (next suspect: SchNet implementation differences vs the paper's reference
  code — edge construction, message-passing details, interaction-block
  structure).
- v4 seed_00 ~3.62–3.68 → marginal; consider 3 seeds before deciding.
- v4 seed_00 ≳ 3.68 → strong signal that init was the issue, proceed to
  10-seed v4 array.

## CLI plumbing committed alongside v4

- `pygv/config/base_config.py`: new field `init_method: str = "kaiming_normal"`
  (default preserves current behavior).
- `pygv/pipe/args.py`: new `--init_method` flag with `choices=['xavier_normal',
  'xavier_uniform', 'kaiming_normal', 'kaiming_uniform', 'orthogonal',
  'normal', 'uniform']` (the set already supported by `init_weights`).
- `pygv/pipe/master_pipeline.main`: wired `args.init_method` → `config.init_method`.
- `pygv/pipe/training.py:331`: `init_for_vamp(model, method=args.init_method)`
  (was hardcoded `'kaiming_normal'`).

## Submission

Module rebuild needed first (new CLI flag + new config field).  Then:

```
sbatch cluster_scripts/villin_repro_v4.sh
```

Single job (no `--array`).  Output: `/mnt/hdd/experiments/villin_repro_v4/seed_00/`.
Estimated wall time: ~3.5 h on RTX 5090.

## Outputs to compare against

Best Val VAMP-2 in:
  `/mnt/hdd/experiments/villin_repro_v4/seed_00/exp_villin_<TIMESTAMP>/logs/log_*.txt`
look for the last `New best model with score: <X.XXXX>` line.
