# VILLIN_REPRO_V3_LOG.md — Villin reproduction v3 (single-seed probe)

Companion to `VILLIN_REPRO_LOG.md` and `VILLIN_REPRO_V2_LOG.md`.  Same target
— Ghorbani et al. 2022 Table S1 VAMP-2 = 3.78 ± 0.02 on DE Shaw 2F4K-0.

## Where v2 landed (10 seeds)

| Stat | v1 | v2 |
|---|---|---|
| Mean Val VAMP-2 | 3.611 | **3.663** |
| Stdev | 0.053 | 0.053 |
| Range | [3.545, 3.737] | [3.606, 3.754] |
| Seeds inside paper ±0.05 window | 0/10 | 2/10 (seeds 02, 07) |
| Gap to 3.78 | −0.169 | **−0.117** |

v2 closed about half the gap by removing the pre-encoder MLP, the classifier
dropout/BN, and early stopping.  Two seeds reached the paper band, so the
architecture *can* hit 3.78 — most seeds settle ~0.12 below.

## What v3 changes (and why)

| # | Change | v2 setting | v3 setting | Reason |
|---|---|---|---|---|
| 1 | Optimizer | AdamW, `weight_decay=1e-5` | `--weight_decay 0` (plain Adam math) | Paper says Adam; AdamW with WD=0 is mathematically identical to `torch.optim.Adam` |
| 2 | Training-time jitter | `training_jitter=1e-6` | `--training_jitter 0.0` | N(0,σ) noise added to node features each forward pass; not in paper |

Note on activation: paper uses **tanh** for the activation following the
residual connection (and is ambiguous about the filter-generating network).
Our run already uses tanh.  Not changed.

Everything else identical to v2: data, k=4, lag=20 ns, n_neighbors=10,
hidden_dim=16, n_interactions=4, gaussian_expansion_dim=16, no attention,
no pre-encoder MLP, linear softmax head, no early stopping, batch=1000,
lr=5e-4, val_split=0.3, 100 epochs.

## Probe scope: single seed

Only seed 0 — sanity check before a full 10-seed sweep.  v1 seed_00 best
= 3.5685, v2 seed_00 best = 3.6057.  Decision rule:

- v3 seed_00 ≲ 3.62 → these levers don't help, abandon and look elsewhere
  (init scheme, data prep, n_neighbors=10 vs paper-actual k-NN choice).
- v3 seed_00 ~3.62–3.68 → marginal; consider 3 seeds before deciding.
- v3 seed_00 ≳ 3.68 → strong signal, proceed to 10-seed v3 array.

## CLI plumbing committed alongside v3

`pygv/pipe/args.py` and `pygv/pipe/master_pipeline.main` gained:

- `--training_jitter <float>` — overrides `config.training_jitter`.

Plain Adam needed no new flag: existing `--weight_decay 0` does it (decoupled
WD term in AdamW becomes the zero map).

## Submission

Module rebuild needed first (new CLI flag).  Then:

```
sbatch cluster_scripts/villin_repro_v3.sh
```

Single job (no `--array`).  Output: `/mnt/hdd/experiments/villin_repro_v3/seed_00/`.
Estimated wall time: ~3.5 h on RTX 5090 (same as v2 per-seed).

## Outputs to compare against

Best Val VAMP-2 in:
  `/mnt/hdd/experiments/villin_repro_v3/seed_00/exp_villin_<TIMESTAMP>/logs/log_*.txt`
look for the last `New best model with score: <X.XXXX>` line.
