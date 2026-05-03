# VILLIN_REPRO_V5_LOG.md — Villin reproduction v5 (SchNet v2 encoder probe)

Companion to `VILLIN_REPRO_LOG.md`, `VILLIN_REPRO_V2_LOG.md`,
`VILLIN_REPRO_V3_LOG.md`, `VILLIN_REPRO_V4_LOG.md`.  Same target —
Ghorbani et al. 2022 Table S1 VAMP-2 = 3.78 ± 0.02 on DE Shaw 2F4K-0.

## Where v4 (xavier_normal init) landed (10 seeds)

| Stat | v1 | v2 | v4 |
|---|---|---|---|
| Mean Val VAMP-2 | 3.611 | 3.663 | **3.686** |
| Stdev | 0.053 | 0.053 | 0.045 |
| Seeds inside paper ±0.05 window | 0/10 | 2/10 | 3/10 |
| Gap to 3.78 | −0.169 | −0.117 | **−0.094** |

v4 closed +0.052 over v2 → +0.075 total over v1.  Real but bimodal: 8/10
seeds beat their v2 counterpart, but only 3/10 land inside the paper
window.  Architectural levers exposed via the CLI are exhausted — what
remains is in the encoder code itself.

## Hypothesis under test in v5

A diff against the Ghorbani 2022 reference
(`github.com/ghorbanimahdi73/GraphVampNet`, `src/model.py`) surfaced one
clear architectural delta still present in v4: the reference applies
`nn.ReLU()` to per-atom features between the residual conv loop and the
global mean pool (line 337, `self.conv_activation = nn.ReLU()`).  Our v1
encoder pools directly without this activation.

ReLU and mean-pool **do not commute** — applying ReLU per-atom before the
pool gates negative contributions independently before averaging, which
produces a different aggregate than thresholding the pooled vector.  In a
small network (35 atoms × hidden=16) this is plausibly load-bearing.

## What v5 changes

| # | Change | v4 setting | v5 setting | Reason |
|---|---|---|---|---|
| 1 | SchNet encoder variant | `pygv/encoder/schnet.py` (no post-loop activation) | `pygv/encoder/schnet_v2.py` (per-atom ReLU before pool) | Match Ghorbani 2022 reference `model.py:337` |

Everything else identical to v4: data, k=4, lag=20 ns, n_neighbors=10,
hidden_dim=16, n_interactions=4, gaussian_expansion_dim=16, no attention,
no pre-encoder MLP, linear softmax head, no early stopping, batch=1000,
lr=5e-4, val_split=0.3, weight_decay=1e-5, **xavier_normal init**, 100 epochs.

## Probe scope: single seed (mirrors v3, v4)

Only seed 0.  Decision rule (slightly tightened — v4 already cleared 3.68):

- v5 seed_00 ≲ 3.72 → Delta A isn't enough to clear the next bar; abandon
  this lever and look at the pooling-head structure (Delta B from the
  comparison report) or RBF range.
- v5 seed_00 ~3.72–3.76 → marginal; consider 3 seeds before deciding.
- v5 seed_00 ≳ 3.76 → strong signal that the per-atom ReLU was the
  remaining piece, proceed to 10-seed v5 array.

## CLI plumbing committed alongside v5

- New file `pygv/encoder/schnet_v2.py` with `SchNetEncoderNoEmbedV2`
  (subclass of v1 overriding `forward` with the ReLU insertion).
- New file `pygv/encoder/SCHNET_VERSIONS.md` documenting v1/v2 deltas.
- `pygv/config/base_config.py`: new field `encoder_variant: str = "v1"`
  (default preserves current behavior).
- `pygv/pipe/args.py`: new `--encoder_variant {v1,v2}` flag.
- `pygv/pipe/master_pipeline.main`: wired `args.encoder_variant` →
  `config.encoder_variant`.
- `pygv/pipe/training.py`: encoder construction site dispatches to
  `SchNetEncoderNoEmbedV2` when `args.encoder_variant == 'v2'`.

## Submission

Module rebuild needed first (new CLI flag + new file + new config field).
Then:

```
sbatch cluster_scripts/villin_repro_v5.sh
```

Single job (no `--array`).  Output:
`/mnt/hdd/experiments/villin_repro_v5/seed_00/`.  Estimated wall time:
~3.5 h on RTX 5090 (same architecture size as v4).

## Outputs to compare against

Best Val VAMP-2 in:
  `/mnt/hdd/experiments/villin_repro_v5/seed_00/exp_villin_<TIMESTAMP>/logs/log_*.txt`
look for the last `New best model with score: <X.XXXX>` line.
