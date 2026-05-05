# VILLIN_REPRO_V8_LOG.md — Villin reproduction v8 (h_g rank-bottleneck head probe)

Companion to `VILLIN_REPRO_LOG.md`, `VILLIN_REPRO_V2_LOG.md`,
`VILLIN_REPRO_V3_LOG.md`, `VILLIN_REPRO_V4_LOG.md`, `VILLIN_REPRO_V5_LOG.md`,
`VILLIN_REPRO_V6_LOG.md`, `VILLIN_REPRO_V7_LOG.md`.  Same target —
Ghorbani et al. 2022 Table S1 VAMP-2 = 3.78 ± 0.02 on DE Shaw 2F4K-0.

## Where v6 and v7 (RBF probes) landed

| Run | seed_00 best | Δ vs v4 |
|---|---|---|
| v1 | 3.5685 | — |
| v2 | 3.6057 | — |
| v3 | 3.6124 | — |
| v4 | 3.7126 | — |
| v5 | 3.7074 | −0.005 |
| v6 (range pin) | 3.6158 | −0.097 |
| v7 (range + σ pin) | 3.7005 | −0.012 |

v6 and v7 conclusively ruled out the RBF basis (range and σ together)
as a source of the residual gap — neither helps over v4 at the args.py-
default RBF values.  Architectural levers tested so far have all been
either neutral or harmful.

Crucially: **v4 already matches every villin hyperparameter listed in
Table I of Ghorbani 2022** (graph layers=4, neurons=16, clusters=4,
batch=1000, lr=5e-4, atoms=35, neighbors=10, Gaussians=16).  So the
residual ~0.10 gap **must** come from something not tabulated.

Levers not in Table I but visible in the reference's run scripts /
abstract / source:

- **`h_g` (post-pool projection dim)** — TrpCage's `gpu_1.sh` uses h_g=2.
  Paper abstract: "graph embeddings in the last layer of GraphVAMPNet
  were transformed into 2D and trained by maximizing the VAMP-2 score"
  → strong signal h_g=2 is uniform across systems including villin.
- **Train/val split methodology** — not documented; could be a real
  methodological gap.

v8 tests the first.

## Hypothesis under test in v8

The reference's classifier head (`model.py:fc_classes` and `amino_emb`)
is:

```
pool [B, h_a=16] -> [Linear(16, h_g=2)] -> Linear(h_g=2, n_classes) -> softmax
```

with **no activation between the two Linears**.  Two unactivated linears
compose to a single Linear of rank ≤ h_g — a deliberate **rank-2
bottleneck**.

Our v4–v7 head is `Linear(16, 4) -> softmax` (full rank-4).  The
reference's effective head is `Linear(16, 4)` of rank ≤ 2 — strictly
less expressive on paper, but counter-intuitively this can HELP in
VAMPNet:

- The slow processes in MD are typically 1–3 dimensional.  A rank-2
  bottleneck aligns the model's effective capacity with the actual
  dimensionality of the slow modes.
- It strongly regularizes against the model spending capacity on
  fast/noisy modes that don't contribute to slow-process resolution.

If villin's slow mode structure is ≤ 2D (4 metastable states often
arranged in a roughly 2D landscape), the rank-2 head should match or
beat v4's full-rank head.

## What v8 changes

| # | Change | v4 setting | v8 setting | Reason |
|---|---|---|---|---|
| 1 | Classifier head | `--clf_num_layers 1` (Linear(16, 4)) | `--clf_num_layers 2 --clf_hidden_dim 2` (Linear(16, 2) → Linear(2, 4), rank ≤ 2) | Match Ghorbani 2022 h_g=2 |

Everything else identical to v4: data, k=4, lag=20 ns, n_neighbors=10,
hidden_dim=16, n_interactions=4, gaussian_expansion_dim=16,
**data-derived RBF range** (no v6/v7 pins),
no attention, no pre-encoder MLP, no early stopping, batch=1000, lr=5e-4,
val_split=0.3, weight_decay=1e-5, **xavier_normal init**, encoder
**v1** (no per-atom ReLU), 100 epochs.

## Why no `SoftmaxMLP` refactor was needed

Before drafting v8, we examined whether `SoftmaxMLP(num_layers=2)`
would produce the no-activation 2-Linear chain matching the reference,
or whether it would insert a ReLU between the two Linears (which would
break the rank-bottleneck).

Verified (`tests/test_classifier.py:TestRankBottleneckHead`, 3 tests):
- PyG `torch_geometric.nn.models.MLP` applies activations BETWEEN
  Linears, not after the last one.  With `num_layers=1`, MLP is a
  single Linear with no activation.
- `SoftmaxMLP(num_layers=2, hidden_channels=H, out_channels=C)` =
  `MLP(num_layers=1)` (single Linear in→H, no act) +
  `final_layer = nn.Sequential(nn.Linear(H, C), nn.Softmax)`
  = `Linear(in, H) → Linear(H, C) → softmax`, **no activation between**.
- SVD probe confirms num_layers=2, hidden_channels=2, out_channels=4
  produces logits of effective rank 2.

So the rank-bottleneck head is achievable with **existing CLI flags**,
no module rebuild needed.

## Probe scope: single seed (mirrors v3–v7)

Only seed 0.  Decision rule:

- v8 seed_00 ≲ 3.72 → rank-2 bottleneck doesn't help.  Either h_g=2 is
  wrong for villin (the abstract's "2D embedding" might be a separate
  visualization step, not the actual training architecture), or the
  remaining gap is methodological (train/val split, VAMP-2 numerical
  formulation, etc.).
- v8 seed_00 ~3.72–3.76 → marginal; consider 3 seeds before deciding.
- v8 seed_00 ≳ 3.76 → strong signal that the rank-2 head was the
  missing piece, proceed to 10-seed v8 array.

## What v8 does NOT change

- Initialization scheme (xavier_normal, kept from v4).
- RBF basis — data-derived range (kept from v4; v6/v7 ruled out fixed
  range / σ).
- Encoder structure — v1 (no per-atom ReLU; v5 ruled out v2).
- Residual connections in conv blocks (already on at
  `pygv/encoder/schnet.py:268`, matches reference).
- All other v4 settings.

## CLI plumbing

**No new code or flags committed for v8.**  The `--clf_num_layers` and
`--clf_hidden_dim` flags already exist (`pygv/pipe/args.py:138-141`)
and are wired through `master_pipeline.main` (`pygv/pipe/master_pipeline.py:
1131-1134`).

The only new addition is a regression test:

- `tests/test_classifier.py:TestRankBottleneckHead` (3 tests) —
  pins the rank-bottleneck behavior of `SoftmaxMLP(num_layers=2)`.
  If a future PyG release changes MLP activation semantics, these
  tests fail loudly.  All 3 passing as of 2026-05-05.

## Submission

**No module rebuild needed** (no new flags or constructor changes).
Just:

```
sbatch cluster_scripts/villin_repro_v8.sh
```

Single job (no `--array`).  Output:
`/mnt/hdd/experiments/villin_repro_v8/seed_00/`.  Estimated wall time:
~3.5 h on RTX 5090 (same architecture size as v4-v7).

## Outputs to compare against

Best Val VAMP-2 in:
  `/mnt/hdd/experiments/villin_repro_v8/seed_00/exp_villin_<TIMESTAMP>/logs/log_*.txt`
look for the last `New best model with score: <X.XXXX>` line.
