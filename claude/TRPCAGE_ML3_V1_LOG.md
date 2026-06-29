# TRPCAGE_ML3_V1_LOG.md — Trp-cage ML3 encoder, apples-to-apples vs SchNet/GIN

Third encoder in the Trp-cage comparison: **identical to the SchNet reproduction
(`trpcage_repro_v1`) in data selection, lag, seed set, and training schedule —
only the encoder changed** (`--model schnet → --model ml3`). Goal: does the ML3
encoder (spectral machinery) change VAMP-2 on Trp-cage at matched embedding
width/depth?

See [TRPCAGE_REPRO_V1_LOG.md](TRPCAGE_REPRO_V1_LOG.md) for the SchNet baseline
and [TRPCAGE_GIN_V1_LOG.md](TRPCAGE_GIN_V1_LOG.md) for GIN.

## Setup

- Script: `cluster_scripts/trpcage_ml3_v1_array.sh` (SchNet repro script with
  `--model ml3`). **Width-matched, NOT param-matched:** ML3 reads its own dims,
  so `--ml3_node/edge/hidden/output_dim 16` and `--ml3_num_layers 4` set the VAMP
  bottleneck (output=16) and depth (4) to the SchNet/GIN values; the rest of the
  ML3 preset (nout1/nout2, edge_mode, nfreq, recfield, `ml3_use_attention=True`)
  has no SchNet analog and is left at preset. Shared knobs match the baseline:
  n_neighbors=7, gaussian=16, no embedding, clf 1-layer/no-norm, init
  xavier_normal, lr=5e-4, wd=1e-5, 100 epochs, batch=1000, val_split=0.3, lag=20
  ns, 5 states, no state discovery, no retrains. SLURM-script knobs, not a new
  preset class.
- Exposing `--ml3_*` on the pipeline CLI was a new capability (commit 73682d7);
  all 10 seeds ran on the working-tree code via `PYGVAMP_SRC_OVERRIDE`.
- Data: DESRES 2JOF Cα, 0.2 ns/frame, 1.04M frames.
- Model: `VAMPNet`/ML3 encoder, 4 layers, **46,365 params** (vs ~7.3k for
  SchNet/GIN — ML3 carries fixed spectral machinery; the control is on embedding
  dim + depth, not parameter count).
- Runs: seed 0 = job 613 (2026-06-05, ~4 h); seeds 1–9 = job 696, `--array=1-9%2`
  (2026-06-26→27, ~25 h wall). All 10 seeds exit 0, 100 epochs, full pipeline
  (training **and** analysis) completed. Default `split_mode=random` throughout —
  identical `torch.Generator().manual_seed(seed)` split as seed 0 (the blocked-
  split feature added in 6f4bcab is default-off and not invoked here), so all ten
  runs are on one consistent pipeline.

## Result

Aggregated:
`python cluster_scripts/aggregate_trpcage_v1_array.py --root /mnt/hdd/experiments/trpcage_ml3_v1`

### Headline (cross-seed, n=10)

| | perbatch_mean | best concat | Δ vs paper (4.79 ± 0.01) |
|---|---|---|---|
| **ML3** | **4.6209 ± 0.0335** | 4.6276 ± 0.0365 | −0.169 |
| SchNet baseline | 4.6516 ± 0.0175 | — | −0.138 |
| GIN | 4.5955 ± 0.0750 | 4.5980 ± 0.0697 | −0.195 |

**ML3 lands between SchNet and GIN** — ~0.031 below SchNet's mean, ~0.025 above
GIN's — with intermediate cross-seed variance (~2× SchNet's SD, ~½ GIN's). At
matched embedding width/depth, the extra ML3 machinery did **not** help on
Trp-cage. All three encoders miss the paper (4.79); ML3 does not close the gap.

### Per-seed table (perbatch_mean @ best-concat epoch)

| seed | epoch | best concat | perbatch_mean | perbatch_std |
|---|---|---|---|---|
| 0 | 87 | 4.6562 | 4.6431 | 0.1740 |
| 1 | 52 | 4.6407 | 4.6359 | 0.1342 |
| 2 | 71 | 4.6393 | 4.6352 | 0.1255 |
| 3 | 80 | 4.6286 | 4.6179 | 0.1879 |
| **4** | 86 | **4.5600** | **4.5585** | 0.1082 |
| 5 | 95 | 4.6664 | 4.6521 | 0.1739 |
| 6 | 46 | 4.6472 | 4.6360 | 0.2011 |
| **7** | 83 | **4.5629** | **4.5602** | 0.1147 |
| 8 | 71 | 4.6337 | 4.6341 | 0.1584 |
| 9 | 60 | 4.6405 | 4.6358 | 0.1700 |

Seeds 4 and 7 are the low outliers (~4.56); the other 8 cluster tightly at
4.63–4.67. Unlike GIN's seed 8, no single seed dominates the spread and there is
no under-convergence signature (best-concat epochs span 46–95, all plateaued).

## Verdict

- **ML3 ≈ slightly worse than SchNet, slightly better than GIN** on Trp-cage at
  matched embedding width/depth (−0.031 vs SchNet in the mean), despite ~6× the
  parameters. The added spectral machinery buys nothing here.
- All three encoders miss the paper (4.79) by 0.14–0.20. The gap is consistent
  across architectures — pointing to a data/normalization issue rather than
  encoder expressiveness (cf. the Villin τ-normalization finding).

Analysis artifacts (transition matrices, ITS, CK, state structures) were
generated for all seeds — the ITS/CK subsampling caveat applies here too
(absolute timescales aren't time-calibrated; populations are fine).
