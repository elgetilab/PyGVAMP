# TRPCAGE_GIN_V1_LOG.md — Trp-cage GIN encoder, apples-to-apples vs SchNet

Encoder-comparison run: **identical to the Trp-cage SchNet reproduction
(`trpcage_repro_v1`) in every hyperparameter, data selection, lag, and seed
set — only the encoder changed** (`--model schnet → --model gin`). Goal: does
swapping SchNet for GIN (same config) change VAMP-2 on Trp-cage?

See [TRPCAGE_REPRO_V1_LOG.md](TRPCAGE_REPRO_V1_LOG.md) for the SchNet baseline.

## Setup

- Script: `cluster_scripts/trpcage_gin_v1_array.sh` (SchNet repro script with
  `--model gin`; all GIN-preset defaults overridden to the repro values —
  hidden=16, output=16, n_interactions=4, n_neighbors=7, gaussian=16,
  attention on, no embedding, clf 1-layer/no-norm, init xavier_normal, lr=5e-4,
  wd=1e-5, 100 epochs, batch=1000, val_split=0.3, lag=20 ns, 5 states, no state
  discovery, no retrains). SLURM-script knobs, not a new preset class.
- Data: DESRES 2JOF Cα, 0.2 ns/frame, 1.04M frames.
- Run: job 602, `--array=0-9%2`, 2026-06-04→05, ~17.5 h wall. All 10 seeds
  exit 0; full pipeline (training **and** analysis) completed — the deployed
  in-training-OOM + recursive-lookup fixes held on a full run.
- Model: `VAMPNet`/`GINEncoder`, 4 layers, 7321 params.

## Result

Aggregated from training logs:
`python cluster_scripts/aggregate_trpcage_v1_array.py --root /mnt/hdd/experiments/trpcage_gin_v1`

### Headline (cross-seed, n=10)

| | perbatch_mean | best concat | Δ vs paper (4.79 ± 0.01) |
|---|---|---|---|
| **GIN** | **4.5955 ± 0.0750** | 4.5980 ± 0.0697 | −0.195 |
| SchNet baseline | 4.6516 ± 0.0175 | — | −0.138 |

**GIN lands ~0.056 below SchNet and is ~4× noisier across seeds.** Swapping
the encoder (same config) did not help on Trp-cage — slightly worse. Both miss
the paper.

### Per-seed table (perbatch_mean @ best-concat epoch)

| seed | epoch | best concat | perbatch_mean | perbatch_std |
|---|---|---|---|---|
| 0 | 94 | 4.6314 | 4.6292 | 0.1277 |
| 1 | 98 | 4.6402 | 4.6387 | 0.1274 |
| 2 | 83 | 4.6081 | 4.6096 | 0.1174 |
| 3 | 90 | 4.6133 | 4.6141 | 0.1352 |
| 4 | 96 | 4.6083 | 4.6092 | 0.1170 |
| 5 | 100 | 4.6158 | 4.6118 | 0.1265 |
| 6 | 99 | 4.6366 | 4.6347 | 0.1278 |
| 7 | 94 | 4.6069 | 4.6059 | 0.1204 |
| **8** | **99** | **4.4026** | **4.3846** | **0.2208** |
| 9 | 85 | 4.6168 | 4.6174 | 0.1301 |

### Seed 8 — slow convergence, NOT instability

Seed 8 drives most of the spread. Its training curve is **monotonic but slow**:
3.70 (ep1) → 4.17 (ep11) → 4.30 (ep41) → 4.40 (ep100), **still rising at epoch
100** (best concat 4.4026 at ep99; no plateau/collapse, no NaN). The other 9
seeds plateaued at 4.61–4.64 by ~ep85–100. So seed 8 is an **under-converged**
run from an unlucky init, not a divergence — it would likely have kept climbing
with more epochs.

Diagnostic only (NOT the reported number, no seed dropping): the other 9 seeds
average ~4.619 ± 0.012 — tight, and still ~0.03 below the SchNet baseline. So
the GIN < SchNet gap looks real-but-small; seed 8 inflates the variance, not the
central conclusion.

## Verdict

- **GIN ≈ slightly worse than SchNet on Trp-cage** at matched config
  (−0.056 in the mean), and **less reliable to converge in 100 epochs**
  (1/10 seeds under-converged; 4× cross-seed SD).
- Both encoders miss the paper (4.79); GIN does not close the Trp-cage gap.
- Follow-ups worth considering: longer training / LR schedule to see if GIN's
  slow seeds catch up; whether GIN's higher per-epoch variance is intrinsic.

Analysis artifacts (transition matrices, ITS, CK, state structures) were
generated for all seeds — the ITS/CK subsampling caveat in
[ANALYSIS_RECOVERY_AND_OPEN_ITEMS.md](ANALYSIS_RECOVERY_AND_OPEN_ITEMS.md)
(open item #1) applies here too.
