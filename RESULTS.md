# PyGVAMP — consolidated results

Every VAMP-2 number produced by the campaign, in one table per category, with a
pointer to the detailed tracker for each. **This is a summary, not a source** —
each row's per-seed values, protocol and caveats live in the linked document.

Last updated 2026-08-25.

---

## ⚠️ Read this before comparing any two numbers

1. **Two estimators are in use and they are NOT interchangeable.**
   The GraphVAMPNet rows are *per-batch* scores (`perbatch_mean`, the paper's
   methodology); the RevGraphVAMP rows are the *concatenated* validation score.
   Never compare a Rev row against a GraphVAMPNet row.
2. **Every score has a hard ceiling of k** (the number of states). A VAMP-2 above
   k is an estimator artefact, not a better model — this was a real defect, fixed
   2026-07-28 (commit cf3c4fe). All rows below were audited against their ceilings
   and are clean.
3. **VAMP-2 = 1.0 exactly is the degenerate value** (rank-1, no dynamics learned),
   not a low score. It has appeared here from a NaN-masked collapse.
4. **Cross-system comparisons of the same encoder are not meaningful** — k, lag and
   dataset size all differ.

---

## Category 1 — Reproductions

| System | Paper VAMP-2 | Ours (10 seeds) | Δ | Estimator | Status |
|---|---|---|---|---|---|
| Trp-cage | 4.79 ± 0.01 | **4.6516 ± 0.0175** | −0.138 | perbatch | below paper (7.9σ ours) |
| Villin | 3.78 ± 0.02 | **3.6923 ± 0.0458** | −0.088 | perbatch | closest (1.9σ ours) |
| NTL9 | 4.59 ± 0.09 | **4.3459 ± 0.0435** | −0.244 | perbatch | furthest below |
| Alanine dipeptide (Rev) | 4.41 ± 0.01 | **4.402 ± 0.244** | −0.008 | concat | **REPRODUCED** (seed variance ~24× theirs) |
| Aβ42 red (Rev) | 3.99 ± 0.002 | **3.9830 ± 0.0005** | −0.0070 | concat | **REPRODUCED** (job 849, post-ceiling-fix) |

**Open blocker:** the three GraphVAMPNet systems undershoot systematically, scaling
with system size (Villin −0.088 < Trp-cage −0.138 < NTL9 −0.244). Ruled out:
encoder choice, train/val split leakage, per-batch estimator bias, and VAMP-2
ceiling breaches. Remaining candidates: τ-normalisation, LR schedule / epoch budget,
edge-feature normalisation. See `claude/EXPERIMENT_CHECKLIST.md`.

Detail: `experiments/revgraphvamp_repro.md`, `experiments/ab42_red.md`,
`claude/{TRPCAGE_REPRO_V1_LOG,NTL9_REPRO_V2_LOG}.md`.
*(`claude/VILLIN_REPRO_V11_LOG.md` is **stale** — it documents only the single-seed
probe; the 10-seed number comes from `aggregate_villin_v11_array.py`.)*

---

## Category 2 — Encoder comparison (CLOSED, all null)

### Trp-cage — τ=20 ns, k=5, ceiling 5 → [`experiments/trpcage_encoders.md`](experiments/trpcage_encoders.md)

| Encoder | Regime | Seeds | VAMP-2 | Δ vs SchNet | params |
|---|---|---|---|---|---|
| **SchNet** | paper cfg | 10 | **4.6516 ± 0.0175** | — | 7,685 |
| GIN | de-tuned | 10 | 4.5955 ± 0.0750 | −0.056 | ~7k |
| GIN | native | 10 | 4.6481 ± 0.0343 | ≈ tie | 76,328 |
| ML3 | de-tuned | 10 | 4.6209 ± 0.0335 | −0.031 | ~46k |
| ML3 | native | 10 | 4.5743 ± 0.0770 | −0.077 | 86,905 |
| SchNet + angular | de-tuned | 10 | 4.6545 ± 0.0095 | **+0.0029 (null)** | 12,821 |

Angular arm: paired t = 0.729, CIs overlap. Bound: **no effect larger than ~0.02**.

### Villin — τ=20 ns, k=4, ceiling 4 → [`experiments/villin_encoders.md`](experiments/villin_encoders.md)

| Encoder | Regime | Seeds | VAMP-2 | Δ vs SchNet |
|---|---|---|---|---|
| **SchNet** | paper cfg | 10 | **3.6923 ± 0.0458** | — |
| GIN | native | 10 | 3.5894 ± 0.0468 | −0.103 |
| ML3 | native | 10 | 3.5513 ± 0.0610 | −0.141 |

### NTL9 — τ=200 ns, k=5, ceiling 5 → [`experiments/ntl9_encoders.md`](experiments/ntl9_encoders.md)

| Encoder | Regime | Seeds | VAMP-2 | Note |
|---|---|---|---|---|
| **SchNet** | paper cfg | 10 | **4.3459 ± 0.0435** | clean |
| GIN | native | 8 of 10 | ~4.28 ± 0.05 | **unstable — 2/10 collapsed to 1.0** |
| ML3 | native | 0 | — | **no usable run (CUDA-OOM 9/10 + 1 collapse)** |

### α3D — τ=50 ns, k=3, ceiling 3 → [`experiments/a3d_encoders.md`](experiments/a3d_encoders.md)

| Encoder | Seeds | VAMP-2 | 95% CI | % of ceiling | params |
|---|---|---|---|---|---|
| **SchNet** | 10 | **2.9633 ± 0.0063** | [2.9594, 2.9672] | 98.78% | 38,976 |
| **PaiNN** | 10 | **2.9637 ± 0.0060** | [2.9599, 2.9674] | 98.79% | 38,990 |

Δ = +0.0004, paired t = 0.203, paired CI [−0.0032, +0.0039], MDE ~0.005.
Parameter-matched single-variable swap. 0 collapses / 0 OOM / 0 NaN across 20 runs.
**⚠️ Both arms at 98.8% of ceiling — the null is partly about the task's lack of
headroom, not only the encoder. The seed effect dwarfs the encoder effect.**

### Verdict

**Five variants, three independent lines of evidence, all null.**

| line of evidence | variant | result |
|---|---|---|
| aggregation | GIN | ties Trp-cage, −0.10 Villin, unstable NTL9 |
| aggregation | ML3 | below everywhere, least stable |
| descriptor **information** | SchNet + angular features | null, ≤0.02 |
| descriptor **information** | k-NN truncation measurement | discarded geometry recoverable; 4-hop coverage 95–100% |
| **equivariant inductive bias** | PaiNN | null, ≤0.005 |

**For Cα-graph VAMPNets on these systems, the encoder is not the lever.**
Do not open a new encoder arm without first finding an operating point where the
baseline does *not* saturate. Truncation measurements: `claude/PAINN_SCOPE.md`.

---

## Category 3 — Multi-lag exploration

### GTT / FiP35 WW domain, seed 0 → [`experiments/gtt_lagsweep.md`](experiments/gtt_lagsweep.md)

| τ (ns) | VAMP-2 @k=10 | VAMP-2 @k=2 | k\* | t₂ | populations |
|---|---|---|---|---|---|
| 1 | 9.6652 | 1.9995 | 2 | 4.00 µs | 0.228 / 0.772 |
| 2 | 9.5801 | 1.9994 | 2 | 6.66 µs | 0.228 / 0.772 |
| 5 | 9.3455 | 1.9990 | 2 | 9.99 µs | 0.23 / 0.77 |
| 10 | 8.8415 | 1.9982 | 2 | 11.10 µs | 0.228 / 0.772 |
| 20 | 8.2231 | 1.9969 | 2 | **12.88 µs** | 0.229 / 0.771 |
| 50 | 6.3599 | 1.9921 | 2 | **12.61 µs** | 0.227 / 0.773 |

**k\*(τ) = 2 across a 50× range in τ** — FiP35 is two-state throughout.
The richer result: **t₂ plateaus at ~12.6–12.9 µs from τ ≈ 20 ns**, the
implied-timescale convergence test. t₂ is derived from validation VAMP-2
(σ₂ = √(VAMP-2 − 1)), *not* from the pipeline's ITS output, which uses a
non-contiguous subsample and is not time-calibrated.

⚠️ Three separate caps biased k\* en route (`--max_states` from above, the
`max(2,…)` floor from below, `--max_retrains` truncating mid-descent). Always check
a k\* against the cap that produced it. The descent *path* is not reproducible
under a fixed seed; only the endpoint is. **Single seed — no error bars.**

### α3D k\* probe (job 921) → [`experiments/a3d_encoders.md`](experiments/a3d_encoders.md)

Descended 10→9→8→7→6→5→4→3 and **converged at k\*=3** (7 of 10 rounds, not
cap-truncated), populations **0.469 / 0.362 / 0.169**. VAMP-2: 9.0016 @k=10,
4.9290 @k=5, 3.9365 @k=4, 2.9810 @k=3.
**The only non-two-state system in the campaign.**

---

## Where the raw runs live

`/mnt/hdd/experiments/` — one directory per campaign, e.g.
`trpcage_repro_v1/`, `a3d_encoder_v1/{schnet,painn}/`, `gtt_lagsweep_v1/seed_00/`,
`trpcage_angular_pretest/`. Per-seed scores are in
`<exp>/training/*/*/models/training_complete.json` (`best_score`).

**Before trusting any run, audit it** — four distinct failures in this project
exited 0 with complete-looking output. See `claude/PORTING.md` §5 for the checklist.

Aggregate a campaign with:
```bash
python cluster_scripts/aggregate_trpcage_v1_array.py --root /mnt/hdd/experiments/<run_dir>
```
(takes perbatch at the best-**concat** epoch, avoiding max-over-epoch selection bias)

## Document map

| lane | document |
|---|---|
| Master status & protocol | `claude/EXPERIMENT_CHECKLIST.md` |
| Porting to another machine | `claude/PORTING.md` |
| PaiNN design + truncation measurements | `claude/PAINN_SCOPE.md` |
| Requeue/resume design | `claude/RESUME_PLAN_2026-08-03.md` |
| Per-system detail | `experiments/*.md` |
