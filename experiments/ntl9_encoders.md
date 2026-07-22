# NTL9 — Encoder comparison (native regime)

Native-regime three-encoder comparison on NTL9, analogous to
[villin_encoders.md](villin_encoders.md) and the native half of
[trpcage_encoders.md](trpcage_encoders.md). **This is a messy result: native
GIN is unstable on NTL9, and native ML3 did not produce a usable run at all.**
Reported as findings, not clean numbers (decision 2026-07-14: leave training as
is, do not sink another ~2–3 weeks of compute into reruns).

| Property | Value |
|---|---|
| Data | DESRES NTL9-{0,1,2,3} combined, 0.2 ns/frame, 39 Cα, 14.7M frames (1.11 ms) |
| Lag / k / n_neighbors | 200 ns / 5 states / 10 |
| Paper target | **4.59 ± 0.09** (Ghorbani 2022) |
| SchNet baseline | **4.3459 ± 0.0435** (`ntl9_repro_v2`) |
| Seeds | 10 per encoder (as submitted) |

## Results (perbatch_mean VAMP-2)

| Encoder | Regime | Result | run dir | Status |
|---|---|---|---|---|
| **SchNet** | paper cfg | **4.3459 ± 0.0435** | `ntl9_repro_v2` | clean (10 seeds) |
| **GIN** | native | **~4.28 ± 0.05** (8 non-collapsed) | `ntl9_gin_native_v1` | **unstable: 2/10 collapsed** |
| **ML3** | native | — | `ntl9_ml3_native_v1` | **no usable run (OOM/collapse)** |

### GIN native — unstable (job 723)

All 10 seeds trained, but **2/10 (seeds 0 and 7) collapsed to VAMP-2 = 1.0** —
degenerate single-state solutions (native lr=1e-3 is too hot for NTL9's 14.7M
frames). The 8 non-collapsed seeds:

| seed | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|
| perbatch | 4.3492 | 4.3148 | 4.1971 | 4.2622 | 4.2119 | 4.3158 | 4.2679 | 4.3520 |

Non-collapsed mean ≈ **4.284 ± 0.055**, i.e. **~0.06 below** SchNet (4.3459) —
same "native GIN < paper-SchNet" direction as Villin. Reported the raw 10-seed
mean (3.63 ± 1.39) is meaningless — the two 1.0 seeds dominate it; the finding
is the **20% collapse rate + non-collapsed ≈ slightly below SchNet**.

Aggregator: `aggregate_ntl9_v1_array.py --root .../ntl9_gin_native_v1` (VAMP-2
parsed from training logs — see analysis note below).

### ML3 native — no usable run (job 725)

- seed 0: trained 100 epochs but **collapsed to VAMP-2 = 1.0**.
- **seeds 1–9: CUDA out-of-memory in the encoder at epoch 1.** Running 4
  concurrent (`%4`) on `shard:2` (~8 GB each) starved native ML3's heavier
  spectral model; the pipeline caught it gracefully ("Training failed: CUDA out
  of memory in encoder") and exited 0, so the jobs *looked* done but produced no
  training. **0 usable data points.**
- To recover this cell would need `shard:2 → shard:4` (and/or smaller batch) +
  ~2–3 weeks of rerun. Left undone by decision; recorded as "native ML3 OOMs at
  the benchmark's data scale on the available shard budget."

## Two OOM failure modes seen on NTL9 native (don't conflate)

1. **Host-RAM OOM, exit 137 (GIN native, all seeds).** Training finished and
   saved `best_model.pt`, but the *in-training post-analysis block* (training.py,
   full-trajectory rebuild, no subsampling) blew past `--mem=32G` and killed the
   process before PHASE 3 → all analysis/ dirs empty. Same bug as the SchNet v2
   run. VAMP-2 scores survive in the training logs. **Recovered** via
   `ntl9_gin_native_analysis_array.sh` (`--only_analysis --resume`, 50k-frame
   subsample, mem 120G) — produces the ITS/CK/state/report artifacts without
   retraining.
2. **GPU CUDA-OOM (ML3 native, seeds 1–9).** A different failure: the encoder
   itself didn't fit in VRAM under `shard:2` × 4-concurrent. Not recoverable
   without a rerun at higher shard budget.

## Conclusion

On NTL9, native GIN is **unstable** (20% seed collapse) and, where it converges,
lands **slightly below** paper-config SchNet — consistent with Villin. Native
ML3 could not be evaluated at NTL9's data/VRAM scale under the shard budget.
**Neither alternative encoder beats SchNet on NTL9**; the practical story is
that SchNet's small, well-conditioned config is the most *reliable* on the
largest system. Cross-system native picture in
[villin_encoders.md](villin_encoders.md).

## Gotcha
- `--timestep 0.2` is mandatory (DESRES DCD metadata 1 ps/frame; actual 200).
