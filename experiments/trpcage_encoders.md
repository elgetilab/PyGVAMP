# Trp-cage — Encoder Comparison — Tracker

Apples-to-apples comparison of graph encoders (SchNet vs GIN vs ML3) on the
Trp-cage VAMP-2 benchmark, in two regimes: **de-tuned** (all encoders forced
into the SchNet repro's hyperparameters — single-variable encoder swap) and
**native** (each encoder gets its own preset architecture; only the
benchmark-invariants held fixed). Motivated by: GIN and ML3 are more
WL/spectral-expressive than SchNet, yet came out *worse* de-tuned — why?

## System Info

| Property | Value |
|----------|-------|
| Protein | Trp-cage (DESRES 2JOF) |
| Topology | `/mnt/hdd/data/trpcage/DESRES-Trajectory_2JOF-0-c-alpha/topol.pdb` |
| Trajectories | `/mnt/hdd/data/trpcage/DESRES-Trajectory_2JOF-0-c-alpha/2JOF-0-c-alpha/` |
| File pattern | `2JOF-0-c-alpha-*.dcd` (single long DESRES Cα trajectory) |
| Frames | **1,044,000** (0.2 ns/frame) |
| Selection | `name CA` (20 atoms) |
| Timestep | **`--timestep 0.2` MANDATORY** — DESRES DCD metadata reports 1 ps/frame but the physical step is 200 ps |
| Lag / states | 20 ns / 5 |
| Paper target | **4.79 ± 0.01** (Ghorbani 2022) |

## Held fixed across ALL runs (benchmark-invariants)
data, `name CA`, timestep 0.2, stride 1, lag 20 ns, n_states 5,
`--no_discover_states --max_retrains 0 --no_warm_start_retrains`,
epochs 100, **n_neighbors 7**, **val_split 0.3**, seeds 0–9.

## Results (perbatch_mean VAMP-2, aggregated over seeds)

| Encoder | Regime | Seeds | VAMP-2 | std | params | Δ vs SchNet | script |
|---------|--------|-------|--------|-----|--------|-------------|--------|
| **SchNet** | baseline | 10 | **4.6516** | ±0.0175 | ~7k | — | `trpcage_repro_v1_array.sh` |
| GIN | de-tuned | 10 | 4.5955 | ±0.0750 | ~7k | −0.056 | `trpcage_gin_v1_array.sh` |
| **GIN** | **native** | 10 | **4.6481** | ±0.0343 | 76,328 | ≈ tie | `trpcage_gin_native_v1_array.sh` |
| ML3 | de-tuned | 1 (smoke) | 4.6431 | — | ~46k | (1 seed) | `trpcage_ml3_v1_array.sh` |
| **ML3** | **native** | 10 | **4.5743** | ±0.0770 | 86,905 | **−0.077** | `trpcage_ml3_native_v1_array.sh` |

All native encoders sit ~0.14 below the paper's 4.79.

### Regime definitions
- **De-tuned** (matched to SchNet repro): hidden=16, output=16, n_interactions=4,
  gaussian=16, attention on, **no embedding**, **clf 1-layer / no-norm**,
  init=xavier_normal, lr=5e-4, wd=1e-5, batch=1000.
- **Native**: each encoder's own preset architecture — **batch_norm**, **embedding**,
  native width (GIN hidden=128/out=64; ML3 hidden=30/out=32/4-layer),
  init=kaiming, lr=1e-3, wd=1e-4. batch=1000 (native 32 infeasible at 1.04M
  frames → ~3 days/seed; batch_norm, not tiny batch, is the stabilizer).
  val_split held at 0.3 (not native 0.2) so the held-out set matches the baseline.

### Per-seed (native runs)
- **GIN native** best 7=4.6970 / worst 2=4.5662 — tight (±0.0343).
- **ML3 native** best 2=4.6494 / worst **6=4.4435** — wide (±0.0770); seeds 5,6 collapse to ~4.44.

## Conclusions

1. **GIN's deficit was DE-TUNING, not the encoder.** Native GIN recovers to
   SchNet parity (4.5955 → 4.6481) and *halves* its variance (0.075 → 0.034).
   The key stripped ingredient was **batch_norm** (sum-aggregation needs it).
2. **ML3 does NOT recover** in its native regime (4.5743 ± 0.0770) — lower mean
   *and* high seed instability (worst ~4.44), about where de-tuned GIN sat. Its
   deficit looks **intrinsic** (spectral/multi-layer machinery is harder to
   optimize reliably), not just tuning.
3. **More WL/spectral expressiveness does not help on this task.** Conformational
   states are distinguished by *geometry* (Cα distance map), which all three
   encoders capture via the RBF distance edge features; WL power (graph
   topology) has little to exploit. SchNet's geometric inductive bias + good
   conditioning ties or wins. Native ranking: **SchNet ≈ GIN > ML3**.
4. **Single-seed caution:** ML3 native seed 0 alone was 4.6483 (a lucky seed);
   the cross-seed sweep was required to reveal the instability.

## Gotchas
- **`--timestep 0.2` is mandatory** (DESRES metadata is wrong → 1000× off lag).
- **batch=32 (GIN/ML3 preset default) is infeasible** at 1.04M frames; use 1000.
- **Slow despite a tiny protein:** 1.04M frames → ~2–4 h/seed. Wall-clock scales
  with frame count, not protein size (cf. AT1R: 319 residues but only ~20k frames → ~6 min).
- **Memory monitoring:** watch `free` **MemAvailable**, not `scontrol FreeMem` —
  loading the 1M-frame trajectory fills RAM with reclaimable page cache that makes
  MemFree look critically low while MemAvailable stays ~200 G.

## Aggregate any run
```bash
python cluster_scripts/aggregate_trpcage_v1_array.py --root /mnt/hdd/experiments/<run_dir>
# run dirs: trpcage_repro_v1, trpcage_gin_v1, trpcage_gin_native_v1,
#           trpcage_ml3_v1, trpcage_ml3_native_v1
```

## The ~0.14 gap to the paper (4.65 vs 4.79) — what we ruled out

All native encoders land ~4.65–4.67, ~0.14 below Ghorbani 2022's 4.79. We tested
the leading explanations directly (frozen SchNet seed-0; tools:
`cluster_scripts/split_leak_test1_eval.py`, `split_leak_estimator_check.py`;
blocked-split feature in `pygv/dataset/splits.py`). **Three causes excluded:**

| Hypothesis | Test | Result | Verdict |
|---|---|---|---|
| Encoder / WL-expressiveness | native-regime sweep (this file) | flat: SchNet ≈ GIN > ML3 | ❌ not it |
| Train/val **split leakage** | Test 1: re-score full-val VAMP-2 under random vs **temporally-blocked** eval partition | concat: random **4.670**, blocked **4.756**, whole **4.668** | ❌ blocked ≥ random — opposite of leakage; if anything random under-scores |
| Per-batch **estimator bias** | batch-size sweep (shuffled), perbatch_mean vs concat | concat flat **4.67** for all batch; perbatch **falls** at small batch (4.67→3.95@64), **never reaches 4.79** | ❌ not it (and contradicts the "perbatch biased high" claim in `vampnet.evaluate()` docstring — ε-regularization makes it biased *low*) |

**Honest number: concat ≈ 4.67** (whole trajectory, batch-independent, < 5.0 ceiling ✓).
The residual ~0.12 to 4.79 is real but **not** dynamical, split, or estimator — most
likely featurization/architecture/training detail vs the reference, or what "4.79"
itself measures (best-seed? training score?). Not pursued further (diminishing
returns on a saturated score). The blocked-split feature (`--split_mode blocked`)
is retained as reusable infrastructure even though the test came back null.

## Open threads (not pursued)
- ML3 instability: worst seeds 5,6 → ~4.44 — init vs LR-conditioning of the spectral layers?
- The 0.12 residual gap — featurization/architecture diff vs the reference (no compute; not done).
