# Villin — Encoder comparison (native regime)

Three-encoder comparison on the Villin VAMP-2 benchmark, **native regime only**:
each encoder runs in its OWN preset architecture; only the benchmark-invariants
(data, lag, k, n_neighbors, val_split, seed set) are held fixed. This mirrors
the native half of [trpcage_encoders.md](trpcage_encoders.md) — the de-tuned
(width-matched) regime was dropped for Villin/NTL9 by decision (2026-07-01): the
question is "does each encoder at its own native settings match the paper's
SchNet config?", and if it doesn't, that itself is the finding.

| Property | Value |
|---|---|
| Data | DESRES 2F4K-0 Cα, 0.2 ns/frame, 35 Cα |
| Lag / k / n_neighbors | 20 ns / 4 states / 10 |
| Paper target | **3.78 ± 0.02** (Ghorbani 2022) |
| Seeds | 10 per encoder |

**Interpretation caveat:** the SchNet arm is the paper's *small* config
(hidden=16, no embedding, no norm, lr 5e-4) — i.e. our existing reproduction —
while GIN/ML3 native are their *larger* presets (hidden 128/30, embedding,
batch_norm, lr 1e-3). So a gap reflects each encoder's **whole native recipe**
vs paper-SchNet, not the encoder in isolation. (For the isolated single-variable
swap, see the de-tuned Trp-cage rows in trpcage_encoders.md.)

## Results (perbatch_mean VAMP-2, aggregated over 10 seeds)

| Encoder | Regime | VAMP-2 | std | Δ vs SchNet | Δ vs paper | run dir |
|---|---|---|---|---|---|---|
| **SchNet** | paper cfg | **3.6923** | ±0.0458 | — | −0.088 | `villin_repro_v11` |
| **GIN** | native | 3.5894 | ±0.0468 | −0.103 | −0.191 | `villin_gin_native_v1` |
| **ML3** | native | 3.5513 | ±0.0610 | −0.141 | −0.229 | `villin_ml3_native_v1` |

**Ranking: SchNet > GIN > ML3.** Neither alternative encoder in its native
config reaches paper-config SchNet on Villin.

### Per-seed (perbatch_mean @ best-concat epoch)

GIN native — `aggregate_villin_v11_array.py --root .../villin_gin_native_v1`:

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| perbatch | 3.5518 | 3.6353 | 3.5049 | 3.5725 | 3.6259 | 3.6042 | 3.6252 | 3.6225 | 3.5288 | 3.6232 |

ML3 native — `aggregate_villin_v11_array.py --root .../villin_ml3_native_v1`:

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| perbatch | 3.5379 | 3.5378 | 3.5122 | 3.5435 | 3.5782 | 3.5185 | 3.5232 | 3.5112 | 3.7157 | 3.5348 |

Both alternative encoders are stable on Villin (no collapse; tight spreads).
ML3 seed 8 (3.7157) is a high outlier but not a different regime.

## Conclusion

On Villin, **more WL/spectral expressiveness does not help** — same pattern as
Trp-cage native, but stronger: on Trp-cage native GIN *tied* SchNet (4.6481 vs
4.6516), whereas on Villin native GIN sits ~0.10 **below** SchNet. So GIN's
competitiveness with SchNet is **system-dependent**; ML3 trails on both systems.
Conformational states here are distinguished by geometry (Cα distance map),
which SchNet's distance-RBF inductive bias already captures — topological power
has little to exploit.

## Cross-system encoder picture (native regime)

| System | SchNet (paper cfg) | GIN native | ML3 native |
|---|---|---|---|
| Trp-cage | 4.6516 | 4.6481 (≈ tie) | 4.5743 |
| Villin | 3.6923 | 3.5894 (−0.10) | 3.5513 (−0.14) |
| NTL9 | 4.3459 | ~4.28 non-collapsed (unstable, see [ntl9_encoders.md](ntl9_encoders.md)) | OOM/collapse — no result |

## Gotcha
- `--timestep 0.2` is mandatory (DESRES DCD metadata reports 1 ps/frame; actual
  is 200 ps/frame — 1000× off lag otherwise).
